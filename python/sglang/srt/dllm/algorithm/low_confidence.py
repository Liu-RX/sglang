from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self.full_sequence:
            return self._run_full_sequence(model_runner, forward_batch)

        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]
                if self.shift_logits:
                    curr_logits = torch.cat([curr_logits[:1], curr_logits[:-1]], dim=0)

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph

    def _run_full_sequence(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        seq_lens = (
            forward_batch.extend_seq_lens_cpu
            if forward_batch.extend_seq_lens_cpu is not None
            else forward_batch.seq_lens_cpu.tolist()
        )
        seq_starts = []
        offset = 0
        for seq_len in seq_lens:
            seq_starts.append(offset)
            offset += seq_len

        start_list = []
        for batch_id, seq_len in enumerate(seq_lens):
            seq_start = seq_starts[batch_id]
            curr_input_ids = forward_batch.input_ids[seq_start : seq_start + seq_len]
            mask_positions = torch.nonzero(
                curr_input_ids == self.mask_id, as_tuple=True
            )[0]
            start_list.append(
                mask_positions[0].item() if mask_positions.numel() > 0 else seq_len
            )

        if not (forward_batch.input_ids == self.mask_id).any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        steps = self.block_size
        timesteps = torch.linspace(
            1, 1e-3, steps + 1, device=forward_batch.input_ids.device
        )
        can_run_cuda_graph = False
        logits_output = None
        for i in range(steps):
            if not (forward_batch.input_ids == self.mask_id).any():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            for batch_id, seq_len in enumerate(seq_lens):
                seq_start = seq_starts[batch_id]
                seq_end = seq_start + seq_len
                curr_input_ids = forward_batch.input_ids[seq_start:seq_end]
                curr_logits = logits_output.full_logits[seq_start:seq_end]
                if self.shift_logits:
                    curr_logits = torch.cat([curr_logits[:1], curr_logits[:-1]], dim=0)

                mask_index = curr_input_ids == self.mask_id
                if not mask_index.any():
                    continue

                mask_logits = curr_logits[mask_index]
                probs = F.softmax(mask_logits, dim=-1)
                x0 = torch.argmax(probs, dim=-1)
                confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

                t = timesteps[i]
                s = timesteps[i + 1]
                num_mask_token = mask_index.sum().item()
                num_transfer = (
                    int(num_mask_token * (1 - s / t))
                    if i < steps - 1
                    else int(num_mask_token)
                )
                if num_transfer <= 0:
                    continue

                full_confidence = torch.full_like(
                    curr_input_ids, -torch.inf, dtype=curr_logits.dtype
                )
                full_confidence[mask_index] = confidence
                _, transfer_index = torch.topk(full_confidence, num_transfer)

                x_ = torch.full_like(curr_input_ids, self.mask_id)
                x_[mask_index] = x0
                curr_input_ids[transfer_index] = x_[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids_list = []
        for batch_id, seq_len in enumerate(seq_lens):
            seq_start = seq_starts[batch_id]
            curr_input_ids = forward_batch.input_ids[seq_start : seq_start + seq_len]
            next_token_ids_list.append(curr_input_ids[start_list[batch_id] :])

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
