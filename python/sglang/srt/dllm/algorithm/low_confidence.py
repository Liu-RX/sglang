import dataclasses
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


SPIFFY_GRAPH_D1 = [{"indices": [0], "level": 1, "parents": [-1]}]
SPIFFY_GRAPH_D2 = [
    {"indices": [0], "level": 1, "parents": [-1]},
    {"indices": [0, 1], "level": 2, "parents": [0]},
]
SPIFFY_GRAPH_D3 = [
    {"indices": [0], "level": 1, "parents": [-1]},
    {"indices": [0, 1], "level": 2, "parents": [0]},
    {"indices": [0, 1, 2], "level": 3, "parents": [1]},
]
SPIFFY_GRAPH_D7 = [
    {"indices": [0], "level": 1, "parents": [-1]},
    {"indices": [0, 1], "level": 2, "parents": [0]},
    {"indices": [0, 2], "level": 2, "parents": [0]},
    {"indices": [0, 3], "level": 2, "parents": [0]},
    {"indices": [0, 1, 2], "level": 3, "parents": [1, 2]},
    {"indices": [0, 1, 3], "level": 3, "parents": [1, 3]},
    {"indices": [0, 1, 2, 3], "level": 4, "parents": [4, 5]},
]
SPIFFY_GRAPH_D8 = [
    {"indices": [0], "level": 1, "parents": [-1]},
    {"indices": [0, 1], "level": 2, "parents": [0]},
    {"indices": [0, 2], "level": 2, "parents": [0]},
    {"indices": [0, 3], "level": 2, "parents": [0]},
    {"indices": [0, 1, 2], "level": 3, "parents": [1, 2]},
    {"indices": [0, 1, 3], "level": 3, "parents": [1, 3]},
    {"indices": [0, 1, 2, 3], "level": 4, "parents": [4, 5]},
    {"indices": [0, 1, 2, 3, 4], "level": 5, "parents": [6]},
]

_SPIFFY_MAX_SPEC = 5
_SPEC_TO_GRAPH = {
    1: SPIFFY_GRAPH_D1,
    2: SPIFFY_GRAPH_D2,
    3: SPIFFY_GRAPH_D3,
    4: SPIFFY_GRAPH_D7,
    5: SPIFFY_GRAPH_D8,
}


def _get_spiffy_graph(n_spec: int) -> List[Dict]:
    return _SPEC_TO_GRAPH[min(n_spec, _SPIFFY_MAX_SPEC)]


def _dynamic_verify_batch_size(n_spec: int, global_vbs: int) -> int:
    if n_spec == 0:
        return 1
    natural = 2 if n_spec <= 2 else 4
    return min(natural, global_vbs)


def _max_prob_and_token(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_logits, tokens = logits.max(dim=-1)
    probs = torch.exp(max_logits - torch.logsumexp(logits, dim=-1))
    return probs, tokens


def _gather_token_probs_from_logits(
    logits: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    selected_logits = logits.index_select(0, positions)
    token_probs = F.softmax(selected_logits, dim=-1).gather(
        dim=-1,
        index=tokens.unsqueeze(-1),
    )
    return token_probs.squeeze(-1)


def _build_verify_candidates(
    x_base: torch.Tensor,
    spec_positions: List[int],
    spec_confidences: List[float],
    mask_id: int,
    global_vbs: int,
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    n_spec = len(spec_positions)
    assert n_spec > 0

    vbs = _dynamic_verify_batch_size(n_spec, global_vbs)
    order = sorted(range(n_spec), key=lambda i: spec_confidences[i], reverse=True)
    sorted_pos = [spec_positions[o] for o in order]

    tier_counts = []
    for tier in range(vbs):
        frac = (vbs - 1 - tier) / (vbs - 1)
        tier_counts.append(math.ceil(n_spec * frac))

    seen = set()
    unique_counts = []
    for count in tier_counts:
        if count not in seen:
            seen.add(count)
            unique_counts.append(count)
    if 0 not in seen:
        unique_counts.append(0)

    candidates = []
    candidate_spec_positions = []
    for count in unique_counts:
        cand = x_base.clone()
        active_positions = sorted_pos[:count]
        for pos in sorted_pos[count:]:
            cand[pos] = mask_id
        candidates.append(cand)
        candidate_spec_positions.append(list(active_positions))

    return candidates, candidate_spec_positions


def _select_best_verified_candidate(
    logits_verify: List[torch.Tensor],
    candidates: List[torch.Tensor],
    candidate_spec_positions: List[List[int]],
    threshold: float,
    mask_id: int,
) -> Tuple[torch.Tensor, List[int], int]:
    for cand_idx, (cand_seq, spec_pos_list) in enumerate(
        zip(candidates, candidate_spec_positions)
    ):
        if len(spec_pos_list) == 0:
            return cand_seq, [], cand_idx

        pos = torch.tensor(spec_pos_list, dtype=torch.long, device=cand_seq.device)
        tok = cand_seq[pos]
        token_probs = _gather_token_probs_from_logits(
            logits_verify[cand_idx],
            pos,
            tok,
        )
        if bool(((tok != mask_id) & (token_probs >= threshold)).all().item()):
            return cand_seq, spec_pos_list, cand_idx

    return candidates[-1], [], len(candidates) - 1


def _build_spiffy_verify_candidates(
    x_base: torch.Tensor,
    spec_positions: List[int],
    spec_tokens: List[int],
    spec_confidences: List[float],
    mask_id: int,
) -> Tuple[List[torch.Tensor], List[List[int]], List[Dict], List[int], torch.Tensor]:
    n_spec = len(spec_positions)
    assert n_spec > 0

    order = sorted(range(n_spec), key=lambda i: spec_confidences[i], reverse=True)
    n_use = min(n_spec, _SPIFFY_MAX_SPEC)
    discarded_pos = [spec_positions[order[i]] for i in range(n_use, n_spec)]

    order = order[:n_use]
    sorted_pos = [spec_positions[o] for o in order]
    sorted_tok = [spec_tokens[o] for o in order]
    graph_structure = _get_spiffy_graph(n_use)

    baseline_seq = x_base.clone()
    for pos in sorted_pos + discarded_pos:
        baseline_seq[pos] = mask_id

    candidates = []
    cand_spec_pos = []
    for node in graph_structure:
        cand = baseline_seq.clone()
        active_pos = []
        for idx in node["indices"]:
            if idx < n_use:
                pos = sorted_pos[idx]
                cand[pos] = sorted_tok[idx]
                active_pos.append(pos)
        candidates.append(cand)
        cand_spec_pos.append(active_pos)

    return candidates, cand_spec_pos, graph_structure, discarded_pos, baseline_seq


def _spiffy_accept_reject(
    candidates: List[torch.Tensor],
    cand_spec_pos: List[List[int]],
    graph_structure: List[Dict],
    logits_verify: List[torch.Tensor],
    mask_id: int,
    threshold: float,
    baseline_seq: torch.Tensor,
) -> Tuple[torch.Tensor, List[int], int]:
    node_accepted = [False] * len(graph_structure)
    best_node_idx: Optional[int] = None
    max_level = 0

    def one_step_target(node_idx: int) -> torch.Tensor:
        logits = logits_verify[node_idx]
        cand = candidates[node_idx].clone()
        mask_pos = cand == mask_id
        if not mask_pos.any():
            return cand
        mask_indices = torch.nonzero(mask_pos, as_tuple=True)[0]
        confidence, predicted = _max_prob_and_token(logits[mask_indices])
        best_idx = confidence.argmax()
        transfer = mask_indices[best_idx]
        cand[transfer] = predicted[best_idx]
        return cand

    target_cache: Dict[int, torch.Tensor] = {}

    def get_target(node_idx: int) -> torch.Tensor:
        if node_idx not in target_cache:
            target_cache[node_idx] = one_step_target(node_idx)
        return target_cache[node_idx]

    for k, node in enumerate(graph_structure):
        level = node["level"]
        filled_k = set(cand_spec_pos[k])

        for parent in node["parents"]:
            if parent == -1:
                extra_pos = list(filled_k)
                if extra_pos:
                    pos = torch.tensor(
                        extra_pos, dtype=torch.long, device=candidates[k].device
                    )
                    tok = candidates[k][pos]
                    conf_ok = bool(
                        (
                            _gather_token_probs_from_logits(
                                logits_verify[0],
                                pos,
                                tok,
                            )
                            >= threshold
                        )
                        .all()
                        .item()
                    )
                else:
                    conf_ok = True
                if conf_ok:
                    node_accepted[k] = True
                    if level > max_level:
                        max_level = level
                        best_node_idx = k
                break

            if not node_accepted[parent]:
                continue

            filled_p = set(cand_spec_pos[parent])
            extra_pos = list(filled_k - filled_p)
            target = get_target(parent)
            if extra_pos:
                pos = torch.tensor(
                    extra_pos, dtype=torch.long, device=candidates[k].device
                )
                struct_ok = bool((candidates[k][pos] == target[pos]).all().item())
            else:
                struct_ok = True
            if not struct_ok:
                continue
            if extra_pos:
                tok = candidates[k][pos]
                conf_ok = bool(
                    (
                        _gather_token_probs_from_logits(
                            logits_verify[parent],
                            pos,
                            tok,
                        )
                        >= threshold
                    )
                    .all()
                    .item()
                )
            else:
                conf_ok = True
            if conf_ok:
                node_accepted[k] = True
                if level > max_level:
                    max_level = level
                    best_node_idx = k
                break

    if best_node_idx is None:
        return baseline_seq, [], 0

    return candidates[best_node_idx], list(cand_spec_pos[best_node_idx]), best_node_idx


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.alg = config.algorithm_config.get("alg", "entropy").lower()
        if self.alg not in ("entropy", "origin", "confidence_threshold", "topk_margin"):
            raise ValueError(
                "LowConfidence alg supports "
                '"entropy", "origin", "confidence_threshold", or "topk_margin".'
            )
        self.speculative_decoding = config.algorithm_config.get(
            "speculative_decoding", False
        )
        self.verify_batch_size = config.algorithm_config.get("verify_batch_size", 1)
        if isinstance(self.verify_batch_size, str):
            self.verify_batch_size = self.verify_batch_size.lower()
            if self.verify_batch_size != "spiffy":
                self.verify_batch_size = int(self.verify_batch_size)
        if self.verify_batch_size not in (1, 2, "spiffy"):
            raise ValueError(
                "LowConfidence speculative decoding supports "
                'verify_batch_size: 1, 2, or "spiffy".'
            )
        self.speculative_confidence_threshold = config.algorithm_config.get(
            "speculative_confidence_threshold", 0.5
        )
        self.speculative_decoding_mode = config.algorithm_config.get(
            "speculative_decoding_mode", "greedy"
        ).lower()
        if self.speculative_decoding_mode not in ("greedy", "confidence"):
            raise ValueError(
                "LowConfidence speculative_decoding_mode supports "
                '"greedy" or "confidence".'
            )
        self.self_speculative_confidence_threshold = config.algorithm_config.get(
            "self_speculative_confidence_threshold",
            self.speculative_confidence_threshold,
        )
        self.confidence_speculative_threshold = config.algorithm_config.get(
            "confidence_speculative_threshold", 0.8
        )
        self.num_speculate_tokens = config.algorithm_config.get(
            "num_speculate_tokens", 3
        )

    def _profile_begin(self, seq_lens: List[int]) -> Optional[Dict]:
        if not os.environ.get("SGLANG_DLLM_PROFILE_PATH"):
            return None
        return {
            "pid": os.getpid(),
            "start_time": time.time(),
            "elapsed_sec": 0.0,
            "seq_lens": [int(x) for x in seq_lens],
            "batch_size": len(seq_lens),
            "verify_batch_size": self.verify_batch_size,
            "speculative_decoding": self.speculative_decoding,
            "speculative_decoding_mode": self.speculative_decoding_mode,
            "steps": 0,
            "normal_steps": 0,
            "batched_vbs2_steps": 0,
            "batched_vbs2_spec_positions": 0,
            "forward": {},
        }

    def _profile_finish(self, profile: Optional[Dict]):
        if profile is None or profile.get("_finished"):
            return
        profile["_finished"] = True
        profile["elapsed_sec"] = time.time() - profile["start_time"]
        path = os.environ.get("SGLANG_DLLM_PROFILE_PATH")
        if not path:
            return
        with open(path, "a") as f:
            f.write(json.dumps(profile) + "\n")

    def _profile_forward(
        self,
        label: str,
        token_count: int,
        branch_count: int,
        fn,
    ):
        profile = getattr(self, "_profile_state", None)
        if profile is None:
            return fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        stats = profile["forward"].setdefault(
            label,
            {"calls": 0, "time_sec": 0.0, "tokens": 0, "branches": 0},
        )
        stats["calls"] += 1
        stats["time_sec"] += elapsed
        stats["tokens"] += int(token_count)
        stats["branches"] += int(branch_count)
        return out

    def _profile_skip_candidate_batch(self, reason: str):
        profile = getattr(self, "_profile_state", None)
        if profile is None:
            return
        skips = profile.setdefault("candidate_batch_skips", {})
        skips[reason] = skips.get(reason, 0) + 1

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self.full_sequence:
            old_profile = getattr(self, "_profile_state", None)
            try:
                return self._run_full_sequence(model_runner, forward_batch)
            except Exception:
                self._profile_state = old_profile
                raise

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

    def _shift_logits_if_needed(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self.shift_logits:
            return logits

        shifted_logits = logits.clone()
        for start, seq_len in self._iter_sequences(forward_batch):
            end = start + seq_len
            shifted_logits[start:end] = torch.cat(
                [logits[start : start + 1], logits[start : end - 1]], dim=0
            )
        return shifted_logits

    def _forward_logits_with_input_ids(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        profile_label: str = "normal_forward",
    ) -> Tuple[torch.Tensor, LogitsProcessorOutput, bool]:
        original_input_ids = forward_batch.input_ids
        try:
            forward_batch.input_ids = input_ids
            out = self._profile_forward(
                profile_label,
                int(input_ids.numel()),
                1,
                lambda: model_runner.forward(forward_batch, pp_proxy_tensors=None),
            )
        finally:
            forward_batch.input_ids = original_input_ids

        logits = self._shift_logits_if_needed(
            out.logits_output.full_logits, forward_batch
        )
        return logits, out.logits_output, out.can_run_graph

    def _build_expanded_forward_batch(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        candidate_input_ids: List[torch.Tensor],
    ) -> Tuple[Optional[ForwardBatch], Optional[torch.Tensor]]:
        if len(candidate_input_ids) == 1:
            self._profile_skip_candidate_batch("single_candidate")
            return dataclasses.replace(
                forward_batch, input_ids=candidate_input_ids[0]
            ), None
        seq_lens_cpu = (
            list(forward_batch.extend_seq_lens_cpu)
            if forward_batch.extend_seq_lens_cpu is not None
            else forward_batch.seq_lens_cpu.tolist()
        )
        base_batch_size = len(seq_lens_cpu)
        seq_len = sum(int(x) for x in seq_lens_cpu)
        if any(cand.numel() != seq_len for cand in candidate_input_ids):
            self._profile_skip_candidate_batch("candidate_len_mismatch")
            return None, None
        if any(forward_batch.extend_prefix_lens_cpu or []):
            self._profile_skip_candidate_batch("has_prefix_lens")
            return None, None
        if getattr(model_runner.attn_backend, "use_paged", False):
            self._profile_skip_candidate_batch("attn_backend_use_paged")
            return None, None

        num_candidates = len(candidate_input_ids)
        expanded_batch_size = base_batch_size * num_candidates
        kv_indptr = getattr(model_runner.attn_backend, "kv_indptr", None)
        if kv_indptr is not None and len(kv_indptr) > 0:
            # FlashInfer plans with buffers sized by the server's batch capacity.
            # If the wrapper was initialized for a smaller batch, fall back to the
            # serial path instead of overrunning the planning buffers.
            if kv_indptr[0].numel() <= expanded_batch_size:
                self._profile_skip_candidate_batch("kv_indptr_capacity")
                return None, None

        input_ids = torch.cat(candidate_input_ids, dim=0)
        num_tokens = input_ids.numel()

        seq_lens = torch.full(
            (expanded_batch_size,),
            0,
            dtype=forward_batch.seq_lens.dtype,
            device=forward_batch.seq_lens.device,
        )
        seq_lens[:] = forward_batch.seq_lens.repeat(num_candidates)
        seq_lens_cpu_expanded = torch.tensor(
            seq_lens_cpu * num_candidates,
            dtype=forward_batch.seq_lens_cpu.dtype,
            device=forward_batch.seq_lens_cpu.device,
        )

        allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return None, None
        if getattr(allocator, "page_size", 1) == 1:
            out_cache_loc = allocator.alloc(num_tokens)
        else:
            prefix_lens = torch.zeros_like(seq_lens)
            prefix_lens_cpu = torch.zeros_like(seq_lens_cpu_expanded)
            last_loc = torch.full(
                (expanded_batch_size,),
                -1,
                dtype=torch.int64,
                device=forward_batch.input_ids.device,
            )
            out_cache_loc = allocator.alloc_extend(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu_expanded,
                last_loc=last_loc,
                extend_num_tokens=num_tokens,
            )
        if out_cache_loc is None or out_cache_loc.numel() != num_tokens:
            if out_cache_loc is not None:
                allocator.free(out_cache_loc)
            self._profile_skip_candidate_batch("kv_alloc_failed_or_wrong_size")
            return None, None

        extend_seq_lens = None
        extend_prefix_lens = None
        extend_start_loc = None
        if forward_batch.extend_seq_lens is not None:
            extend_seq_lens = forward_batch.extend_seq_lens.repeat(num_candidates)
            extend_prefix_lens = torch.zeros_like(extend_seq_lens)
            extend_start_loc = torch.cumsum(
                torch.cat(
                    [
                        torch.zeros(
                            1,
                            dtype=forward_batch.extend_start_loc.dtype,
                            device=forward_batch.extend_start_loc.device,
                        ),
                        extend_seq_lens[:-1],
                    ]
                ),
                dim=0,
            )
            extend_start_loc = extend_start_loc.to(
                dtype=forward_batch.extend_start_loc.dtype,
            )

        positions = None
        if forward_batch.positions is not None:
            positions = forward_batch.positions.repeat(num_candidates)

        req_pool_indices = forward_batch.req_pool_indices.repeat(num_candidates)
        orig_seq_lens = (
            forward_batch.orig_seq_lens.repeat(num_candidates)
            if forward_batch.orig_seq_lens is not None
            else None
        )

        expanded = dataclasses.replace(
            forward_batch,
            batch_size=expanded_batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu_expanded,
            seq_lens_sum=num_tokens,
            orig_seq_lens=orig_seq_lens,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=None,
            extend_num_tokens=num_tokens
            if forward_batch.extend_num_tokens is not None
            else None,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=[0] * expanded_batch_size
            if forward_batch.extend_prefix_lens_cpu is not None
            else None,
            extend_seq_lens_cpu=seq_lens_cpu * num_candidates
            if forward_batch.extend_seq_lens_cpu is not None
            else None,
            extend_logprob_start_lens_cpu=list(
                forward_batch.extend_logprob_start_lens_cpu
            )
            * num_candidates
            if forward_batch.extend_logprob_start_lens_cpu is not None
            else None,
            positions=positions,
            num_token_non_padded=torch.tensor(
                num_tokens,
                dtype=torch.int32,
                device=forward_batch.input_ids.device,
            )
            if forward_batch.num_token_non_padded is not None
            else None,
            num_token_non_padded_cpu=num_tokens,
            global_num_tokens_cpu=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
        )
        return expanded, out_cache_loc

    def _forward_candidate_logits_batch(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        candidates: List[torch.Tensor],
        allow_serial_fallback: bool = True,
    ) -> Optional[List[torch.Tensor]]:
        expanded_batch, temp_cache_loc = self._build_expanded_forward_batch(
            model_runner, forward_batch, candidates
        )
        if expanded_batch is None:
            if not allow_serial_fallback:
                return None
            return [
                self._forward_logits_with_input_ids(
                    model_runner,
                    forward_batch,
                    cand,
                    profile_label="candidate_serial_forward",
                )[0]
                for cand in candidates
            ]

        try:
            profile_label = (
                "candidate_batch_forward"
                if len(candidates) > 1
                else "candidate_single_forward"
            )
            out = self._profile_forward(
                profile_label,
                int(expanded_batch.input_ids.numel()),
                len(candidates),
                lambda: model_runner.forward(expanded_batch, pp_proxy_tensors=None),
            )
            logits = out.logits_output.full_logits
            if self.shift_logits:
                candidate_num_tokens = candidates[0].numel()
                logits = logits.view(len(candidates), candidate_num_tokens, -1)
                shifted_logits = logits.clone()
                for start, seq_len in self._iter_sequences(forward_batch):
                    end = start + seq_len
                    shifted_logits[:, start:end] = torch.cat(
                        [logits[:, start : start + 1], logits[:, start : end - 1]],
                        dim=1,
                    )
                logits = shifted_logits.view(
                    len(candidates) * candidate_num_tokens, -1
                )
            seq_len = candidates[0].numel()
            return [
                logits[i * seq_len : (i + 1) * seq_len]
                for i in range(len(candidates))
            ]
        finally:
            if temp_cache_loc is not None:
                model_runner.token_to_kv_pool_allocator.free(temp_cache_loc)

    def _iter_sequences(self, forward_batch: ForwardBatch):
        seq_lens = (
            forward_batch.extend_seq_lens_cpu
            if forward_batch.extend_seq_lens_cpu is not None
            else forward_batch.seq_lens_cpu.tolist()
        )
        offset = 0
        for seq_len in seq_lens:
            yield offset, seq_len
            offset += seq_len

    def _verify_vbs2_batch_reject_all(
        self,
        forward_batch: ForwardBatch,
        logits_base: torch.Tensor,
        x: torch.Tensor,
        speculate_index: List[List[int]],
        rejected_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[bool]]:
        batch_size = forward_batch.batch_size
        logits_full = logits_base
        unmask_per_sample = [True] * batch_size

        rejected_specs = []
        for batch_id, (seq_start, seq_len) in enumerate(
            self._iter_sequences(forward_batch)
        ):
            spec_pos = speculate_index[batch_id]
            if not spec_pos:
                continue
            pos = torch.tensor(spec_pos, dtype=torch.long, device=x.device)
            tok = x[pos]
            token_probs = _gather_token_probs_from_logits(logits_base, pos, tok)
            accepted = bool(
                ((tok != self.mask_id) & (token_probs >= self.threshold)).all().item()
            )
            entry = (batch_id, seq_start, seq_len, pos)
            if not accepted:
                rejected_specs.append(entry)

        if not rejected_specs:
            return x, logits_full, unmask_per_sample

        for batch_id, seq_start, seq_len, pos in rejected_specs:
            x[pos] = self.mask_id
            rejected_pos[batch_id, pos - seq_start] = True
            unmask_per_sample[batch_id] = False

        return x, logits_full, unmask_per_sample

    def _verify_speculation(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        logits_base: torch.Tensor,
        x: torch.Tensor,
        speculate_index: List[List[int]],
        speculate_conf: List[List[float]],
        speculate_token: List[List[int]],
        rejected_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[bool]]:
        batch_size = forward_batch.batch_size
        logits_full = logits_base
        unmask_per_sample = [True] * batch_size

        has_any_spec = any(len(speculate_index[i]) > 0 for i in range(batch_size))
        if not has_any_spec:
            return x, logits_full, unmask_per_sample

        if self.verify_batch_size == "spiffy":
            for batch_id, (seq_start, seq_len) in enumerate(
                self._iter_sequences(forward_batch)
            ):
                spec_pos = speculate_index[batch_id]
                if not spec_pos:
                    continue

                cands, cand_spec, graph, discarded, baseline = (
                    _build_spiffy_verify_candidates(
                        x,
                        spec_pos,
                        speculate_token[batch_id],
                        speculate_conf[batch_id],
                        self.mask_id,
                    )
                )
                for pos in discarded:
                    rejected_pos[batch_id, pos - seq_start] = True

                logits_verify = self._forward_candidate_logits_batch(
                    model_runner, forward_batch, cands
                )

                best_seq, accepted_pos, best_idx = _spiffy_accept_reject(
                    cands,
                    cand_spec,
                    graph,
                    logits_verify,
                    self.mask_id,
                    self.threshold,
                    baseline,
                )
                x = best_seq.clone()
                if batch_size == 1:
                    logits_full = logits_verify[best_idx]
                else:
                    if logits_full is logits_base:
                        logits_full = logits_base.clone()
                    logits_full[seq_start : seq_start + seq_len] = logits_verify[
                        best_idx
                    ][seq_start : seq_start + seq_len]

                if len(accepted_pos) == 0:
                    unmask_per_sample[batch_id] = False
                    for pos in spec_pos:
                        rejected_pos[batch_id, pos - seq_start] = True
            return x, logits_full, unmask_per_sample

        if self.verify_batch_size <= 1:
            for batch_id, (seq_start, _seq_len) in enumerate(
                self._iter_sequences(forward_batch)
            ):
                spec_pos = speculate_index[batch_id]
                if not spec_pos:
                    continue
                pos = torch.tensor(spec_pos, dtype=torch.long, device=x.device)
                tok = x[pos]
                reject = (
                    _gather_token_probs_from_logits(logits_base, pos, tok)
                    < self.threshold
                )
                if bool(reject.any().item()):
                    rejected_abs = pos[reject]
                    x[rejected_abs] = self.mask_id
                    rejected_pos[batch_id, rejected_abs - seq_start] = True
                    unmask_per_sample[batch_id] = False
            return x, logits_full, unmask_per_sample

        if self.verify_batch_size == 2:
            return self._verify_vbs2_batch_reject_all(
                forward_batch,
                logits_base,
                x,
                speculate_index,
                rejected_pos,
            )

        for batch_id, (seq_start, seq_len) in enumerate(
            self._iter_sequences(forward_batch)
        ):
            spec_pos = speculate_index[batch_id]
            if not spec_pos:
                continue

            cands, cand_spec = _build_verify_candidates(
                x,
                spec_pos,
                speculate_conf[batch_id],
                self.mask_id,
                int(self.verify_batch_size),
            )
            seq_lens_cpu = (
                list(forward_batch.extend_seq_lens_cpu)
                if forward_batch.extend_seq_lens_cpu is not None
                else forward_batch.seq_lens_cpu.tolist()
            )
            can_batch_all_candidates = (
                forward_batch.extend_seq_lens is None
                and len(seq_lens_cpu) == 1
                and all(cand.numel() == cands[0].numel() for cand in cands)
            )
            if can_batch_all_candidates:
                logits_verify = self._forward_candidate_logits_batch(
                    model_runner, forward_batch, cands
                )
            else:
                logits_verify = [logits_base]
                if len(cands) > 1:
                    logits_verify.extend(
                        self._forward_candidate_logits_batch(
                            model_runner, forward_batch, cands[1:]
                        )
                    )

            best_seq, accepted_spec, best_idx = _select_best_verified_candidate(
                logits_verify,
                cands,
                cand_spec,
                self.threshold,
                self.mask_id,
            )
            x = best_seq.clone()
            if batch_size == 1:
                logits_full = logits_verify[best_idx]
            else:
                if logits_full is logits_base:
                    logits_full = logits_base.clone()
                logits_full[seq_start : seq_start + seq_len] = logits_verify[best_idx][
                    seq_start : seq_start + seq_len
                ]

            if len(accepted_spec) == 0:
                unmask_per_sample[batch_id] = False
                for pos in spec_pos:
                    rejected_pos[batch_id, pos - seq_start] = True

        return x, logits_full, unmask_per_sample

    def _build_self_speculation(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        logits_full: torch.Tensor,
        rejected_pos: torch.Tensor,
    ) -> Tuple[List[List[int]], List[List[float]], List[List[int]]]:
        batch_size = forward_batch.batch_size
        speculate_index = [[] for _ in range(batch_size)]
        speculate_conf = [[] for _ in range(batch_size)]
        speculate_token = [[] for _ in range(batch_size)]

        for batch_id, (seq_start, seq_len) in enumerate(
            self._iter_sequences(forward_batch)
        ):
            seq_end = seq_start + seq_len
            mask_positions = torch.nonzero(
                x[seq_start:seq_end] == self.mask_id, as_tuple=True
            )[0]
            if mask_positions.numel() == 0:
                continue

            probs, toks = _max_prob_and_token(
                logits_full[seq_start:seq_end][mask_positions]
            )
            abs_positions = mask_positions + seq_start
            if self.speculative_decoding_mode == "confidence":
                keep = (
                    probs > self.confidence_speculative_threshold
                ) & ~rejected_pos[batch_id, mask_positions]
            else:
                keep = (
                    probs > self.self_speculative_confidence_threshold
                ) & ~rejected_pos[batch_id, mask_positions]
            if not keep.any():
                continue

            kept_probs = probs[keep]
            kept_toks = toks[keep]
            kept_positions = abs_positions[keep]
            if self.speculative_decoding_mode == "confidence":
                selected_positions = kept_positions
                selected_toks = kept_toks
                selected_probs = kept_probs
            else:
                top_k = min(self.num_speculate_tokens, kept_positions.numel())
                _, order = torch.topk(kept_probs, k=top_k)
                selected_positions = kept_positions[order]
                selected_toks = kept_toks[order]
                selected_probs = kept_probs[order]
            x[selected_positions] = selected_toks
            speculate_index[batch_id].extend(selected_positions.tolist())
            speculate_conf[batch_id].extend(selected_probs.tolist())
            speculate_token[batch_id].extend(selected_toks.tolist())

        return speculate_index, speculate_conf, speculate_token

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
        profile = self._profile_begin(seq_lens)
        old_profile = getattr(self, "_profile_state", None)
        self._profile_state = profile
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
            out = self._profile_forward(
                "no_mask_forward",
                int(forward_batch.input_ids.numel()),
                len(seq_lens),
                lambda: model_runner.forward(forward_batch, pp_proxy_tensors=None),
            )
            self._profile_finish(profile)
            self._profile_state = old_profile
            return out.logits_output, [], out.can_run_graph

        steps = self.block_size
        timesteps = torch.linspace(
            1, 1e-3, steps + 1, device=forward_batch.input_ids.device
        )
        can_run_cuda_graph = False
        logits_output = None
        rejected_pos = torch.zeros(
            batch_size,
            max(seq_lens),
            dtype=torch.bool,
            device=forward_batch.input_ids.device,
        )
        speculate_index: List[List[int]] = [[] for _ in range(batch_size)]
        speculate_conf: List[List[float]] = [[] for _ in range(batch_size)]
        speculate_token: List[List[int]] = [[] for _ in range(batch_size)]

        for i in range(steps):
            if not (forward_batch.input_ids == self.mask_id).any():
                break

            if profile is not None:
                profile["steps"] += 1
            unmask_per_sample = [True] * batch_size
            if profile is not None:
                profile["normal_steps"] += 1
            logits_base, logits_output, can_run_cuda_graph = (
                self._forward_logits_with_input_ids(
                    model_runner, forward_batch, forward_batch.input_ids
                )
            )
            logits_full = logits_base

            if self.speculative_decoding:
                (
                    verified_input_ids,
                    logits_full,
                    unmask_per_sample,
                ) = self._verify_speculation(
                    model_runner,
                    forward_batch,
                    logits_base,
                    forward_batch.input_ids,
                    speculate_index,
                    speculate_conf,
                    speculate_token,
                    rejected_pos,
                )
                forward_batch.input_ids = verified_input_ids

            for batch_id, seq_len in enumerate(seq_lens):
                seq_start = seq_starts[batch_id]
                seq_end = seq_start + seq_len
                curr_input_ids = forward_batch.input_ids[seq_start:seq_end]
                curr_logits = logits_full[seq_start:seq_end]

                mask_index = curr_input_ids == self.mask_id
                if not mask_index.any() or not unmask_per_sample[batch_id]:
                    continue

                mask_logits = curr_logits[mask_index]
                if self.alg == "topk_margin":
                    top2_logits, top2_tokens = torch.topk(mask_logits, k=2, dim=-1)
                    top2_probs = torch.exp(
                        top2_logits - torch.logsumexp(mask_logits, dim=-1, keepdim=True)
                    )
                    confidence_masked = top2_probs[:, 0] - top2_probs[:, 1]
                    x0_masked = top2_tokens[:, 0]
                elif self.alg in ("origin", "confidence_threshold"):
                    confidence_masked, x0_masked = _max_prob_and_token(mask_logits)
                else:
                    probs = F.softmax(mask_logits, dim=-1)
                    x0_masked = torch.argmax(probs, dim=-1)
                    confidence_masked = torch.sum(
                        probs * torch.log(probs + 1e-10), dim=-1
                    )

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
                    full_confidence[mask_index] = confidence_masked
                    _, transfer_index = torch.topk(full_confidence, num_transfer)

                    x_ = torch.full_like(curr_input_ids, self.mask_id)
                    x_[mask_index] = x0_masked
                    curr_input_ids[transfer_index] = x_[transfer_index]
                    continue

                confidence = torch.full_like(
                    curr_input_ids, -torch.inf, dtype=curr_logits.dtype
                )
                confidence[mask_index] = confidence_masked
                transfer_index = F.one_hot(
                    confidence.argmax(), num_classes=confidence.shape[0]
                ).bool()
                if self.alg == "confidence_threshold":
                    transfer_index = transfer_index | (confidence > self.threshold)

                transfer_index = transfer_index & mask_index
                x_ = torch.full_like(curr_input_ids, self.mask_id)
                x_[mask_index] = x0_masked
                curr_input_ids[transfer_index] = x_[transfer_index]

            if self.speculative_decoding:
                (
                    speculate_index,
                    speculate_conf,
                    speculate_token,
                ) = self._build_self_speculation(
                    forward_batch,
                    forward_batch.input_ids,
                    logits_full,
                    rejected_pos,
                )

        out = self._profile_forward(
            "final_forward",
            int(forward_batch.input_ids.numel()),
            len(seq_lens),
            lambda: model_runner.forward(forward_batch, pp_proxy_tensors=None),
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids_list = []
        for batch_id, seq_len in enumerate(seq_lens):
            seq_start = seq_starts[batch_id]
            curr_input_ids = forward_batch.input_ids[seq_start : seq_start + seq_len]
            next_token_ids_list.append(curr_input_ids[start_list[batch_id] :])

        self._profile_finish(profile)
        self._profile_state = old_profile
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
