from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType


class DreamModel(Qwen2ForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY
        self.logits_processor = LogitsProcessor(self.config, return_full_logits=True)


EntryClass = DreamModel
