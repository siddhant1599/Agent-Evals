"""Loading a backbone for QLoRA training: the same 4-bit NF4 base that serving uses (D-008, D-041).

Serving quantizes in flight inside vLLM; training loads the same pinned checkpoint through
transformers and bitsandbytes instead, with the format taken from the training config. Keeping the
loader here keeps quantization decided in ``serving/`` alone (AGENTS.md §4.1).
"""

from __future__ import annotations

from typing import Any

from ..config import TrainingConfig
from ..logging_utils import get_logger
from .backbones import BackboneSpec
from .base import ServingError

__all__ = ["load_training_model"]

logger = get_logger(__name__)

#: The registry's quantization method this loader reproduces for training.
_BITSANDBYTES = "bitsandbytes"


def load_training_model(backbone: BackboneSpec, cfg: TrainingConfig) -> tuple[Any, Any]:
    """Load a backbone in 4-bit, with its tokenizer, at the registry's pinned revision.

    Args:
        backbone: The registry row to load.
        cfg: Training configuration; supplies the 4-bit format and compute dtype.

    Returns:
        ``(model, tokenizer)``: the quantized causal LM on the current CUDA device, and its
        tokenizer. The LoRA adapter is added later, by the trainer.

    Raises:
        ServingError: If the backbone is not served with bitsandbytes or no GPU is available;
            4-bit bitsandbytes weights need CUDA.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if backbone.quantization != _BITSANDBYTES:
        raise ServingError(
            f"{backbone.name} is served with {backbone.quantization!r}; training loads only "
            f"{_BITSANDBYTES!r} backbones, so the trained base matches the served one (D-008)"
        )
    if not torch.cuda.is_available():
        raise ServingError("4-bit training needs a CUDA device; none is available")
    dtype = getattr(torch, cfg.compute_dtype)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_use_double_quant=cfg.double_quant,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        backbone.hf_path,
        revision=backbone.hf_revision,
        quantization_config=quantization,
        dtype=dtype,
        device_map={"": torch.cuda.current_device()},
    )
    tokenizer = AutoTokenizer.from_pretrained(backbone.hf_path, revision=backbone.hf_revision)
    logger.info(
        "loaded %s @ %s for training: 4-bit %s, double quant %s, compute %s",
        backbone.hf_path,
        backbone.hf_revision,
        cfg.quant_type,
        cfg.double_quant,
        cfg.compute_dtype,
    )
    return model, tokenizer
