import logging
import re
from typing import Dict, Optional

import torch
from transformers import AutoModel

logger = logging.getLogger("dinov3")


# Mapping from HF to DINOv3 (inverse of convert_dinov3_vit_to_hf.py)
# We manually create this because simple dict inversion doesn't handle regex
HF_TO_DINOV3_KEY_MAPPING = {
    r"embeddings.cls_token": r"cls_token",
    r"embeddings.mask_token": r"mask_token",
    r"embeddings.register_tokens": r"storage_tokens",
    r"embeddings.patch_embeddings": r"patch_embed.proj",
    r"inv_freq": r"periods",
    r"rope_embeddings": r"rope_embed",
    r"layer.(\d+).attention.o_proj": r"blocks.\1.attn.proj",
    r"layer.(\d+).attention.": r"blocks.\1.attn.",
    r"layer.(\d+).layer_scale(\d+).lambda1": r"blocks.\1.ls\2.gamma",
    r"layer.(\d+).mlp.up_proj": r"blocks.\1.mlp.fc1",
    r"layer.(\d+).mlp.down_proj": r"blocks.\1.mlp.fc2",
    r"layer.(\d+).mlp": r"blocks.\1.mlp",
    r"layer.(\d+).norm": r"blocks.\1.norm",
    r"gate_proj": r"w1",
    r"up_proj": r"w2",
    r"down_proj": r"w3",
}


def _convert_hf_keys_to_dinov3(state_dict_keys):
    """
    Convert HuggingFace DINOv3 state dict keys to native DINOv3 format.

    Uses the inverse of the official ORIGINAL_TO_CONVERTED_KEY_MAPPING.

    Args:
        state_dict_keys: List of HuggingFace model state dict keys

    Returns:
        Dictionary mapping old keys to new keys
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in HF_TO_DINOV3_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def _fuse_qkv_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Fuse separate Q, K, V weights from HuggingFace format into single qkv.

    This reverses the split_qkv operation from convert_dinov3_vit_to_hf.py.

    Args:
        state_dict: State dict with separate q_proj, k_proj, v_proj

    Returns:
        State dict with fused qkv weights
    """
    keys_to_fuse = [x for x in state_dict.keys() if ("q_proj" in x or "k_proj" in x or "v_proj" in x)]

    # Group by block and param type
    blocks = {}
    for key in keys_to_fuse:
        pattern = r"blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj)\.(weight|bias)"
        match = re.search(pattern, key)
        if match:
            block_idx = match.group(1)
            proj_type = match.group(2)
            param_type = match.group(3)

            if block_idx not in blocks:
                blocks[block_idx] = {}
            if param_type not in blocks[block_idx]:
                blocks[block_idx][param_type] = {}

            blocks[block_idx][param_type][proj_type] = key

    # Fuse q, k, v for each block
    for block_idx, param_types in blocks.items():
        for param_type, proj_keys in param_types.items():
            if len(proj_keys) == 3:
                q = state_dict.pop(proj_keys["q_proj"])
                k = state_dict.pop(proj_keys["k_proj"])
                v = state_dict.pop(proj_keys["v_proj"])

                qkv = torch.cat([q, k, v], dim=0)
                new_key = f"blocks.{block_idx}.attn.qkv.{param_type}"
                state_dict[new_key] = qkv

    return state_dict


def load_huggingface_model(model_id: str, cfg, cache_dir: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Load a DINOv3 model from HuggingFace Hub and convert to native format.

    Args:
        model_id: HuggingFace model identifier
                  (e.g., "facebook/dinov3-vitb16")
        cfg: DINOv3 training configuration
        cache_dir: Optional directory to cache downloaded models

    Returns:
        State dict compatible with native DINOv3 models

    Raises:
        ValueError: If model architecture is incompatible
        RuntimeError: If model loading fails
    """
    logger.info(f"Loading HuggingFace model: {model_id}")

    try:
        # Download model
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
        original_state_dict = model.state_dict()

        # Convert keys using the official mapping (inverted)
        logger.info("Converting HuggingFace keys to DINOv3 format")
        original_keys = list(original_state_dict.keys())
        new_keys = _convert_hf_keys_to_dinov3(original_keys)

        converted_state_dict = {}
        for key in original_keys:
            new_key = new_keys[key]
            weight_tensor = original_state_dict[key]

            # Skip keys not needed in DINOv3
            if "bias_mask" in key or "local_cls_norm" in key:
                continue
            if "inv_freq" in new_key:
                continue

            # Handle mask token shape
            if "mask_token" in new_key:
                weight_tensor = weight_tensor.unsqueeze(1)

            converted_state_dict[new_key] = weight_tensor

        # Fuse q, k, v weights into qkv
        converted_state_dict = _fuse_qkv_weights(converted_state_dict)

        logger.info(f"Successfully loaded and converted HuggingFace model {model_id}")
        logger.info(f"Model contains {len(converted_state_dict)} parameters")

        return converted_state_dict

    except Exception as e:
        error_msg = f"Failed to load HuggingFace model {model_id}: {str(e)}"
        raise RuntimeError(error_msg) from e
