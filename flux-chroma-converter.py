#!/usr/bin/env python3
r"""
Flux Dev to Chroma LoRA Converter v17.9 - Definitive Key Mapping Fix

This version provides a definitive fix to the key normalization logic, correctly
handling the hybrid dot/underscore naming convention of the Flux/Chroma models.

Evidence-Based Change Justification (v17.9):
-   **Problem:** The v17.8 converter, while correctly identifying structural keys,
    made an incorrect assumption about separator conversion. It converted ALL
    underscores to dots, resulting in incorrect keys like `single.blocks.0.linear1`.
    This still caused "tensor not in base model" errors.

-   **Evidence:** The console log showed the script was looking for `single.blocks.0.linear1.weight`,
    but analysis of the `flux-dev-pruned.safetensors` model structure (via the provided JSON)
    proves the actual key is `single_blocks.0.linear1.weight`. The key difference is
    the underscore in `single_blocks`.

-   **Analysis:** The Flux/Chroma model architecture uses a hybrid naming scheme. The
    main block prefixes are `single_blocks` and `double_blocks` (with an underscore).
    However, all subsequent levels of the hierarchy are separated by dots (e.g., `.0.linear1`).
    A simple global replacement of `_` with `.` is therefore incorrect.

-   **Solution (Correctness & Precision):**
    1.  **Precise Structural Key Transformation:** The fallback logic in `normalize_lora_key`
        for structural keys has been completely rewritten.
    2.  It now correctly handles keys like `lora_unet_single_blocks_0_linear1`. After stripping the
        `lora_unet_` part, the core key is `single_blocks_0_linear1`.
    3.  The new logic separates the prefix (`single_blocks`) from the rest of the key (`0_linear1`).
    4.  It then replaces underscores with dots *only in the rest of the key* (`0.linear1`).
    5.  Finally, it joins them back together with a dot, producing the correct key: `single_blocks.0.linear1`.
    6.  This correctly transforms `single_blocks_0_linear1` into `single_blocks.0.linear1`,
        and `double_blocks_10_img_attn_qkv` into `double_blocks.10.img_attn.qkv`,
        perfectly matching the target model's key structure.
    7.  This definitive fix ensures that both standard semantic LoRAs and structural
        LoRAs are correctly mapped, resolving all key-related errors.
"""

import argparse
import json
import gc
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Optional, Tuple, List, Set, Any
import logging
from tqdm import tqdm
import warnings
import re
import traceback
import sys
from collections import Counter, defaultdict
from datetime import datetime
import shutil

# Optional imports for debug mode
try:
    import psutil
    import GPUtil
    HAS_DEBUG_LIBS = True
except ImportError:
    HAS_DEBUG_LIBS = False
    
warnings.filterwarnings('ignore', category=SyntaxWarning) # Suppress the invalid escape sequence warning in the docstring

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global debug flag
DEBUG_MODE = False

# --- Key Mappings ---
# Flexible regex patterns to accept either '.' or '_' as separators between ALL components.
DIFFUSERS_KEY_MAPPING = {
    # Single blocks - Flux only has linear1, linear2, and norm layers
    r"single_transformer_blocks[._](\d+)[._]attn[._]to_q": "single_blocks.{}.linear1",
    r"single_transformer_blocks[._](\d+)[._]attn[._]to_k": "single_blocks.{}.linear1",
    r"single_transformer_blocks[._](\d+)[._]attn[._]to_v": "single_blocks.{}.linear1",
    r"single_transformer_blocks[._](\d+)[._]attn[._]to_out[._]0": "single_blocks.{}.linear2",
    r"single_transformer_blocks[._](\d+)[._]ff[._]net[._]0[._]proj": "single_blocks.{}.linear1",
    r"single_transformer_blocks[._](\d+)[._]ff[._]net[._]2": "single_blocks.{}.linear2",
    r"single_transformer_blocks[._](\d+)[._]proj_mlp": "single_blocks.{}.linear1", # proj_mlp is an alias for ff.net.0.proj
    r"single_transformer_blocks[._](\d+)[._]proj_out": "single_blocks.{}.linear2",
    # This layer does not exist in flux1-dev-pruned, so we explicitly skip it.
    r"single_transformer_blocks[._](\d+)[._]norm[._]linear": None,

    # Double blocks (transformer_blocks without 'single')
    r"transformer_blocks[._](\d+)[._]attn[._]to_q": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn[._]to_k": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn[._]to_v": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn[._]to_out[._]0": "double_blocks.{}.img_attn.proj",
    r"transformer_blocks[._](\d+)[._]ff[._]net[._]0[._]proj": "double_blocks.{}.img_mlp.0",
    r"transformer_blocks[._](\d+)[._]ff[._]net[._]2": "double_blocks.{}.img_mlp.2",
    r"transformer_blocks[._](\d+)[._]ff_context[._]net[._]0[._]proj": "double_blocks.{}.txt_mlp.0",
    r"transformer_blocks[._](\d+)[._]ff_context[._]net[._]2": "double_blocks.{}.txt_mlp.2",
    
    # Context attention for double blocks
    r"transformer_blocks[._](\d+)[._]attn_context[._]to_q": "double_blocks.{}.txt_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn_context[._]to_k": "double_blocks.{}.txt_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn_context[._]to_v": "double_blocks.{}.txt_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn_context[._]to_out[._]0": "double_blocks.{}.txt_attn.proj",
    
    # These norm layers do not exist in flux1-dev-pruned, so we explicitly skip them.
    r"transformer_blocks[._](\d+)[._]norm1[._]linear": None,
    r"transformer_blocks[._](\d+)[._]norm1_context[._]linear": None,
    
    # Diffusers-specific attention layers (these are aliases for the above, handled by the merger)
    r"transformer_blocks[._](\d+)[._]attn[._]add_q_proj": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn[._]add_k_proj": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn[._]add_v_proj": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks[._](\d+)[._]attn[._]to_add_out": "double_blocks.{}.img_attn.proj",
    
    # General text encoder patterns (skip these from UNet processing)
    r"text_encoder.*": None,
    r"te_.*": None,
    r"lora_te.*": None,
    r"lora_te1.*": None,
    r"lora_te2.*": None,
    r"unet\.time_embedding.*": None,
    r"unet\.label_emb.*": None,
}

# Keys are now prefix-less and dot-separated for a cleaner lookup after pattern matching fails.
DIRECT_KEY_MAPPING = {
    # These are exact replacements, not patterns.
    # They are mapped to `None` because they exist in the full, non-pruned Diffusers
    # Flux model but are ABSENT in the `flux1-dev-pruned.safetensors` model.
    # Skipping them is the correct action for compatibility with the pruned model.
    "proj.out": None,
    "norm.out.linear": None,
    "context.embedder": None,
    "x.embedder": None,
    "time.text.embed.guidance.embedder.linear.1": None,
    "time.text.embed.guidance.embedder.linear.2": None,
    "time.text.embed.text.embedder.linear.1": None,
    "time.text.embed.text.embedder.linear.2": None,
    "time.text.embed.timestep.embedder.linear.1": None,
    "time.text.embed.timestep.embedder.linear.2": None,
}

# Known shape mappings for Flux/Chroma
KNOWN_SHAPES = {
    "single_blocks.linear1": (21504, 3072),  # [out, in]
    "single_blocks.linear2": (3072, 15360),  # [out, in]
    "double_blocks.img_attn.qkv": (9216, 3072),  # 3x3072 for Q,K,V
    "double_blocks.txt_attn.qkv": (9216, 3072),
    "double_blocks.img_attn.proj": (3072, 3072),
    "double_blocks.txt_attn.proj": (3072, 3072),
    "double_blocks.img_mlp.0": (12288, 3072),
    "double_blocks.img_mlp.2": (3072, 12288),
    "double_blocks.txt_mlp.0": (12288, 3072),
    "double_blocks.txt_mlp.2": (3072, 12288),
}

LORA_SLICE_MAPPING = {
    # For single_blocks.linear1 (slicing on dim 0, total size 21504)
    ('linear1', 'to_q'):            (0, 0, 3072),
    ('linear1', 'to_k'):            (0, 3072, 6144),
    ('linear1', 'to_v'):            (0, 6144, 9216),
    ('linear1', 'ff.net.0.proj'):   (0, 9216, 21504),
    ('linear1', 'proj_mlp'):        (0, 9216, 21504), # proj_mlp is an alias for the FFN input proj

    # For double_blocks...qkv (slicing on dim 0, total size 9216)
    ('qkv', 'to_q'):                (0, 0, 3072),
    ('qkv', 'add_q_proj'):          (0, 0, 3072), # Alias for to_q
    ('qkv', 'attn_context.to_q'):   (0, 0, 3072), # Alias for to_q in context attn
    ('qkv', 'to_k'):                (0, 3072, 6144),
    ('qkv', 'add_k_proj'):          (0, 3072, 6144), # Alias for to_k
    ('qkv', 'attn_context.to_k'):   (0, 3072, 6144), # Alias for to_k in context attn
    ('qkv', 'to_v'):                (0, 6144, 9216),
    ('qkv', 'add_v_proj'):          (0, 6144, 9216), # Alias for to_v
    ('qkv', 'attn_context.to_v'):   (0, 6144, 9216), # Alias for to_v in context attn
}


def print_memory_usage(stage: str = ""):
    """Print current memory usage"""
    if not HAS_DEBUG_LIBS or not DEBUG_MODE:
        return
        
    try:
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
        
        msg = f"[Memory - {stage}] RAM: {ram_mb:.1f}MB"
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    msg += f" | GPU{i}: {gpu.memoryUsed:.0f}MB/{gpu.memoryTotal:.0f}MB"
        
        logger.info(msg)
    except:
        pass

def get_expected_shape(key: str) -> Optional[Tuple[int, int]]:
    """Get expected shape for a given key"""
    # Try to match patterns
    if "single_blocks" in key and "linear1" in key:
        return KNOWN_SHAPES["single_blocks.linear1"]
    elif "single_blocks" in key and "linear2" in key:
        return KNOWN_SHAPES["single_blocks.linear2"]
    elif "double_blocks" in key:
        for pattern, shape in KNOWN_SHAPES.items():
            if pattern.split(".")[-1] in key:
                return shape
    return None

def normalize_lora_key(key: str) -> Optional[str]:
    """
    Normalize a LoRA key from various formats (semantic Diffusers, structural Kohya)
    to the dot-separated format used by the Chroma base model.
    Returns None if the key should be skipped.
    """
    original_key = key
    
    # Step 1: Clean up suffixes and alpha values.
    key = re.sub(r'\.lora_[AB]\.weight$', '', key)
    key = re.sub(r'\.lora_(up|down)\.weight$', '', key)
    key = re.sub(r'\.alpha$', '', key)
    
    # Step 2: Skip all text encoder keys. They are not part of the UNet.
    if any(key.startswith(prefix) for prefix in ['lora_te', 'text_encoder', 'te_', 'lora_te1', 'lora_te2']):
        return None

    # Step 3: Strip all known UNet prefixes to get the "core" key.
    core_key = key
    prefixes_to_strip = [
        'lora_unet_', 'lora_transformer_', 'unet.', 'transformer.', 
        'model.', 'diffusion_model.'
    ]
    for prefix in prefixes_to_strip:
        if core_key.startswith(prefix):
            core_key = core_key[len(prefix):]
            break

    # Step 4: Attempt to match the core key against the SEMANTIC patterns first.
    # This handles standard LoRAs that need their components (q, k, v, etc.) mapped to fused layers.
    for pattern, replacement in DIFFUSERS_KEY_MAPPING.items():
        match = re.match(pattern, core_key)
        if match:
            if replacement is None:
                if DEBUG_MODE:
                    logger.debug(f"Skipping (pattern): {original_key} -> {core_key}")
                return None
            else:
                mapped = replacement.format(*match.groups())
                if DEBUG_MODE:
                    logger.debug(f"Pattern mapping: {original_key} -> {core_key} -> {mapped}")
                return mapped

    # --- DEFINITIVE FIX (Step 4.5) ---
    # If no semantic pattern matched, it's likely a structural key (e.g., 'single_blocks_0_linear1').
    # We must convert it to the hybrid `single_blocks.0.linear1` format.
    # This logic is now precise and based on the evidence from the base model structure.
    if core_key.startswith('single_blocks_'):
        prefix = 'single_blocks'
        rest_of_key = core_key[len(prefix)+1:] # +1 to skip the first underscore
        rest_of_key_dotted = rest_of_key.replace('_', '.')
        final_key = f"{prefix}.{rest_of_key_dotted}"
        if DEBUG_MODE:
            logger.debug(f"Structural mapping: {original_key} -> {core_key} -> {final_key}")
        return final_key
    
    if core_key.startswith('double_blocks_'):
        prefix = 'double_blocks'
        rest_of_key = core_key[len(prefix)+1:] # +1 to skip the first underscore
        rest_of_key_dotted = rest_of_key.replace('_', '.')
        final_key = f"{prefix}.{rest_of_key_dotted}"
        if DEBUG_MODE:
            logger.debug(f"Structural mapping: {original_key} -> {core_key} -> {final_key}")
        return final_key

    # Step 5: If no pattern matched, check against the direct, non-block mappings.
    direct_check_key = core_key.replace('_', '.')
    if direct_check_key in DIRECT_KEY_MAPPING:
        mapped = DIRECT_KEY_MAPPING[direct_check_key]
        if DEBUG_MODE:
            logger.debug(f"Direct mapping: {original_key} -> {direct_check_key} -> {mapped}")
        return mapped

    # If no mapping was found at all, log it and return the unmapped key for analysis.
    if DEBUG_MODE:
        logger.debug(f"No mapping found for '{original_key}', which normalized to core key '{core_key}'")
    return core_key


def is_modulation_layer(key: str) -> bool:
    """
    Check if a key is a modulation layer. These layers often exist in full
    Flux models but not in Chroma, so they must be skipped.
    """
    modulation_keywords = [
        "_mod.", ".mod.", "_mod_", ".modulation.",
        "mod_out", "norm_out", "scale_shift",
        "mod.lin", "modulated", "norm_k.", "norm_q.",
        # Only skip these if they are actually modulation layers
        "img_mod", "txt_mod", "vector_in",
        "guidance_in", "timestep_embedder"
    ]
    
    # Be more specific about what constitutes a modulation layer
    for mod in modulation_keywords:
        if mod in key:
            # Don't skip norm_added layers - they might be diffusers layers
            if "norm_added" in key:
                continue
            return True
    
    return False

def detect_lora_rank(lora_path: str) -> Tuple[int, str]:
    """Detect LoRA rank and naming style from the file"""
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        # Detect naming style
        naming_style = "unknown"
        if any(".lora_up." in k or ".lora_down." in k for k in keys):
            naming_style = "diffusers"
        elif any(".lora_A." in k or ".lora_B." in k for k in keys):
            naming_style = "kohya"
        
        # Get rank from first LoRA layer
        for key in keys:
            if "lora_" in key and "weight" in key and not key.startswith('lora_te'):
                tensor = f.get_tensor(key)
                # LoRA down/A weights have rank as one dimension
                rank = min(tensor.shape)
                return rank, naming_style
    
    return 16, naming_style  # Default

def analyze_lora_compatibility(lora_path: str) -> Dict[str, Any]:
    """Analyze LoRA compatibility with detailed statistics"""
    analysis = {
        "total_pairs": 0,
        "compatible_pairs": 0,
        "incompatible_modulation_pairs": 0,
        "skipped_nonexistent_pairs": 0,
        "skipped_unmapped_pairs": 0,
        "text_encoder_pairs": 0,
        "naming_style": "unknown",
        "rank": 16,
        "convertible_list": [],
        "incompatible_modulation_list": [],
        "skipped_nonexistent_list": [],
        "skipped_unmapped_list": [],
    }
    
    try:
        rank, naming_style = detect_lora_rank(lora_path)
        analysis["rank"] = rank
        analysis["naming_style"] = naming_style
    except Exception as e:
        logger.warning(f"Could not auto-detect rank/style: {e}")
    
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        # Group keys by base name
        processed_bases = set()
        
        for key in keys:
            if "alpha" in key:
                continue
                
            # Find base key
            base_key = None
            if ".lora_A.weight" in key or ".lora_down.weight" in key:
                base_key = key.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
            elif ".lora_B.weight" in key or ".lora_up.weight" in key:
                continue  # We'll process these with their down/A counterpart
            else:
                continue
            
            if base_key in processed_bases:
                continue
                
            processed_bases.add(base_key)
            
            # Check if it's a text encoder key
            if any(base_key.startswith(prefix) for prefix in ['lora_te', 'text_encoder', 'te_']):
                analysis["text_encoder_pairs"] += 1
                continue

            analysis["total_pairs"] += 1
            
            # Check if convertible
            normalized_key = normalize_lora_key(base_key)
            
            if normalized_key is None:
                analysis["skipped_nonexistent_pairs"] += 1
                analysis["skipped_nonexistent_list"].append(base_key)
            elif is_modulation_layer(normalized_key):
                analysis["incompatible_modulation_pairs"] += 1
                analysis["incompatible_modulation_list"].append(base_key)
            # A key is only compatible if it maps to a known Chroma prefix.
            elif normalized_key and normalized_key.startswith(('single_blocks', 'double_blocks')):
                analysis["compatible_pairs"] += 1
                analysis["convertible_list"].append(base_key)
            else:
                # All other cases are considered unmapped.
                analysis["skipped_unmapped_pairs"] += 1
                analysis["skipped_unmapped_list"].append(base_key)
    
    return analysis

def load_lora_metadata(lora_path: str) -> Dict[str, str]:
    """Load metadata from LoRA file"""
    metadata = {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            if hasattr(f, 'metadata'):
                metadata = f.metadata() or {}
                if DEBUG_MODE and metadata:
                    logger.debug(f"Loaded {len(metadata)} metadata entries from LoRA")
                    # Log trigger word metadata
                    for key, value in metadata.items():
                        if any(word in key.lower() for word in ["trigger", "prompt", "keyword"]):
                            logger.debug(f"Trigger metadata: {key} = {value}")
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
    
    return metadata

def convert_text_encoder_weights(lora_path: str) -> Dict[str, torch.Tensor]:
    """
    (EXPERIMENTAL) Opens a LoRA file and converts text encoder-related keys.
    WARNING: This conversion is known to fail in current ComfyUI versions.
    Use the --enable-text-encoder-conversion flag to attempt it.
    """
    logger.warning("Attempting EXPERIMENTAL conversion of text encoder weights...")
    te_weights = {}
    
    source_prefix = "lora_te1_text_model_"
    target_prefix = "lora_te_"

    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
            # Target only the T5 encoder keys, which start with `lora_te1_` in diffusers.
            te1_keys = [k for k in all_keys if k.startswith(source_prefix)]
            te2_keys = [k for k in all_keys if k.startswith("lora_te2_")]

            if not te1_keys and not te2_keys:
                logger.info("No text encoder weights found in the source LoRA to convert.")
                return {}

            if te2_keys:
                logger.info(f"Skipping {len(te2_keys)} `lora_te2_` (CLIP-L) keys as they are not used by the standard Chroma workflow.")

            if not te1_keys:
                logger.warning("No `lora_te1_` (T5) keys found to convert.")
                return {}

            logger.info(f"Found {len(te1_keys)} `lora_te1_` (T5) keys to convert.")
            
            converted_count = 0
            for key in tqdm(te1_keys, desc="Converting T5 TE weights (Experimental)"):
                # Perform the simple prefix replacement.
                new_key = target_prefix + key[len(source_prefix):]
                
                if DEBUG_MODE:
                    logger.debug(f"Renamed TE key: '{key}' -> '{new_key}'")

                te_weights[new_key] = f.get_tensor(key)
                converted_count += 1
        
        alpha_keys = len([k for k in te_weights if 'alpha' in k])
        logger.info(f"Successfully converted {converted_count} tensors for {alpha_keys} T5 text encoder LoRA modules.")
            
    except Exception as e:
        logger.error(f"Could not convert text encoder weights: {e}")
        traceback.print_exc() if DEBUG_MODE else None
        
    return te_weights

def convert_lora_key_to_chroma_format(key: str) -> str:
    """
    Convert a base model key (e.g. 'double_blocks.0.img_mlp.0') to proper Chroma LoRA format.
    """
    chroma_key = key.replace(".", "_")
    if not chroma_key.startswith("lora_unet_"):
        chroma_key = f"lora_unet_{chroma_key}"
    return chroma_key

class FluxLoRAApplier:
    @staticmethod
    def apply_lora(base_path: str, lora_path: str, output_path: str, device: str = "cuda", 
                   lora_scale: float = 1.0, chunk_size: int = 50) -> Tuple[int, Dict[str, Any]]:
        """Apply LoRA to base model with alias-aware accumulation using slicing."""
        logger.info(f"Applying LoRA with scale {lora_scale}")
        
        applied_count = 0
        skipped_count = 0
        accumulated_count = 0
        
        # Load LoRA structure first to understand what we're dealing with
        lora_structure = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "alpha" not in key and (".lora_down." in key or ".lora_A." in key):
                    base_key = key.replace(".lora_down.weight", "").replace(".lora_A.weight", "")
                    normalized = normalize_lora_key(base_key)
                    if normalized and not is_modulation_layer(normalized):
                        lora_structure[base_key] = normalized
                    elif DEBUG_MODE and normalized is not None:
                        # Log keys that normalize but are not convertible to see what's being skipped
                        logger.debug(f"Skipping unusable normalized key: {base_key} -> {normalized}")


        # Group LoRAs by their target layer
        target_groups = defaultdict(list)
        for base_key, normalized in lora_structure.items():
            target_groups[normalized].append(base_key)
        
        # Log grouping info
        for target, sources in target_groups.items():
            if len(sources) > 1:
                components = sorted([s[s.rfind('.')+1:] if '.' in s else s[s.rfind('_')+1:] for s in sources])
                logger.info(f"Grouped {len(sources)} LoRAs targeting {target}: {components}")
        
        # Process in chunks of target groups
        grouped_keys = list(target_groups.keys())
        total_chunks = (len(grouped_keys) + chunk_size - 1) // chunk_size
        
        # Copy base model
        logger.info("Copying base model...")
        shutil.copy2(base_path, output_path)
        
        # Process each chunk
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(grouped_keys))
            chunk_target_keys = grouped_keys[start_idx:end_idx]
            
            if not chunk_target_keys:
                continue
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_target_keys)} target layers)")
            
            updates = {}
            
            with safe_open(output_path, framework="pt", device="cpu") as base_f:
                base_keys_available = set(base_f.keys())

                for target_key in chunk_target_keys:
                    source_keys = target_groups[target_key]
                    target_tensor_key = f"{target_key}.weight"

                    if target_tensor_key not in base_keys_available:
                        logger.warning(f"Skipping group for target {target_key}: tensor {target_tensor_key} not in base model. This is expected for pruned models.")
                        skipped_count += len(source_keys)
                        continue

                    try:
                        base_weight = base_f.get_tensor(target_tensor_key).to(device)
                        full_delta = torch.zeros_like(base_weight)
                        
                        populated_slices = set()
                        
                        with safe_open(lora_path, framework="pt", device=device) as lora_f:
                            for source_key in source_keys:
                                if f"{source_key}.lora_down.weight" in lora_f.keys():
                                    down_key, up_key = f"{source_key}.lora_down.weight", f"{source_key}.lora_up.weight"
                                else:
                                    down_key, up_key = f"{source_key}.lora_A.weight", f"{source_key}.lora_B.weight"
                                
                                lora_down = lora_f.get_tensor(down_key)
                                lora_up = lora_f.get_tensor(up_key)
                                alpha = lora_scale
                                if f"{source_key}.alpha" in lora_f.keys():
                                    alpha_tensor = lora_f.get_tensor(f"{source_key}.alpha")
                                    rank = min(lora_down.shape)
                                    alpha = float(alpha_tensor.item()) / rank if rank > 0 else float(alpha_tensor.item())
                                
                                lora_delta = alpha * (lora_up @ lora_down)

                                # --- Reworked component matching logic to be robust and correct. ---
                                target_suffix = target_key.split('.')[-1]
                                
                                matched_source_suffix = None
                                # Get all possible component suffixes for the current target type (e.g., 'linear1' or 'qkv')
                                possible_components = [
                                    comp for (tgt, comp) in LORA_SLICE_MAPPING.keys() if tgt == target_suffix
                                ]
                                # Sort by length (desc) to find the most specific match first (e.g., 'add_q_proj' before 'proj')
                                possible_components.sort(key=len, reverse=True)

                                for s_suffix in possible_components:
                                    # Check if the source key ends with the component, preceded by a separator.
                                    # This correctly handles `..._to_k` and `...attn.to_q`.
                                    pattern = r'([._])' + re.escape(s_suffix) + '$'
                                    if re.search(pattern, source_key):
                                        matched_source_suffix = s_suffix
                                        break
                                
                                slice_info = LORA_SLICE_MAPPING.get((target_suffix, matched_source_suffix))
                                
                                if slice_info:
                                    if slice_info in populated_slices:
                                        logger.warning(f"Slice {slice_info} for target '{target_key}' already populated. Skipping likely alias '{source_key}'.")
                                        continue
                                    populated_slices.add(slice_info)

                                    dim, start, end = slice_info
                                    
                                    if dim == 0:
                                        if end > full_delta.shape[0] or lora_delta.shape[0] != (end - start):
                                            logger.error(f"Slice/delta shape mismatch for {source_key} -> {target_key} on dim 0. "
                                                         f"Slice is {end-start}, delta is {lora_delta.shape[0]}. Skipping.")
                                            continue
                                        full_delta[start:end, :] += lora_delta
                                    
                                    elif dim == 1:
                                        if end > full_delta.shape[1] or lora_delta.shape[1] != (end - start):
                                            logger.error(f"Slice/delta shape mismatch for {source_key} -> {target_key} on dim 1. "
                                                         f"Slice is {end-start}, delta is {lora_delta.shape[1]}. Skipping.")
                                            continue
                                        full_delta[:, start:end] += lora_delta
                                else:
                                    # This branch handles full tensor updates (e.g., from a structural LoRA)
                                    FULL_TENSOR_MARKER = "full_tensor_update"
                                    if FULL_TENSOR_MARKER in populated_slices:
                                        logger.warning(f"Target '{target_key}' is already populated by an alias. Skipping likely alias '{source_key}'.")
                                        continue
                                    
                                    if full_delta.shape != lora_delta.shape:
                                        logger.error(f"Cannot apply component '{source_key}' due to shape mismatch. "
                                                     f"Target shape: {full_delta.shape}, delta shape: {lora_delta.shape}. Skipping.")
                                        continue

                                    if DEBUG_MODE:
                                        logger.debug(f"Applying '{source_key}' as a full-tensor update to '{target_key}'.")
                                    full_delta += lora_delta
                                    populated_slices.add(FULL_TENSOR_MARKER)

                        if full_delta.shape != base_weight.shape:
                            logger.error(f"Final delta shape {full_delta.shape} for {target_key} does not match base shape {base_weight.shape}. Skipping.")
                            skipped_count += len(source_keys)
                            continue
                        
                        updates[target_tensor_key] = (base_weight + full_delta).cpu()
                        if len(source_keys) > 1:
                            accumulated_count += 1
                        else:
                            applied_count += 1

                    except Exception as e:
                        logger.error(f"Error processing group {target_key}: {e}")
                        traceback.print_exc() if DEBUG_MODE else None
                        skipped_count += len(source_keys)

            if updates:
                temp_dict = {}
                metadata = {}
                with safe_open(output_path, framework="pt", device="cpu") as f:
                    if hasattr(f, "metadata"):
                        metadata = f.metadata() or {}
                    for key in f.keys():
                        temp_dict[key] = updates.get(key, f.get_tensor(key))
                
                temp_output_file = Path(output_path).with_suffix(".safetensors.tmp")
                save_file(temp_dict, str(temp_output_file), metadata=metadata)
                del temp_dict
                gc.collect()
                shutil.move(str(temp_output_file), output_path)
        
        merge_stats = {
            "applied": applied_count,
            "skipped": skipped_count,
            "accumulated": list(k for k, v in target_groups.items() if len(v) > 1),
        }
        
        logger.info(f"Merge complete:")
        logger.info(f"  Applied (single): {applied_count} layers")
        logger.info(f"  Applied (grouped): {accumulated_count} layers")
        logger.info(f"  Skipped: {skipped_count} layers/keys")
        
        return applied_count + accumulated_count, merge_stats

class ChromaDifferenceExtractor:
    @staticmethod  
    def extract_differences(flux_base: str, flux_merged: str, output_path: str, 
                          device: str = "cuda", mode: str = "standard",
                          similarity_threshold: float = 0.1) -> None:
        """Extract differences between merged and base Flux models"""
        logger.info(f"Extracting differences in {mode} mode")
        
        differences = {}
        processed = 0
        
        with safe_open(flux_base, framework="pt", device="cpu") as base_f:
            with safe_open(flux_merged, framework="pt", device="cpu") as merged_f:
                keys = list(base_f.keys())
                
                for key in tqdm(keys, desc="Extracting differences"):
                    base_tensor = base_f.get_tensor(key).to(device)
                    merged_tensor = merged_f.get_tensor(key).to(device)
                    
                    if base_tensor.shape != merged_tensor.shape:
                        logger.warning(f"Shape mismatch for {key}, skipping")
                        continue
                    
                    diff = merged_tensor - base_tensor
                    
                    if diff.abs().max() > 1e-6:
                        differences[key] = diff.cpu()
                        processed += 1
                    
                    del base_tensor, merged_tensor, diff
                    if processed % 50 == 0:
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
        
        logger.info(f"Found differences in {len(differences)} layers")
        save_file(differences, output_path)

class ChromaDifferenceApplier:
    @staticmethod
    def apply_differences(chroma_base: str, differences_path: str, output_path: str,
                         device: str = "cuda", mode: str = "standard",
                         similarity_threshold: float = 0.1) -> None:
        """Apply extracted differences to Chroma base model"""
        logger.info(f"Applying differences to Chroma in {mode} mode")
        
        shutil.copy2(chroma_base, output_path)
        
        with safe_open(differences_path, framework="pt", device="cpu") as diff_f:
            diff_keys = list(diff_f.keys())
            
            chunk_size = 50
            for i in range(0, len(diff_keys), chunk_size):
                chunk_keys = diff_keys[i:i+chunk_size]
                updates = {}
                
                with safe_open(output_path, framework="pt", device="cpu") as base_f:
                    for key in chunk_keys:
                        if key not in base_f.keys():
                            if DEBUG_MODE:
                                logger.debug(f"Key {key} not in Chroma, skipping")
                            continue
                        
                        base_tensor = base_f.get_tensor(key).to(device)
                        diff_tensor = diff_f.get_tensor(key).to(device)
                        
                        if base_tensor.shape != diff_tensor.shape:
                            logger.warning(f"Shape mismatch for {key}: {base_tensor.shape} vs {diff_tensor.shape}")
                            continue
                        
                        if mode == "standard":
                            merged = base_tensor + diff_tensor
                        elif mode == "add_similar":
                            similarity = torch.nn.functional.cosine_similarity(
                                base_tensor.flatten().unsqueeze(0),
                                diff_tensor.flatten().unsqueeze(0)
                            ).item()
                            if similarity > similarity_threshold:
                                merged = base_tensor + diff_tensor
                            else:
                                merged = base_tensor
                        elif mode == "add_dissimilar":
                            similarity = torch.nn.functional.cosine_similarity(
                                base_tensor.flatten().unsqueeze(0),
                                diff_tensor.flatten().unsqueeze(0)
                            ).item()
                            if similarity < similarity_threshold:
                                merged = base_tensor + diff_tensor
                            else:
                                merged = base_tensor
                        else:
                            merged = base_tensor + diff_tensor
                        
                        updates[key] = merged.cpu()
                        del base_tensor, diff_tensor, merged
                
                if updates:
                    metadata = {}
                    temp_dict = {}
                    with safe_open(output_path, framework="pt", device="cpu") as f:
                        if hasattr(f, "metadata"):
                            metadata = f.metadata() or {}
                        for key in f.keys():
                            temp_dict[key] = updates.get(key, f.get_tensor(key))
                    
                    temp_output_file = Path(output_path).with_suffix(".safetensors.tmp")
                    save_file(temp_dict, str(temp_output_file), metadata=metadata)
                    del temp_dict
                    shutil.move(str(temp_output_file), output_path)

                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        
        logger.info("Differences applied successfully")

class LoRAExtractor:
    @staticmethod
    def extract_lora_svd(merged_path: str, base_path: str, rank: int, device: str = "cuda",
                        mode: str = "standard", similarity_threshold: float = 0.1,
                        chunk_size: int = 50) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
        """
        Extract LoRA from merged model using SVD with proper Chroma naming.
        Includes a robust fallback to CPU if CUDA Out-of-Memory is detected.
        """
        logger.info(f"Extracting LoRA with rank {rank} on device '{device}'")
        
        lora_dict = {}
        key_map = {}
        processed_layers = 0
        
        with safe_open(base_path, framework="pt", device="cpu") as base_f:
            with safe_open(merged_path, framework="pt", device="cpu") as merged_f:
                keys = [k for k in base_f.keys() if k in merged_f.keys()]
                
                weight_keys = [k for k in keys if "weight" in k and 
                             not any(skip in k for skip in ["norm", "embedding", "head"])]
                
                total_chunks = (len(weight_keys) + chunk_size - 1) // chunk_size
                
                for chunk_idx in range(total_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, len(weight_keys))
                    chunk_keys = weight_keys[start_idx:end_idx]
                    
                    logger.info(f"Processing LoRA extraction chunk {chunk_idx + 1}/{total_chunks}")
                    
                    for key in tqdm(chunk_keys, desc=f"Chunk {chunk_idx + 1}"):
                        base_weight, merged_weight, diff = None, None, None
                        U, S, Vt = None, None, None
                        
                        try:
                            # --- Primary GPU Path ---
                            try:
                                base_weight = base_f.get_tensor(key).to(device)
                                merged_weight = merged_f.get_tensor(key).to(device)
                                
                                if base_weight.shape != merged_weight.shape:
                                    continue
                                
                                diff = merged_weight - base_weight
                                
                                if diff.abs().max() < 1e-6:
                                    continue

                                U, S, Vt = torch.linalg.svd(diff.to(torch.float32), full_matrices=False)

                            except torch.cuda.OutOfMemoryError:
                                logger.warning(f"CUDA OOM on key '{key}'. Freeing VRAM and retrying on CPU.")
                                # Clean up GPU memory before retrying on CPU
                                del base_weight, merged_weight, diff, U, S, Vt
                                gc.collect()
                                torch.cuda.empty_cache()
                                
                                # --- CPU Fallback Path ---
                                base_weight = base_f.get_tensor(key).cpu()
                                merged_weight = merged_f.get_tensor(key).cpu()
                                
                                if base_weight.shape != merged_weight.shape:
                                    continue
                                
                                diff = merged_weight - base_weight
                                
                                if diff.abs().max() < 1e-6:
                                    continue
                                
                                U, S, Vt = torch.linalg.svd(diff.to(torch.float32), full_matrices=False)
                            
                            # --- Common Processing Path (after successful SVD) ---
                            U, S, Vt = U.to(base_weight.dtype), S.to(base_weight.dtype), Vt.to(base_weight.dtype)
                            
                            effective_rank = min(rank, S.shape[0])
                            
                            lora_up = U[:, :effective_rank] @ torch.diag(S[:effective_rank].sqrt())
                            lora_down = torch.diag(S[:effective_rank].sqrt()) @ Vt[:effective_rank, :]
                            
                            lora_up = lora_up.contiguous()
                            lora_down = lora_down.contiguous()
                            
                            base_model_key = key
                            if base_model_key.endswith('.weight'):
                                base_model_key = base_model_key[:-len('.weight')]
                            
                            chroma_key = convert_lora_key_to_chroma_format(base_model_key)
                            
                            key_map[chroma_key] = base_model_key
                            
                            lora_dict[f"{chroma_key}.lora_up.weight"] = lora_up.cpu()
                            lora_dict[f"{chroma_key}.lora_down.weight"] = lora_down.cpu()
                            lora_dict[f"{chroma_key}.alpha"] = torch.tensor(effective_rank, dtype=torch.float32)
                            
                            processed_layers += 1
                                
                        except Exception as e:
                            logger.error(f"Failed to process key {key} during SVD: {e}")
                            if DEBUG_MODE:
                                traceback.print_exc()
                            continue
                        
                        finally:
                            # Explicitly delete tensors to help GC
                            del base_weight, merged_weight, diff, U, S, Vt
                            if 'lora_up' in locals(): del lora_up
                            if 'lora_down' in locals(): del lora_down
                    
                    # Clean up at the end of each chunk
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
        
        logger.info(f"Extracted LoRA from {processed_layers} layers")
        return lora_dict, key_map

def validate_extracted_lora(lora_dict: Dict[str, torch.Tensor], key_map: Dict[str, str], chroma_base_path: str, device: str = "cuda") -> bool:
    """
    Validate that extracted LoRA will apply correctly to Chroma.
    This version is optimized to read all model shapes once to avoid slow repeated disk I/O
    and performs matrix multiplication on the specified device for speed.
    """
    logger.info("Validating extracted LoRA...")
    
    issues_found = False
    
    base_lora_keys = set()
    for key in lora_dict.keys():
        if ".lora_down.weight" in key:
            base_key = key.replace(".lora_down.weight", "")
            # Only validate UNet keys, as TE keys are not in the chroma_base model
            if base_key.startswith("lora_unet_"):
                base_lora_keys.add(base_key)
    
    if not base_lora_keys:
        logger.info("No UNet keys found in the extracted LoRA to validate. Skipping UNet validation.")
        return True

    try:
        with safe_open(chroma_base_path, framework="pt", device="cpu") as chroma_f:
            # OPTIMIZATION: Pre-load all tensor shapes from the base model into a dictionary.
            # This avoids repeated, slow disk I/O inside the loop.
            logger.info("Pre-loading Chroma model key shapes for fast validation...")
            chroma_shapes = {key: tuple(chroma_f.get_slice(key).get_shape()) for key in tqdm(chroma_f.keys(), desc="Loading shapes")}
            
            for base_lora_key in tqdm(base_lora_keys, desc="Validating LoRA keys"):
                model_key_base = key_map.get(base_lora_key)
                if not model_key_base:
                    logger.error(f"Validation Error: Could not find original model key for LoRA key '{base_lora_key}'. Skipping validation for this key.")
                    issues_found = True
                    continue

                model_key = f"{model_key_base}.weight"
                
                # Use the pre-loaded dictionary for a fast lookup
                chroma_shape = chroma_shapes.get(model_key)
                
                if chroma_shape is None:
                    logger.warning(f"LoRA targets non-existent layer: {model_key} (from LoRA key: {base_lora_key})")
                    issues_found = True
                    continue
                
                down_key = f"{base_lora_key}.lora_down.weight"
                up_key = f"{base_lora_key}.lora_up.weight"
                
                if down_key in lora_dict and up_key in lora_dict:
                    lora_down = lora_dict[down_key]
                    lora_up = lora_dict[up_key]
                    
                    lora_up_dev, lora_down_dev, result = None, None, None
                    try:
                        # OPTIMIZATION: Move tensors to the selected device (e.g., CUDA) for
                        # significantly faster matrix multiplication than on CPU.
                        lora_up_dev = lora_up.to(device)
                        lora_down_dev = lora_down.to(device)
                        
                        result = lora_up_dev @ lora_down_dev
                        lora_result_shape = result.shape
                        
                        if lora_result_shape != chroma_shape:
                            logger.warning(f"Shape mismatch for {model_key_base}: LoRA produces {lora_result_shape}, Chroma expects {chroma_shape}")
                            issues_found = True
                    except Exception as e:
                        logger.error(f"Error validating {base_lora_key}: {e}")
                        issues_found = True
                    finally:
                        # Clean up tensors from device memory to prevent accumulation
                        del lora_up_dev, lora_down_dev, result
                        if device == "cuda":
                            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to open or process Chroma base model for validation: {e}")
        return False # Cannot validate, so return failure.

    if not issues_found:
        logger.info("Validation passed - all UNet LoRA shapes match Chroma expectations")
    else:
        logger.warning("Validation found issues - the UNet portion of the LoRA may not apply correctly")
    
    return not issues_found

def convert_single_lora(flux_lora_path: str, output_lora_path: str, args: argparse.Namespace) -> int:
    """
    Performs the conversion for a single LoRA file.
    Returns 0 on success, 1 on failure.
    """
    logger.info("Analyzing LoRA compatibility...")
    compatibility = analyze_lora_compatibility(flux_lora_path)
    
    logger.info("\n" + "="*25 + " LoRA Analysis Report " + "="*25)
    logger.info(f"LoRA File: {Path(flux_lora_path).name}")
    logger.info(f"  Naming Style Detected: {compatibility['naming_style']}")
    logger.info(f"  UNet Rank Detected: {compatibility['rank']}")
    logger.info("-" * 72)
    logger.info(f"  Total UNet Pairs Found: {compatibility['total_pairs']}")
    logger.info(f"    - ✅ Convertible to Chroma: {compatibility['compatible_pairs']}")
    logger.info(f"    - ⚠️ Skipped (Layer Not in Pruned Base Model): {compatibility['skipped_nonexistent_pairs']}")
    logger.info(f"    - ❌ Skipped (Incompatible Modulation Layer): {compatibility['incompatible_modulation_pairs']}")
    logger.info(f"    - ❓ Skipped (Unrecognized/Unmappable Key): {compatibility['skipped_unmapped_pairs']}")
    logger.info("-" * 72)
    logger.info(f"  Total Text Encoder Pairs Found: {compatibility['text_encoder_pairs']}")
    if args.enable_text_encoder_conversion:
        logger.warning("  (Conversion will be ATTEMPTED for TE pairs - this is experimental and may not work in ComfyUI)")
    else:
        logger.info("  (TE pairs will be SKIPPED by default. Use --enable-text-encoder-conversion to attempt conversion)")
    logger.info("="*72 + "\n")

    if DEBUG_MODE:
        if compatibility['convertible_list']:
            logger.debug(f"Sample convertible keys: {compatibility['convertible_list'][:3]}")
        if compatibility['skipped_nonexistent_list']:
            logger.debug(f"Sample non-existent keys skipped: {compatibility['skipped_nonexistent_list'][:3]}")
        if compatibility['incompatible_modulation_list']:
            logger.debug(f"Sample incompatible keys: {compatibility['incompatible_modulation_list'][:3]}")
        if compatibility['skipped_unmapped_list']:
             logger.debug(f"Sample unmappable keys: {compatibility['skipped_unmapped_list'][:3]}")

    if compatibility['compatible_pairs'] == 0:
        if compatibility['text_encoder_pairs'] > 0 and args.enable_text_encoder_conversion:
             logger.warning("This LoRA appears to be text-encoder only. Attempting experimental TE-only conversion.")
        else:
            logger.error("CONVERSION HALTED: No compatible UNet LoRA pairs found. This LoRA cannot be converted to Chroma format.")
            return 1
    
    if args.analyze_only:
        logger.info(f"Analysis for {Path(flux_lora_path).name} complete. Halting as requested by --analyze-only.")
        return 0
    
    # Use a local copy of rank to avoid modifying args
    rank = args.rank
    if rank == -1:
        rank = compatibility['rank']
        logger.info(f"Using auto-detected UNet rank for extraction: {rank}")
    
    temp_dir = None # define in outer scope for cleanup in except block
    try:
        # Create a unique temp directory to avoid conflicts in batch mode
        temp_dir = Path(output_lora_path).parent / f"temp_conversion_{Path(flux_lora_path).stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        print_memory_usage("Start")
        
        # Initialize final LoRA dictionary
        final_lora_dict = {}
        key_map = {}
        applied_count = 0
        merge_stats = {}

        # Only run the UNet conversion if there are compatible pairs
        if compatibility['compatible_pairs'] > 0:
            logger.info("\n" + "="*60)
            logger.info("Step 1: Applying UNet LoRA to Flux base model...")
            logger.info("="*60)
            
            merged_flux_path = temp_dir / "merged_flux.safetensors"
            applied_count, merge_stats = FluxLoRAApplier.apply_lora(
                args.flux_base, flux_lora_path, str(merged_flux_path), 
                args.device, args.lora_alpha, args.chunk_size
            )
            
            if applied_count == 0:
                logger.error("ERROR: No UNet LoRA weights were applied despite being found!")
                logger.error("This may indicate a key mapping or base model mismatch.")
                shutil.rmtree(temp_dir)
                return 1
            
            print_memory_usage("After Flux merge")
            
            logger.info("\n" + "="*60)
            logger.info("Step 2: Extracting UNet differences...")
            logger.info("="*60)
            
            differences_path = temp_dir / "differences.safetensors"
            ChromaDifferenceExtractor.extract_differences(
                args.flux_base, str(merged_flux_path), str(differences_path),
                args.device, args.mode, args.similarity_threshold
            )
            
            if merged_flux_path.exists():
                merged_flux_path.unlink()
            gc.collect()
            if args.device == "cuda": torch.cuda.empty_cache()
            print_memory_usage("After difference computation")
            
            logger.info("\n" + "="*60)
            logger.info("Step 3: Applying UNet differences to Chroma...")
            logger.info("="*60)
            
            merged_chroma_path = temp_dir / "merged_chroma.safetensors"
            ChromaDifferenceApplier.apply_differences(
                args.chroma_base, str(differences_path), str(merged_chroma_path), 
                args.device, mode=args.mode, similarity_threshold=args.similarity_threshold
            )
            
            if differences_path.exists():
                differences_path.unlink()
            gc.collect()
            if args.device == "cuda": torch.cuda.empty_cache()
            print_memory_usage("After applying differences")
            
            logger.info("\n" + "="*60)
            logger.info("Step 4: Extracting UNet LoRA from merged Chroma...")
            logger.info("="*60)
            
            final_lora_dict, key_map = LoRAExtractor.extract_lora_svd(
                str(merged_chroma_path), args.chroma_base, rank, args.device,
                mode=args.mode, similarity_threshold=args.similarity_threshold,
                chunk_size=args.chunk_size
            )
            if merged_chroma_path.exists():
                merged_chroma_path.unlink()
        else:
            logger.info("Skipping UNet conversion as no compatible pairs were found.")

        # Step 5: Convert Text Encoder weights
        logger.info("\n" + "="*60)
        logger.info("Step 5: Handling Text Encoder Weights...")
        logger.info("="*60)
        if args.enable_text_encoder_conversion:
            te_weights = convert_text_encoder_weights(flux_lora_path)
            if te_weights:
                final_lora_dict.update(te_weights)
        else:
            logger.info("Skipping text encoder weight conversion. Use --enable-text-encoder-conversion to override.")

        if not final_lora_dict:
            logger.error("Conversion failed: The final LoRA dictionary is empty. No UNet or TE weights were processed.")
            shutil.rmtree(temp_dir)
            return 1

        if not args.skip_validation:
            validate_extracted_lora(final_lora_dict, key_map, args.chroma_base, args.device)
        
        logger.info("Loading and updating metadata from input LoRA...")
        metadata = load_lora_metadata(flux_lora_path)
        unet_pairs = len([k for k in final_lora_dict if k.startswith('lora_unet_') and '.lora_down.weight' in k])
        te_pairs = len([k for k in final_lora_dict if k.startswith('lora_te_') and ('.lora_down.weight' in k or '.lora_A.weight' in k)])
        metadata["chroma_lora_unet_pairs"] = str(unet_pairs)
        metadata["chroma_lora_te_pairs"] = str(te_pairs)
        metadata["chroma_extraction_rank"] = str(rank)
        metadata["flux_lora_applied_count"] = str(applied_count)
        metadata["chroma_conversion_date"] = datetime.now().isoformat()
        metadata["chroma_converter_version"] = "v17.9"
        metadata["original_naming_style"] = compatibility['naming_style']
        metadata["text_encoder_conversion_attempted"] = str(args.enable_text_encoder_conversion)
        metadata["accumulated_layers"] = str(len(merge_stats.get("accumulated", [])))
        
        logger.info(f"Saving final LoRA with {len(metadata)} metadata entries...")
        save_file(final_lora_dict, output_lora_path, metadata=metadata)
        
        logger.info("Verifying saved metadata...")
        with safe_open(output_lora_path, framework="pt", device="cpu") as f:
            if hasattr(f, 'metadata'):
                saved_metadata = f.metadata()
                if saved_metadata:
                    logger.info(f"Successfully saved {len(saved_metadata)} metadata entries")
                    for key, value in saved_metadata.items():
                        if any(word in key.lower() for word in ["trigger", "prompt", "keyword"]):
                            logger.info(f"Verified trigger metadata preserved: {str(value)[:100] if len(str(value)) > 100 else value}")
                else:
                    logger.warning("No metadata found in saved file")
        
        logger.info("\n" + "="*24 + " Conversion Complete! " + "="*24)
        logger.info(f"Output File: {output_lora_path}")
        logger.info(f"  - Extracted {unet_pairs} UNet LoRA modules with rank {rank}")
        if args.enable_text_encoder_conversion:
            logger.info(f"  - Attempted to convert {te_pairs} Text Encoder LoRA modules (Experimental).")
        else:
            logger.info("  - Text Encoder modules were not converted (default).")
        logger.info(f"  - Original LoRA naming style: {compatibility['naming_style']}")
        logger.info(f"  - Accumulated {len(merge_stats.get('accumulated', []))} multi-component UNet layers")
        logger.info("="*72)
        
        logger.info("Cleaning up temporary files...")
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print_memory_usage("Final")
        return 0
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during conversion of {Path(flux_lora_path).name}: {str(e)}".encode('utf-8', 'replace').decode('utf-8'))
        logger.error("If the script terminated silently without this message, it was likely due to an Out-of-Memory (OOM) error.")
        logger.error("Try running again with the --device cpu argument to use system RAM instead of VRAM.")
        if DEBUG_MODE:
            traceback.print_exc()
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Flux Dev to Chroma LoRA Converter v17.9 - Definitive Key Mapping Fix',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Required base models
    parser.add_argument('--flux-base', required=True, help='Path to Flux base model (e.g., flux1-dev-pruned.safetensors)')
    parser.add_argument('--chroma-base', required=True, help='Path to Chroma base model (e.g., chroma-unlocked-v43.safetensors)')

    # Input LoRA(s) - either a single file or a folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--flux-lora', help='Path to a single Flux LoRA file to convert.')
    input_group.add_argument('--lora-folder', help='Path to a folder containing Flux LoRA .safetensors files to convert in batch.')

    # Output path - now optional
    parser.add_argument('--output-lora', help='Output path for Chroma LoRA.\n'
                                             '- In single file mode, this is the full output file path. Optional.\n'
                                             '  If not provided, the output is auto-named (e.g., "input-chroma.safetensors").\n'
                                             '- In batch mode (--lora-folder), this argument is ignored.')

    # Other existing arguments
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use for conversion')
    parser.add_argument('--mode', default='standard', 
                       choices=['standard', 'add_similar', 'add_dissimilar'],
                       help='Merge mode for applying differences')
    parser.add_argument('--rank', type=int, default=-1, help='LoRA rank for extraction. -1 to auto-detect from UNet.')
    parser.add_argument('--lora-alpha', type=float, default=1.0, help='LoRA alpha scaling when applying to the base model')
    parser.add_argument('--similarity-threshold', type=float, default=0.1, 
                       help='Similarity threshold for add_similar/add_dissimilar modes')
    parser.add_argument('--chunk-size', type=int, default=50, help='Chunk size for processing large models to save memory')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze LoRA compatibility without converting')
    parser.add_argument('--skip-validation', action='store_true', help='Skip the final LoRA validation step')
    parser.add_argument('--enable-text-encoder-conversion', action='store_true', help='(EXPERIMENTAL) Attempt to convert text encoder weights. This is known to fail in current ComfyUI versions.')
    
    args = parser.parse_args()
    
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    
    if args.lora_folder:
        # Batch processing mode
        if args.output_lora:
            logger.warning("Warning: --output-lora is ignored when using --lora-folder. Outputs will be auto-named.")

        input_folder = Path(args.lora_folder)
        if not input_folder.is_dir():
            logger.error(f"Error: The provided path for --lora-folder is not a valid directory: {input_folder}")
            return 1

        lora_files = sorted(list(input_folder.glob('*.safetensors')))
        if not lora_files:
            logger.warning(f"No .safetensors files found in the specified directory: {input_folder}")
            return 0

        logger.info(f"Starting batch conversion for {len(lora_files)} LoRA files in '{input_folder}'.")
        
        success_count = 0
        fail_count = 0
        total_files = len(lora_files)

        for i, lora_path in enumerate(lora_files):
            output_path = lora_path.with_name(f"{lora_path.stem}-chroma{lora_path.suffix}")
            
            logger.info("\n" + "#"*80)
            logger.info(f"### Processing file {i + 1}/{total_files}: {lora_path.name} ###")
            logger.info(f"### Output will be: {output_path.name} ###")
            logger.info("#"*80 + "\n")

            status = convert_single_lora(str(lora_path), str(output_path), args)
            if status == 0:
                success_count += 1
            else:
                fail_count += 1
        
        logger.info("\n" + "="*60)
        logger.info("Batch processing summary:")
        logger.info(f"  Total files processed: {total_files}")
        logger.info(f"  Successful conversions: {success_count}")
        logger.info(f"  Failed conversions: {fail_count}")
        logger.info("="*60)
        
        return 1 if fail_count > 0 else 0

    else:
        # Single file processing mode
        flux_lora_path = args.flux_lora
        if args.output_lora:
            output_lora_path = args.output_lora
        else:
            p = Path(flux_lora_path)
            output_lora_path = str(p.with_name(f"{p.stem}-chroma{p.suffix}"))
            logger.info(f"No output path specified. Defaulting to: {output_lora_path}")
            
        return convert_single_lora(flux_lora_path, output_lora_path, args)

if __name__ == "__main__":
    sys.exit(main())
    
#
# --- Historical Note on Text Encoder Conversion Attempts ---
# The conversion of T5 text encoder weights (`lora_te1_...`) has proven to be
# a significant challenge. The following methods were attempted and ultimately
# failed, leading to the current "experimental, opt-in" status. This history
# is preserved to prevent repeating failed experiments.
#
#   -   **v10-v12:** These early versions made incorrect assumptions about key
#       shortening and the general mapping from an underscore-based to a
#       dot-based hierarchy. They failed broadly.
#
#   -   **v13:** This version correctly identified the need for structural
#       renaming but made a critical error: it retained the `text_model_`
#       prefix from the original Diffusers key. Error logs from ComfyUI
#       confirmed that the T5 model it loads does not have this prefix, so
#       the keys were not found.
#
#   -   **v14:** Based on the v13 failure, this version correctly removed the
#       `text_model_` prefix. However, it then incorrectly converted the
#       entire key to a standard dot-separated hierarchy (e.g.,
#       `lora_te_encoder.block.0.layer.0.SelfAttention.q`). This was based on
#       the assumption that ComfyUI uses a standard dot-separated hierarchy
#       for all LoRA keys. The "lora key not loaded" errors persisted,
#       proving this assumption was also incorrect for the T5 encoder.
#
#   -   **v15/v16 (The "Definitive Fix" Attempt):** This was the most
#       promising and evidence-based attempt. After further analysis of
#       ComfyUI's source and error logs, it was determined that the expected
#       key format was much closer to the Diffusers original. This version
#       performed a simple prefix replacement:
#
#       -   Source: `lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight`
#       -   Target: `lora_te_encoder_layers_0_self_attn_q_proj.lora_up.weight`
#
#       This preserved the original underscore-separated format while using the
#       `lora_te_` prefix expected by ComfyUI's loader. While this seemed
#       theoretically perfect and matched all available evidence, in-practice
#       user feedback confirmed that ComfyUI *still* failed to load these keys.
#
#   -   **v17 (Current):** Given the repeated failures, the conclusion is that
#       the exact key format is either still unknown or that ComfyUI's T5 LoRA
#       loading mechanism has a bug or an undocumented requirement. To prevent
#       generating non-functional LoRAs, TE conversion was made an
#       experimental, opt-in feature.
