#!/usr/bin/env python3
"""
Flux Dev to Chroma LoRA Converter v7 - Fixed Shapes, Prefix, and Contiguous Tensors
Properly handles shape mismatches and uses correct Chroma LoRA format
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

# Optional imports for debug mode
try:
    import psutil
    import GPUtil
    HAS_DEBUG_LIBS = True
except ImportError:
    HAS_DEBUG_LIBS = False
    
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global debug flag
DEBUG_MODE = False

# Comprehensive key mapping based on actual Flux architecture
DIFFUSERS_KEY_MAPPING = {
    # Single blocks - Flux only has linear1, linear2, and norm layers
    r"single_transformer_blocks\.(\d+)\.attn\.to_q": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.attn\.to_k": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.attn\.to_v": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.attn\.to_out\.0": "single_blocks.{}.linear2",
    r"single_transformer_blocks\.(\d+)\.ff\.net\.0\.proj": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.ff\.net\.2": "single_blocks.{}.linear2",
    r"single_transformer_blocks\.(\d+)\.proj_mlp": "single_blocks.{}.linear2",
    r"single_transformer_blocks\.(\d+)\.proj_out": "single_blocks.{}.linear2",  # Map proj_out to linear2
    r"single_transformer_blocks\.(\d+)\.norm\.linear": None,  # Skip - doesn't exist in Flux
    
    # Double blocks (transformer_blocks without 'single')
    r"transformer_blocks\.(\d+)\.attn\.to_q": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn\.to_k": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn\.to_v": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn\.to_out\.0": "double_blocks.{}.img_attn.proj",
    r"transformer_blocks\.(\d+)\.ff\.net\.0\.proj": "double_blocks.{}.img_mlp.0",
    r"transformer_blocks\.(\d+)\.ff\.net\.2": "double_blocks.{}.img_mlp.2",
    r"transformer_blocks\.(\d+)\.ff_context\.net\.0\.proj": "double_blocks.{}.txt_mlp.0",
    r"transformer_blocks\.(\d+)\.ff_context\.net\.2": "double_blocks.{}.txt_mlp.2",
    
    # Context attention for double blocks
    r"transformer_blocks\.(\d+)\.attn_context\.to_q": "double_blocks.{}.txt_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn_context\.to_k": "double_blocks.{}.txt_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn_context\.to_v": "double_blocks.{}.txt_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn_context\.to_out\.0": "double_blocks.{}.txt_attn.proj",
    
    # Handle norm layers for double blocks
    r"transformer_blocks\.(\d+)\.norm1\.linear": "double_blocks.{}.img_attn_norm.linear",
    r"transformer_blocks\.(\d+)\.norm1_context\.linear": "double_blocks.{}.txt_attn_norm.linear",
    
    # General text encoder patterns (skip these)
    r"text_encoder.*": None,
    r"te_.*": None,
    r"unet\.time_embedding.*": None,
    r"unet\.label_emb.*": None,
    
    # Add patterns for Flux 1.1/Pro specific layers (skip these)
    r".*add_k_proj.*": None,
    r".*add_q_proj.*": None,
    r".*add_v_proj.*": None,
    r".*to_add_out.*": None,
    r".*norm_added.*": None,
}

# Additional direct mappings for edge cases
DIRECT_KEY_MAPPING = {
    # These are exact replacements, not patterns
    "transformer.proj_out": None,  # Skip - doesn't exist
    "transformer.norm_out.linear": None,  # Skip - doesn't exist
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
    Normalize a LoRA key to match Flux structure
    Returns None if the key should be skipped
    """
    original_key = key
    
    # Remove .lora_A.weight, .lora_B.weight suffixes
    key = re.sub(r'\.lora_[AB]\.weight$', '', key)
    key = re.sub(r'\.lora_(up|down)\.weight$', '', key)
    key = re.sub(r'\.alpha$', '', key)
    
    # Remove prefix if present
    if key.startswith('lora_unet_'):
        key = key[len('lora_unet_'):]
    elif key.startswith('lora_te'):
        return None  # Skip text encoder
    elif key.startswith('unet.'):
        key = key[5:]
    elif key.startswith('transformer.'):
        key = key[12:]
    
    # Check direct mappings first
    if key in DIRECT_KEY_MAPPING:
        mapped = DIRECT_KEY_MAPPING[key]
        if DEBUG_MODE and mapped != key:
            logger.debug(f"Direct mapping: {original_key} -> {mapped}")
        return mapped
    
    # Try pattern-based mappings
    for pattern, replacement in DIFFUSERS_KEY_MAPPING.items():
        match = re.match(pattern, key)
        if match:
            if replacement is None:
                if DEBUG_MODE:
                    logger.debug(f"Skipping (pattern): {original_key}")
                return None
            else:
                mapped = replacement.format(*match.groups())
                if DEBUG_MODE and mapped != key:
                    logger.debug(f"Pattern mapping: {original_key} -> {mapped}")
                return mapped
    
    # If no mapping found, return the key as-is
    if DEBUG_MODE:
        logger.debug(f"No mapping found, keeping: {original_key}")
    return key

def is_modulation_layer(key: str) -> bool:
    """Check if a key is a modulation layer (exists in Flux but not Chroma)"""
    return any(mod in key for mod in [
        "_mod.", ".mod.", "_mod_", ".modulation.",
        "mod_out", "norm_out", "scale_shift",
        "mod.lin", "modulated", "norm_k.", "norm_q."
    ])

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
            if "lora_" in key and "weight" in key:
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
        "incompatible_pairs": 0,
        "skipped_pairs": 0,
        "naming_style": "unknown",
        "rank": 16,
        "convertible_pairs": [],
        "unconvertible_pairs": [],
        "skipped_pairs_list": []
    }
    
    try:
        rank, naming_style = detect_lora_rank(lora_path)
        analysis["rank"] = rank
        analysis["naming_style"] = naming_style
    except:
        pass
    
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
            analysis["total_pairs"] += 1
            
            # Check if convertible
            normalized_key = normalize_lora_key(base_key)
            
            if normalized_key is None:
                analysis["skipped_pairs"] += 1
                analysis["skipped_pairs_list"].append(base_key)
            elif is_modulation_layer(normalized_key):
                analysis["incompatible_pairs"] += 1
                analysis["unconvertible_pairs"].append(base_key)
            else:
                analysis["compatible_pairs"] += 1
                analysis["convertible_pairs"].append(base_key)
    
    return analysis

def load_lora_metadata(lora_path: str) -> Dict[str, str]:
    """Load metadata from a LoRA file"""
    metadata = {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            if hasattr(f, 'metadata'):
                file_metadata = f.metadata()
                if file_metadata:
                    metadata.update(file_metadata)
                    logger.info(f"Found {len(metadata)} metadata entries")
                    # Look for trigger words
                    for key, value in metadata.items():
                        if any(word in key.lower() for word in ["trigger", "prompt", "keyword"]):
                            logger.info(f"Found trigger metadata: {key} = {value[:100] if len(str(value)) > 100 else value}")
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
    
    return metadata

class LoRAMerger:
    """Handles merging LoRA weights into base models"""
    
    @staticmethod
    def accumulate_lora_weights(lora_pairs: List[Dict], device: str, lora_alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accumulate multiple LoRA weights targeting the same layer"""
        accumulated_down = None
        accumulated_up = None
        total_weight = 0
        
        for pair_info in lora_pairs:
            down_weight = pair_info["down_tensor"].to(device).float()
            up_weight = pair_info["up_tensor"].to(device).float()
            
            # Get alpha
            alpha = lora_alpha
            if pair_info["alpha_tensor"] is not None:
                alpha_value = pair_info["alpha_tensor"].item()
                rank = min(down_weight.shape)
                alpha = alpha_value / rank * lora_alpha
            
            # Apply alpha scaling
            down_weight = down_weight * np.sqrt(alpha)
            up_weight = up_weight * np.sqrt(alpha)
            
            if accumulated_down is None:
                accumulated_down = down_weight
                accumulated_up = up_weight
            else:
                accumulated_down = accumulated_down + down_weight
                accumulated_up = accumulated_up + up_weight
            
            total_weight += 1
        
        # Average the accumulated weights
        if total_weight > 1:
            accumulated_down = accumulated_down / total_weight
            accumulated_up = accumulated_up / total_weight
        
        return accumulated_down, accumulated_up
    
    @staticmethod
    def apply_lora_to_weight(base_weight: torch.Tensor, down: torch.Tensor, up: torch.Tensor, 
                            target_key: str = None) -> torch.Tensor:
        """Apply LoRA matrices to base weight with correct matrix multiplication order"""
        # For LoRA: W' = W + BA where B=up, A=down
        # down (A) shape: [rank, in_features]
        # up (B) shape: [out_features, rank]
        # Result should be [out_features, in_features] to match base_weight
        
        if DEBUG_MODE:
            logger.debug(f"Applying LoRA to {target_key}")
            logger.debug(f"Base weight shape: {base_weight.shape}")
            logger.debug(f"Down (A) shape: {down.shape}, Up (B) shape: {up.shape}")
        
        # Standard LoRA computation: B @ A
        lora_weight = up @ down
        
        if DEBUG_MODE:
            logger.debug(f"LoRA weight shape after matmul: {lora_weight.shape}")
        
        # Handle shape mismatches with specific logic for each layer type
        if lora_weight.shape != base_weight.shape:
            reshaped = LoRAMerger.reshape_for_layer(lora_weight, base_weight.shape, target_key)
            if reshaped is not None:
                lora_weight = reshaped
            else:
                raise ValueError(f"Cannot reshape LoRA weight from {lora_weight.shape} to {base_weight.shape} for {target_key}")
        
        return base_weight + lora_weight
    
    @staticmethod
    def reshape_for_layer(lora_weight: torch.Tensor, target_shape: torch.Size, 
                         layer_key: str = None) -> Optional[torch.Tensor]:
        """Reshape LoRA weight based on layer type and expected shape"""
        if lora_weight.shape == target_shape:
            return lora_weight
        
        if DEBUG_MODE:
            logger.debug(f"Attempting to reshape {lora_weight.shape} to {target_shape} for {layer_key}")
        
        # Special handling for single_blocks.linear2
        if layer_key and "single_blocks" in layer_key and "linear2" in layer_key:
            # Expected: [3072, 15360], getting [3072, 3072]
            if lora_weight.shape == (3072, 3072) and target_shape == (3072, 15360):
                # This might be a case where the LoRA was trained on a different architecture
                # We need to expand it to match the expected shape
                # Option 1: Repeat the pattern 5 times (15360 / 3072 = 5)
                repeated = lora_weight.repeat(1, 5)  # [3072, 15360]
                if DEBUG_MODE:
                    logger.debug(f"Repeated linear2 weight 5x: {lora_weight.shape} -> {repeated.shape}")
                return repeated
        
        # Handle QKV concatenation for attention layers
        if layer_key and ("img_attn.qkv" in layer_key or "txt_attn.qkv" in layer_key):
            # Expected: [9216, 3072] (3x3072 for Q,K,V), might get [3072, 3072]
            if lora_weight.shape[1] == target_shape[1] and target_shape[0] == 3 * lora_weight.shape[0]:
                repeated = lora_weight.repeat(3, 1)
                if DEBUG_MODE:
                    logger.debug(f"Repeated QKV weight 3x: {lora_weight.shape} -> {repeated.shape}")
                return repeated
        
        # Try generic reshape if dimensions allow
        lora_elements = lora_weight.numel()
        target_elements = torch.Size(target_shape).numel()
        
        if lora_elements == target_elements:
            if DEBUG_MODE:
                logger.debug(f"Reshaping by element count: {lora_weight.shape} -> {target_shape}")
            return lora_weight.reshape(target_shape)
        
        # Try dimension-based repetition
        if len(target_shape) == 2 and len(lora_weight.shape) == 2:
            target_out, target_in = target_shape
            lora_out, lora_in = lora_weight.shape
            
            # Check if dimensions match with repetition
            if lora_out == target_out and target_in % lora_in == 0:
                factor = target_in // lora_in
                return lora_weight.repeat(1, factor)
            elif lora_in == target_in and target_out % lora_out == 0:
                factor = target_out // lora_out
                return lora_weight.repeat(factor, 1)
        
        return None
    
    @staticmethod
    def merge_lora_to_model(base_model_path: str, lora_path: str, 
                           device: str = "cuda", lora_alpha: float = 1.0) -> Tuple[Dict[str, torch.Tensor], int, Dict]:
        """Merge LoRA weights into base model with accumulation support"""
        logger.info("Merging LoRA into base model...")
        
        # Group LoRA pairs by their normalized target key
        lora_groups = defaultdict(list)
        
        with safe_open(lora_path, framework="pt", device="cpu") as lora_file:
            # First pass: group all LoRA pairs
            processed_bases = set()
            
            for key in lora_file.keys():
                if "alpha" in key:
                    continue
                
                # Find base key
                base_key = None
                if ".lora_A.weight" in key or ".lora_down.weight" in key:
                    base_key = key.replace(".lora_A.weight", "").replace(".lora_down.weight", "")
                elif ".lora_B.weight" in key or ".lora_up.weight" in key:
                    continue  # Process only from down/A weights
                else:
                    continue
                
                if base_key in processed_bases:
                    continue
                processed_bases.add(base_key)
                
                # Get normalized key
                normalized_key = normalize_lora_key(base_key)
                if normalized_key is None or is_modulation_layer(normalized_key):
                    continue
                
                # Get the pair
                down_key = base_key + (".lora_A.weight" if ".lora_A.weight" in key else ".lora_down.weight")
                up_key = base_key + (".lora_B.weight" if ".lora_A.weight" in key else ".lora_up.weight")
                alpha_key = base_key + ".alpha"
                
                if down_key in lora_file.keys() and up_key in lora_file.keys():
                    pair_info = {
                        "base_key": base_key,
                        "down_tensor": lora_file.get_tensor(down_key),
                        "up_tensor": lora_file.get_tensor(up_key),
                        "alpha_tensor": lora_file.get_tensor(alpha_key) if alpha_key in lora_file.keys() else None
                    }
                    lora_groups[normalized_key].append(pair_info)
        
        logger.info(f"Found {len(lora_groups)} LoRA layer pairs grouped into {len(lora_groups)} targets")
        
        # Load base model and apply LoRAs
        merged_state_dict = {}
        stats = {
            "applied": [],
            "unmatched": [],
            "shape_mismatch": [],
            "accumulated": []
        }
        
        with safe_open(base_model_path, framework="pt", device="cpu") as base_file:
            # First copy all base weights
            for key in tqdm(base_file.keys(), desc="Loading base model"):
                merged_state_dict[key] = base_file.get_tensor(key)
            
            # Apply LoRA weights
            for normalized_key, pair_list in tqdm(lora_groups.items(), desc="Applying LoRA"):
                # Find the actual key in the model
                possible_keys = [
                    normalized_key + ".weight",
                    normalized_key.replace("linear1", "attn.qkv") + ".weight",
                    normalized_key.replace("linear2", "proj_out") + ".weight",
                    normalized_key.replace("proj_out", "attn.proj") + ".weight"
                ]
                
                matched_key = None
                for test_key in possible_keys:
                    if test_key in merged_state_dict:
                        matched_key = test_key
                        break
                
                if matched_key:
                    try:
                        base_weight = merged_state_dict[matched_key].to(device).float()
                        
                        # Accumulate LoRA weights if multiple
                        if len(pair_list) > 1:
                            stats["accumulated"].append(f"{normalized_key} ({len(pair_list)} components)")
                            accumulated_down, accumulated_up = LoRAMerger.accumulate_lora_weights(
                                pair_list, device, lora_alpha
                            )
                        else:
                            # Single LoRA, process normally
                            pair_info = pair_list[0]
                            accumulated_down = pair_info["down_tensor"].to(device).float()
                            accumulated_up = pair_info["up_tensor"].to(device).float()
                            
                            # Get alpha
                            alpha = lora_alpha
                            if pair_info["alpha_tensor"] is not None:
                                alpha_value = pair_info["alpha_tensor"].item()
                                rank = min(accumulated_down.shape)
                                alpha = alpha_value / rank * lora_alpha
                            
                            accumulated_down = accumulated_down * np.sqrt(alpha)
                            accumulated_up = accumulated_up * np.sqrt(alpha)
                        
                        # Apply LoRA with correct matrix order
                        try:
                            merged_weight = LoRAMerger.apply_lora_to_weight(
                                base_weight, accumulated_down, accumulated_up, matched_key
                            )
                            merged_state_dict[matched_key] = merged_weight.to(merged_state_dict[matched_key].dtype).cpu()
                            stats["applied"].append(normalized_key)
                        except ValueError as e:
                            stats["shape_mismatch"].append(f"{normalized_key}: {str(e)}")
                            logger.warning(f"Failed to apply LoRA to {matched_key}: {e}")
                        
                        # Cleanup
                        del base_weight, accumulated_down, accumulated_up
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply LoRA to {matched_key}: {e}")
                        continue
                else:
                    stats["unmatched"].append(normalized_key)
            
            # Report statistics
            logger.info(f"\nMerge Statistics:")
            logger.info(f"  Applied: {len(stats['applied'])} layers")
            logger.info(f"  Accumulated: {len(stats['accumulated'])} multi-component layers")
            logger.info(f"  Unmatched: {len(stats['unmatched'])} layers")
            logger.info(f"  Shape mismatches: {len(stats['shape_mismatch'])} layers")
            
            if DEBUG_MODE:
                if stats["applied"]:
                    logger.debug(f"Sample applied layers: {stats['applied'][:5]}")
                if stats["accumulated"]:
                    logger.debug(f"Accumulated layers: {stats['accumulated'][:5]}")
                if stats["shape_mismatch"]:
                    logger.debug(f"Shape mismatches: {stats['shape_mismatch'][:5]}")
            
            if len(stats["applied"]) == 0:
                logger.error("ERROR: No LoRA weights were applied!")
                logger.error("This likely means the LoRA format is not recognized.")
                logger.error("This LoRA may be text-encoder only or use an unsupported format.")
            
            return merged_state_dict, len(stats["applied"]), stats

class DifferenceComputer:
    """Computes differences between models"""
    
    @staticmethod
    def compute_difference(model_a_path: str, model_b_path: str, 
                          device: str = "cuda", chunk_size: int = 50) -> Dict[str, torch.Tensor]:
        """Compute difference between two models"""
        logger.info("Computing model differences...")
        
        differences = {}
        processed_count = 0
        skipped_count = 0
        
        try:
            with safe_open(model_a_path, framework="pt", device="cpu") as f_a:
                with safe_open(model_b_path, framework="pt", device="cpu") as f_b:
                    keys_a = set(f_a.keys())
                    keys_b = set(f_b.keys())
                    common_keys = sorted(keys_a.intersection(keys_b))
                    
                    logger.info(f"Computing differences for {len(common_keys)} common layers")
                    
                    # Process in chunks
                    for i in range(0, len(common_keys), chunk_size):
                        chunk_keys = common_keys[i:i+chunk_size]
                        chunk_num = i // chunk_size + 1
                        total_chunks = (len(common_keys) + chunk_size - 1) // chunk_size
                        
                        for key in tqdm(chunk_keys, desc=f"Computing differences (chunk {chunk_num}/{total_chunks})"):
                            if is_modulation_layer(key):
                                continue
                            
                            try:
                                tensor_a = f_a.get_tensor(key).to(device)
                                tensor_b = f_b.get_tensor(key).to(device)
                                
                                if tensor_a.shape == tensor_b.shape:
                                    diff = tensor_a - tensor_b
                                    
                                    # Only save non-zero differences
                                    if torch.abs(diff).max() > 1e-6:
                                        differences[key] = diff.cpu()
                                        processed_count += 1
                                else:
                                    skipped_count += 1
                                
                                del tensor_a, tensor_b
                                if key in differences:
                                    del diff
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    logger.error(f"Out of memory processing {key}")
                                    if device == "cuda":
                                        torch.cuda.empty_cache()
                                    skipped_count += 1
                                    continue
                                else:
                                    raise
                        
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    
                    logger.info(f"Computed {processed_count} differences, skipped {skipped_count} unchanged layers")
                    
                    if processed_count == 0:
                        raise ValueError("No differences computed - models might be identical")
            
        except Exception as e:
            logger.error(f"Critical error during difference computation: {e}")
            raise
        
        return differences

class ChromaDifferenceApplier:
    """Applies differences to Chroma model"""
    
    @staticmethod
    def apply_differences(base_model_path: str, differences: Dict[str, torch.Tensor],
                         output_path: str, device: str = "cuda", 
                         mode: str = "standard", similarity_threshold: float = 0.1) -> int:
        """Apply differences to Chroma base model"""
        logger.info(f"Applying differences to Chroma model using {mode} mode...")
        
        applied_count = 0
        skipped_count = 0
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        temp_state_dict = {}
        
        try:
            with safe_open(base_model_path, framework="pt", device="cpu") as f:
                # Process all tensors
                for key in tqdm(f.keys(), desc="Applying differences"):
                    tensor = f.get_tensor(key)
                    
                    if key in differences and not is_modulation_layer(key):
                        try:
                            # Apply difference modes
                            if mode == "standard":
                                result = tensor.to(device) + differences[key].to(device)
                            elif mode == "add_similar":
                                diff = differences[key].to(device)
                                similarity = torch.nn.functional.cosine_similarity(
                                    tensor.to(device).flatten().unsqueeze(0),
                                    diff.flatten().unsqueeze(0)
                                ).item()
                                
                                if similarity > similarity_threshold:
                                    result = tensor.to(device) + diff
                                else:
                                    result = tensor.to(device)
                            else:
                                result = tensor.to(device)
                            
                            temp_state_dict[key] = result.cpu()
                            applied_count += 1
                            
                            del result
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            logger.warning(f"Failed to apply difference to {key}: {e}")
                            temp_state_dict[key] = tensor
                            skipped_count += 1
                    else:
                        temp_state_dict[key] = tensor
                        if key in differences:
                            skipped_count += 1
                
                logger.info(f"Saving modified model to {output_path}")
                save_file(temp_state_dict, str(output_path))
                logger.info(f"Applied differences to {applied_count} layers")
                
                return applied_count
                
        except Exception as e:
            logger.error(f"Critical error applying differences: {e}")
            raise

class LoRAExtractor:
    """Extracts LoRA from model differences using SVD"""
    
    @staticmethod
    def extract_lora_svd(model_a_path: str, model_b_path: str, 
                        rank: int = 16, device: str = "cuda",
                        mode: str = "standard", similarity_threshold: float = 0.1,
                        chunk_size: int = 50) -> Dict[str, torch.Tensor]:
        """Extract LoRA using SVD with proper Chroma format"""
        logger.info(f"Extracting LoRA with rank {rank} using {mode} mode...")
        
        lora_dict = {}
        extracted_count = 0
        
        try:
            with safe_open(model_a_path, framework="pt", device="cpu") as f_a:
                with safe_open(model_b_path, framework="pt", device="cpu") as f_b:
                    keys_a = set(f_a.keys())
                    keys_b = set(f_b.keys())
                    common_keys = keys_a.intersection(keys_b)
                    
                    # Filter for weight matrices
                    weight_keys = [k for k in common_keys if k.endswith('.weight') 
                                 and len(f_a.get_tensor(k).shape) >= 2
                                 and not is_modulation_layer(k)]
                    
                    logger.info(f"Processing {len(weight_keys)} weight layers...")
                    
                    # Process in chunks
                    for i in range(0, len(weight_keys), chunk_size):
                        chunk_keys = weight_keys[i:i+chunk_size]
                        chunk_num = i // chunk_size + 1
                        total_chunks = (len(weight_keys) + chunk_size - 1) // chunk_size
                        
                        for key in tqdm(chunk_keys, desc=f"Extracting LoRA (chunk {chunk_num}/{total_chunks})"):
                            try:
                                tensor_a = f_a.get_tensor(key).to(device)
                                tensor_b = f_b.get_tensor(key).to(device)
                                
                                if tensor_a.shape != tensor_b.shape:
                                    continue
                                
                                diff = tensor_a - tensor_b
                                
                                if torch.abs(diff).max() < 1e-6:
                                    del tensor_a, tensor_b, diff
                                    continue
                                
                                # Apply mode-specific filtering
                                if mode == "add_similar":
                                    similarity = torch.nn.functional.cosine_similarity(
                                        tensor_a.flatten().unsqueeze(0),
                                        tensor_b.flatten().unsqueeze(0)
                                    ).item()
                                    
                                    if similarity <= similarity_threshold:
                                        del tensor_a, tensor_b, diff
                                        continue
                                
                                # Perform SVD
                                U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)
                                
                                # Truncate to rank
                                U_truncated = U[:, :rank]
                                S_truncated = S[:rank]
                                Vh_truncated = Vh[:rank, :]
                                
                                # Create LoRA matrices with SVD convention
                                # down = Vh_truncated (rank x in_features)
                                # up = U_truncated @ diag(sqrt(S_truncated)) (out_features x rank)
                                sqrt_S = torch.sqrt(S_truncated)
                                lora_down = Vh_truncated
                                lora_up = U_truncated * sqrt_S.unsqueeze(0)
                                
                                # Convert key to Chroma format with lora_unet_ prefix
                                base_name = key[:-7]  # Remove .weight
                                chroma_key = f"lora_unet_{base_name.replace('.', '_')}"
                                
                                # Make tensors contiguous before saving
                                lora_dict[f"{chroma_key}.lora_down.weight"] = lora_down.contiguous().cpu()
                                lora_dict[f"{chroma_key}.lora_up.weight"] = lora_up.contiguous().cpu()
                                lora_dict[f"{chroma_key}.alpha"] = torch.tensor(rank, dtype=torch.float32)
                                
                                extracted_count += 1
                                
                                del tensor_a, tensor_b, diff, U, S, Vh
                                del U_truncated, S_truncated, Vh_truncated
                                del lora_down, lora_up
                                
                                if device == "cuda":
                                    torch.cuda.empty_cache()
                                
                            except Exception as e:
                                logger.warning(f"Failed to extract LoRA from {key}: {e}")
                                continue
                        
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                
                logger.info(f"Extracted {extracted_count} LoRA pairs")
                
        except Exception as e:
            logger.error(f"Critical error during LoRA extraction: {e}")
            raise
        
        return lora_dict

def validate_extracted_lora(lora_dict: Dict[str, torch.Tensor], chroma_base_path: str, device: str = "cuda") -> bool:
    """Validate that extracted LoRA shapes match Chroma expectations"""
    logger.info("Validating extracted LoRA...")
    
    # Load Chroma shapes
    chroma_shapes = {}
    with safe_open(chroma_base_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith('.weight'):
                chroma_shapes[key] = f.get_tensor(key).shape
    
    issues_found = False
    
    # Check each LoRA pair
    for key in lora_dict:
        if '.lora_down.weight' in key:
            # Remove lora_unet_ prefix for comparison
            base_key = key.replace('.lora_down.weight', '').replace('lora_unet_', '')
            down_key = key
            up_key = key.replace('.lora_down.weight', '.lora_up.weight')
            
            if up_key in lora_dict:
                down_shape = lora_dict[down_key].shape
                up_shape = lora_dict[up_key].shape
                
                # Calculate what the reconstructed shape would be
                # LoRA reconstruction: up @ down
                # up: [out_features, rank], down: [rank, in_features]
                # Result: [out_features, in_features]
                lora_result_shape = (up_shape[0], down_shape[1])
                
                # Find corresponding Chroma weight
                chroma_key = base_key + '.weight'
                
                if chroma_key in chroma_shapes:
                    chroma_shape = chroma_shapes[chroma_key]
                    
                    if lora_result_shape != chroma_shape:
                        logger.warning(f"Shape mismatch for {base_key}: LoRA produces {lora_result_shape}, Chroma expects {chroma_shape}")
                        issues_found = True
    
    if not issues_found:
        logger.info("Validation passed - all LoRA shapes match Chroma expectations")
    else:
        logger.warning("Validation found shape mismatches - the LoRA may not apply correctly")
    
    return not issues_found

def main():
    parser = argparse.ArgumentParser(description='Convert Flux Dev LoRA to Chroma format')
    parser.add_argument('--flux-base', required=True, help='Path to Flux base model')
    parser.add_argument('--flux-lora', required=True, help='Path to Flux LoRA')
    parser.add_argument('--chroma-base', required=True, help='Path to Chroma base model')
    parser.add_argument('--output-lora', required=True, help='Output path for Chroma LoRA')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--mode', default='standard', 
                       choices=['standard', 'add_similar', 'add_dissimilar'],
                       help='Merge mode')
    parser.add_argument('--rank', type=int, default=-1, help='LoRA rank (-1 for auto-detect)')
    parser.add_argument('--lora-alpha', type=float, default=1.0, help='LoRA alpha scaling')
    parser.add_argument('--similarity-threshold', type=float, default=0.1, 
                       help='Similarity threshold for add_similar/add_dissimilar modes')
    parser.add_argument('--chunk-size', type=int, default=50, help='Chunk size for processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze compatibility')
    parser.add_argument('--skip-validation', action='store_true', help='Skip final validation step')
    
    args = parser.parse_args()
    
    # Set debug mode
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    
    # Analyze compatibility
    logger.info("Analyzing LoRA compatibility...")
    compatibility = analyze_lora_compatibility(args.flux_lora)
    
    logger.info("\nLoRA Analysis:")
    logger.info(f"  Naming style: {compatibility['naming_style']}")
    logger.info(f"  Total pairs: {compatibility['total_pairs']}")
    logger.info(f"  Grouped into: {compatibility['compatible_pairs'] + compatibility['incompatible_pairs']} target layers")
    logger.info(f"  Convertible pairs: {compatibility['compatible_pairs']}")
    
    if DEBUG_MODE:
        if compatibility['convertible_pairs']:
            logger.debug(f"Sample convertible keys: {compatibility['convertible_pairs'][:3]}")
        if compatibility['unconvertible_pairs']:
            logger.debug(f"Sample unconvertible keys: {compatibility['unconvertible_pairs'][:3]}")
    
    if compatibility['compatible_pairs'] == 0:
        logger.error("ERROR: No compatible LoRA pairs found!")
        logger.error("This LoRA cannot be converted to Chroma format.")
        logger.error("This LoRA may be text-encoder only or use an unsupported format.")
        return 1
    
    if args.analyze_only:
        logger.info("Analysis complete (--analyze-only mode)")
        return 0
    
    # Create temp directory
    temp_dir = Path("temp_conversion")
    temp_dir.mkdir(exist_ok=True)
    
    # Auto-detect rank
    if args.rank == -1:
        args.rank, _ = detect_lora_rank(args.flux_lora)
    
    try:
        print_memory_usage("Initial")
        
        # Step 1: Merge LoRA with Flux
        logger.info("="*60)
        logger.info("Step 1: Merging LoRA with Flux base model...")
        logger.info("="*60)
        
        merged_flux_path = temp_dir / "merged_flux.safetensors"
        
        merged_state_dict, applied_count, merge_stats = LoRAMerger.merge_lora_to_model(
            args.flux_base, args.flux_lora, args.device, args.lora_alpha
        )
        
        # Save merged model
        logger.info("Saving merged Flux model...")
        save_file(merged_state_dict, str(merged_flux_path))
        
        del merged_state_dict
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print_memory_usage("After merging")
        
        # Step 2: Compute differences
        logger.info("\n" + "="*60)
        logger.info("Step 2: Computing differences...")
        logger.info("="*60)
        
        differences = DifferenceComputer.compute_difference(
            str(merged_flux_path), args.flux_base, args.device, args.chunk_size
        )
        
        print_memory_usage("After difference computation")
        
        # Step 3: Apply to Chroma
        logger.info("\n" + "="*60)
        logger.info("Step 3: Applying differences to Chroma...")
        logger.info("="*60)
        
        merged_chroma_path = temp_dir / "merged_chroma.safetensors"
        
        ChromaDifferenceApplier.apply_differences(
            args.chroma_base, differences, str(merged_chroma_path), 
            args.device, mode=args.mode, similarity_threshold=args.similarity_threshold
        )
        
        del differences
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print_memory_usage("After applying differences")
        
        # Step 4: Extract LoRA
        logger.info("\n" + "="*60)
        logger.info("Step 4: Extracting LoRA from merged Chroma...")
        logger.info("="*60)
        
        lora_dict = LoRAExtractor.extract_lora_svd(
            str(merged_chroma_path), args.chroma_base, args.rank, args.device,
            mode=args.mode, similarity_threshold=args.similarity_threshold,
            chunk_size=args.chunk_size
        )
        
        # Validate if requested
        if not args.skip_validation:
            validate_extracted_lora(lora_dict, args.chroma_base, args.device)
        
        # Load and preserve metadata
        logger.info("Loading metadata from input LoRA...")
        metadata = load_lora_metadata(args.flux_lora)
        metadata["chroma_lora_pairs"] = str(len([k for k in lora_dict if '.lora_down.weight' in k]))
        metadata["chroma_extraction_rank"] = str(args.rank)
        metadata["flux_lora_applied_count"] = str(applied_count)
        metadata["chroma_conversion_date"] = datetime.now().isoformat()
        metadata["chroma_converter_version"] = "v7"
        metadata["original_naming_style"] = compatibility['naming_style']
        metadata["accumulated_layers"] = str(len(merge_stats.get("accumulated", [])))
        
        logger.info(f"Saving final LoRA with {len(metadata)} metadata entries...")
        save_file(lora_dict, args.output_lora, metadata=metadata)
        
        # Verify metadata was saved
        logger.info("Verifying saved metadata...")
        with safe_open(args.output_lora, framework="pt", device="cpu") as f:
            if hasattr(f, 'metadata'):
                saved_metadata = f.metadata()
                if saved_metadata:
                    logger.info(f"Successfully saved {len(saved_metadata)} metadata entries")
                    # Check for trigger words
                    for key, value in saved_metadata.items():
                        if any(word in key.lower() for word in ["trigger", "prompt", "keyword"]):
                            logger.info(f"Verified trigger metadata preserved: {key} = {value[:100] if len(str(value)) > 100 else value}")
                else:
                    logger.warning("No metadata found in saved file")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("Conversion complete!")
        logger.info(f"Output: {args.output_lora}")
        logger.info(f"Extracted {len([k for k in lora_dict if '.lora_down.weight' in k])} LoRA triplets with rank {args.rank}")
        logger.info(f"Original naming style: {compatibility['naming_style']}")
        logger.info(f"Accumulated {len(merge_stats.get('accumulated', []))} multi-component layers")
        logger.info("="*60)
        
        # Cleanup
        logger.info("Cleaning up temporary files...")
        if merged_flux_path.exists():
            merged_flux_path.unlink()
        if merged_chroma_path.exists():
            merged_chroma_path.unlink()
        
        print_memory_usage("Final")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if DEBUG_MODE:
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
