#!/usr/bin/env python3
"""
Flux Dev to Chroma LoRA Converter - Fixed Version
Fixes the critical shape transpose bug and improves compatibility
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

# Layers to skip entirely
SKIP_LAYERS = [
    "lora_te",              # Text encoder layers
    "lora_te1",             # Text encoder layers
    "lora_te2",             # Text encoder layers  
    "text_encoder",         # Text encoder layers
    "text_model",           # Text encoder layers
    "add_k_proj",          # Flux 1.1/Pro only
    "add_q_proj",          # Flux 1.1/Pro only
    "add_v_proj",          # Flux 1.1/Pro only
    "to_add_out",          # Flux 1.1/Pro only
    "single_transformer_blocks",  # Incorrect naming
]

def setup_exception_handler():
    """Set up global exception handler for better error reporting"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error("\n" + "="*60)
        logger.error("UNEXPECTED ERROR - Script crashed!")
        logger.error("="*60)
        
        # Check for common issues
        if "out of memory" in str(exc_value).lower() or exc_type.__name__ == "OutOfMemoryError":
            logger.error("Out of Memory Error Detected!")
            logger.error("\nSolutions:")
            logger.error("1. Use CPU processing: --device cpu")
            logger.error("2. Reduce chunk size: --chunk-size 5")
            logger.error("3. Close other GPU applications")
            logger.error("4. Process one model at a time")
            if torch.cuda.is_available():
                logger.error(f"\nCurrent GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB used")
        
        logger.error(f"\nError Type: {exc_type.__name__}")
        logger.error(f"Error Message: {exc_value}")
        
        if DEBUG_MODE:
            logger.error("\nFull Traceback:")
            logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        else:
            logger.error("\nRun with --debug flag for full traceback")
        
        logger.error("\nIf this persists, please report this error with the full output")
        sys.exit(1)
    
    sys.excepthook = handle_exception

def print_memory_usage(stage: str = ""):
    """Print current memory usage"""
    if not HAS_DEBUG_LIBS or not DEBUG_MODE:
        return
        
    try:
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
        print(f"\n[Memory - {stage}] RAM: {ram_mb:.1f}MB", end="")
        
        if torch.cuda.is_available():
            for i, gpu in enumerate(GPUtil.getGPUs()):
                print(f" | GPU{i}: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB", end="")
        print()
    except:
        pass

def detect_lora_rank(lora_path: str) -> Tuple[int, Dict[str, int]]:
    """Auto-detect LoRA rank from the model and return rank distribution"""
    ranks = []
    rank_by_layer = {}
    
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "lora_down" in key or "lora_A" in key:
                    tensor = f.get_tensor(key)
                    rank = min(tensor.shape)
                    ranks.append(rank)
                    rank_by_layer[key] = rank
                    
        if ranks:
            # Use the most common rank
            rank_counts = Counter(ranks)
            detected_rank = rank_counts.most_common(1)[0][0]
            logger.info(f"Auto-detected LoRA rank: {detected_rank}")
            logger.info(f"Rank distribution: {dict(rank_counts)}")
            return detected_rank, rank_by_layer
    except Exception as e:
        logger.warning(f"Could not auto-detect rank: {e}")
    
    logger.info("Using default rank: 16")
    return 16, {}

def fix_lora_key_for_matching(lora_key: str) -> str:
    """
    Fix LoRA key format to match Flux structure
    This is for the intermediate processing - final output will use Chroma format
    """
    original_key = lora_key
    
    # Remove common prefixes
    prefixes_to_remove = [
        "lora_unet_",
        "lora.",
        "transformer.",
        "model.",
        "diffusion_model.",
        "unet.",
    ]
    
    for prefix in prefixes_to_remove:
        if lora_key.startswith(prefix):
            lora_key = lora_key[len(prefix):]
            if DEBUG_MODE:
                logger.debug(f"Removed prefix '{prefix}' from {original_key}")
    
    # Remove LoRA-specific suffixes to get base key
    suffixes_to_remove = [
        ".lora_A.weight", ".lora_B.weight",
        ".lora_up.weight", ".lora_down.weight",
        ".lora.A.weight", ".lora.B.weight",
        ".lora.up.weight", ".lora.down.weight",
        ".alpha", ".weight", ".bias"
    ]
    
    for suffix in suffixes_to_remove:
        if lora_key.endswith(suffix):
            lora_key = lora_key[:-len(suffix)]
            break
    
    # Transform the key structure properly
    if "double_blocks_" in lora_key or "single_blocks_" in lora_key:
        parts = lora_key.split("_")
        
        if len(parts) >= 3:
            if parts[0] == "double" and parts[1] == "blocks":
                block_num = parts[2]
                remaining = "_".join(parts[3:])
                component = remaining.replace("_attn_", "_attn.").replace("_mlp_", "_mlp.").replace("_mod_", "_mod.")
                lora_key = f"double_blocks.{block_num}.{component}"
                
            elif parts[0] == "single" and parts[1] == "blocks":
                block_num = parts[2]
                remaining = "_".join(parts[3:])
                component = remaining.replace("_", ".")
                lora_key = f"single_blocks.{block_num}.{component}"
    
    elif "double.blocks" in lora_key or "single.blocks" in lora_key:
        lora_key = lora_key.replace("double.blocks", "double_blocks")
        lora_key = lora_key.replace("single.blocks", "single_blocks")
        lora_key = lora_key.replace(".img.attn", ".img_attn")
        lora_key = lora_key.replace(".txt.attn", ".txt_attn")
        lora_key = lora_key.replace(".img.mlp", ".img_mlp")
        lora_key = lora_key.replace(".txt.mlp", ".txt_mlp")
        lora_key = lora_key.replace(".img.mod", ".img_mod")
        lora_key = lora_key.replace(".txt.mod", ".txt_mod")
    
    # Fix other naming conventions
    lora_key = lora_key.replace("single_transformer_blocks", "single_blocks")
    lora_key = lora_key.replace("transformer_blocks", "double_blocks")
    
    if DEBUG_MODE and original_key != lora_key:
        logger.debug(f"Key transformation: {original_key} -> {lora_key}")
    
    return lora_key

def convert_flux_key_to_chroma_format(flux_key: str) -> str:
    """
    Convert Flux key format to Chroma LoRA format for final output
    Example: double_blocks.0.img_attn.proj -> lora_unet_double_blocks_0_img_attn_proj
    """
    # Remove .weight suffix if present
    if flux_key.endswith('.weight'):
        flux_key = flux_key[:-7]
    
    # Convert dots to underscores
    chroma_key = flux_key.replace('.', '_')
    
    # Add lora_unet_ prefix
    if not chroma_key.startswith('lora_unet_'):
        chroma_key = f'lora_unet_{chroma_key}'
    
    return chroma_key

def extract_lora_pairs(lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, str]]:
    """Extract LoRA pairs from state dict with proper key fixing"""
    lora_pairs = {}
    alphas = {}  # Store alpha values separately
    
    for key in lora_state_dict.keys():
        # Skip alpha values for now, store them separately
        if key.endswith('.alpha'):
            base_key = fix_lora_key_for_matching(key)
            alphas[base_key] = key
            continue
        
        # Get the base key for matching
        base_key = fix_lora_key_for_matching(key)
        
        # Skip text encoder and incompatible layers
        if any(skip_pattern in key for skip_pattern in SKIP_LAYERS):
            if DEBUG_MODE:
                logger.debug(f"Skipping layer: {key}")
            continue
        
        if "lora_A" in key:
            lora_pairs[base_key] = lora_pairs.get(base_key, {})
            lora_pairs[base_key]["A"] = key
        elif "lora_B" in key:
            lora_pairs[base_key] = lora_pairs.get(base_key, {})
            lora_pairs[base_key]["B"] = key
        elif "lora_up" in key:
            lora_pairs[base_key] = lora_pairs.get(base_key, {})
            lora_pairs[base_key]["up"] = key
        elif "lora_down" in key:
            lora_pairs[base_key] = lora_pairs.get(base_key, {})
            lora_pairs[base_key]["down"] = key
    
    # Add alpha values to pairs if they exist
    for base_key, alpha_key in alphas.items():
        if base_key in lora_pairs:
            lora_pairs[base_key]["alpha"] = alpha_key
    
    return lora_pairs

class LoRAMerger:
    """Handles LoRA merging operations with fixed key mapping"""
    
    @staticmethod
    def merge_lora_to_model(base_model_path: str, lora_path: str, device: str = "cuda", 
                          lora_alpha: float = 1.0) -> Tuple[Dict[str, torch.Tensor], int]:
        """Merge LoRA weights into base model and return merged state dict and applied count"""
        logger.info("Merging LoRA into base model...")
        merged_state_dict = {}
        
        # Load LoRA state dict
        with safe_open(lora_path, framework="pt", device="cpu") as lora_f:
            lora_state_dict = {k: lora_f.get_tensor(k) for k in lora_f.keys()}
        
        # Extract LoRA pairs with fixed keys
        lora_pairs = extract_lora_pairs(lora_state_dict)
        
        logger.info(f"Found {len(lora_pairs)} LoRA layer pairs")
        
        # Process base model
        with safe_open(base_model_path, framework="pt", device="cpu") as base_f:
            all_keys = list(base_f.keys())
            
            # Verify key matching if in debug mode
            if DEBUG_MODE:
                matched_pairs = sum(1 for base_key in lora_pairs if base_key + ".weight" in all_keys or base_key in all_keys)
                logger.debug(f"Matching report: {matched_pairs}/{len(lora_pairs)} pairs matched")
            
            # First pass: copy all base weights
            for key in tqdm(all_keys, desc="Loading base model"):
                merged_state_dict[key] = base_f.get_tensor(key)
            
            # Second pass: apply LoRA modifications
            applied_count = 0
            
            for base_key, pair_info in tqdm(lora_pairs.items(), desc="Applying LoRA"):
                # Try to find matching key in base model
                matching_key = None
                
                # Try exact match with .weight suffix
                if base_key + ".weight" in merged_state_dict:
                    matching_key = base_key + ".weight"
                # Try exact match
                elif base_key in merged_state_dict:
                    matching_key = base_key
                
                if matching_key:
                    try:
                        base_weight = merged_state_dict[matching_key].to(device).float()
                        
                        # Get LoRA matrices based on naming convention
                        if "A" in pair_info and "B" in pair_info:
                            # lora_A/lora_B format
                            lora_down = lora_state_dict[pair_info["A"]].to(device).float()
                            lora_up = lora_state_dict[pair_info["B"]].to(device).float()
                        elif "down" in pair_info and "up" in pair_info:
                            # lora_down/lora_up format
                            lora_down = lora_state_dict[pair_info["down"]].to(device).float()
                            lora_up = lora_state_dict[pair_info["up"]].to(device).float()
                        else:
                            logger.warning(f"Incomplete LoRA pair for {base_key}")
                            continue
                        
                        # Get alpha value if available
                        alpha = lora_alpha
                        if "alpha" in pair_info:
                            alpha_tensor = lora_state_dict[pair_info["alpha"]]
                            alpha_value = alpha_tensor.item() if alpha_tensor.numel() == 1 else alpha_tensor[0].item()
                            rank = lora_down.shape[0] if lora_down.shape[0] < lora_down.shape[1] else lora_down.shape[1]
                            alpha = alpha_value / rank * lora_alpha
                        
                        # Apply LoRA: W' = W + alpha * BA
                        lora_weight = alpha * (lora_up @ lora_down)
                        
                        # Handle shape differences
                        if lora_weight.shape != base_weight.shape:
                            # Try to reshape if possible
                            if np.prod(lora_weight.shape) == np.prod(base_weight.shape):
                                lora_weight = lora_weight.reshape(base_weight.shape)
                            else:
                                logger.warning(f"Shape mismatch for {matching_key}: base={base_weight.shape}, lora={lora_weight.shape}")
                                continue
                        
                        # Apply the LoRA
                        merged_weight = base_weight + lora_weight
                        
                        # Store in the same dtype as original
                        merged_state_dict[matching_key] = merged_weight.to(merged_state_dict[matching_key].dtype).cpu()
                        applied_count += 1
                        
                        # Cleanup
                        del base_weight, lora_weight, lora_down, lora_up
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply LoRA to {matching_key}: {e}")
                        if DEBUG_MODE:
                            logger.debug(traceback.format_exc())
                        continue
                else:
                    if DEBUG_MODE:
                        logger.debug(f"No matching key found for LoRA layer: {base_key}")
            
            logger.info(f"Applied LoRA to {applied_count} layers out of {len(lora_pairs)}")
            
            if applied_count == 0:
                logger.error("ERROR: No LoRA weights were applied!")
                raise ValueError("LoRA application failed - no weights were applied")
        
        return merged_state_dict, applied_count

class DifferenceComputer:
    """Computes differences between models with improved error handling"""
    
    @staticmethod
    def compute_difference(model_a_path: str, model_b_path: str, 
                          device: str = "cuda", chunk_size: int = 50) -> Dict[str, torch.Tensor]:
        """Compute difference between two models (A - B)"""
        logger.info("Computing model differences...")
        differences = {}
        processed_count = 0
        skipped_count = 0
        
        try:
            # First, check if files exist and are readable
            if not Path(model_a_path).exists():
                raise FileNotFoundError(f"Model A not found: {model_a_path}")
            if not Path(model_b_path).exists():
                raise FileNotFoundError(f"Model B not found: {model_b_path}")
            
            with safe_open(model_a_path, framework="pt", device="cpu") as f_a:
                with safe_open(model_b_path, framework="pt", device="cpu") as f_b:
                    keys_a = set(f_a.keys())
                    keys_b = set(f_b.keys())
                    common_keys = sorted(list(keys_a.intersection(keys_b)))
                    
                    logger.info(f"Computing differences for {len(common_keys)} common layers")
                    
                    if len(common_keys) == 0:
                        raise ValueError("No common keys found between models!")
                    
                    # Process in chunks to save memory
                    for i in range(0, len(common_keys), chunk_size):
                        chunk_keys = common_keys[i:i+chunk_size]
                        chunk_num = i // chunk_size + 1
                        total_chunks = (len(common_keys) + chunk_size - 1) // chunk_size
                        
                        for key in tqdm(chunk_keys, desc=f"Computing differences (chunk {chunk_num}/{total_chunks})"):
                            try:
                                # Load tensors one at a time to save memory
                                tensor_a = f_a.get_tensor(key)
                                tensor_b = f_b.get_tensor(key)
                                
                                # Move to device and compute
                                tensor_a = tensor_a.to(device).float()
                                tensor_b = tensor_b.to(device).float()
                                
                                if tensor_a.shape != tensor_b.shape:
                                    logger.warning(f"Shape mismatch for {key}: A={tensor_a.shape}, B={tensor_b.shape}")
                                    skipped_count += 1
                                    continue
                                
                                # Compute difference
                                diff = tensor_a - tensor_b
                                
                                # Only store significant differences
                                if torch.abs(diff).max() > 1e-8:
                                    differences[key] = diff.cpu()
                                    processed_count += 1
                                else:
                                    skipped_count += 1
                                
                                # Immediate cleanup
                                del tensor_a, tensor_b, diff
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    logger.error(f"Out of memory processing {key}")
                                    # Try to recover
                                    if device == "cuda":
                                        torch.cuda.empty_cache()
                                    # Skip this layer
                                    skipped_count += 1
                                    continue
                                else:
                                    raise
                            except Exception as e:
                                logger.warning(f"Failed to compute difference for {key}: {e}")
                                skipped_count += 1
                                continue
                        
                        # Cleanup after each chunk
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        
                        # Print progress
                        if DEBUG_MODE:
                            print_memory_usage(f"After chunk {chunk_num}/{total_chunks}")
                    
                    logger.info(f"Computed {processed_count} differences, skipped {skipped_count} unchanged layers")
                    
                    if processed_count == 0:
                        logger.error("No differences found between models!")
                        raise ValueError("No differences computed - models might be identical or LoRA not applied")
            
        except Exception as e:
            logger.error(f"Critical error during difference computation: {type(e).__name__}: {e}")
            if DEBUG_MODE:
                logger.debug(traceback.format_exc())
            raise
        
        return differences

def should_skip_layer(key: str) -> bool:
    """Check if a layer should be skipped"""
    for skip_pattern in SKIP_LAYERS:
        if skip_pattern in key:
            return True
    return False

class ChromaDifferenceApplier:
    """Applies differences to Chroma model with memory optimization"""
    
    @staticmethod
    def apply_differences(base_model_path: str, differences: Dict[str, torch.Tensor],
                         output_path: str, device: str = "cuda", 
                         mode: str = "standard", similarity_threshold: float = 0.1) -> int:
        """Apply differences to Chroma base model"""
        logger.info(f"Applying differences to Chroma model using {mode} mode...")
        
        applied_count = 0
        skipped_count = 0
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process all keys and apply differences
        temp_state_dict = {}
        
        try:
            with safe_open(base_model_path, framework="pt", device="cpu") as base_f:
                all_keys = list(base_f.keys())
                
                # Process in chunks
                chunk_size = 100
                for i in range(0, len(all_keys), chunk_size):
                    chunk_keys = all_keys[i:i+chunk_size]
                    
                    for key in tqdm(chunk_keys, desc=f"Applying differences (chunk {i//chunk_size + 1})"):
                        # Load base tensor
                        tensor = base_f.get_tensor(key)
                        
                        # Check if we have a difference for this layer
                        if key in differences:
                            try:
                                diff = differences[key].to(device)
                                base_tensor = tensor.to(device).float()
                                
                                if diff.shape != base_tensor.shape:
                                    logger.warning(f"Shape mismatch for {key}")
                                    temp_state_dict[key] = tensor
                                    continue
                                
                                # Apply based on mode
                                if mode == "add_dissimilar":
                                    # Only apply if difference is significant
                                    relative_diff = torch.norm(diff) / (torch.norm(base_tensor) + 1e-8)
                                    if relative_diff < similarity_threshold:
                                        temp_state_dict[key] = tensor
                                        skipped_count += 1
                                        continue
                                
                                elif mode == "comparative":
                                    # Scale based on similarity
                                    similarity = torch.nn.functional.cosine_similarity(
                                        base_tensor.flatten(), 
                                        (base_tensor + diff).flatten(), 
                                        dim=0
                                    )
                                    scale = 1.0 - similarity.abs().item()
                                    diff = diff * scale
                                
                                # Apply difference
                                modified = base_tensor + diff
                                
                                # Ensure contiguous before saving
                                modified = modified.contiguous()
                                
                                temp_state_dict[key] = modified.to(tensor.dtype).cpu()
                                applied_count += 1
                                
                                del diff, base_tensor, modified
                                
                            except Exception as e:
                                logger.warning(f"Failed to apply difference to {key}: {e}")
                                temp_state_dict[key] = tensor
                                skipped_count += 1
                        else:
                            # Keep original
                            temp_state_dict[key] = tensor
                    
                    # Periodic cleanup
                    if device == "cuda":
                        torch.cuda.empty_cache()
            
            # Ensure all tensors are contiguous before saving
            logger.info("Ensuring all tensors are contiguous...")
            for key in temp_state_dict:
                if not temp_state_dict[key].is_contiguous():
                    temp_state_dict[key] = temp_state_dict[key].contiguous()
            
            # Save the modified model
            logger.info(f"Saving modified model to {output_path}")
            save_file(temp_state_dict, str(output_path))
            
            logger.info(f"Applied differences to {applied_count} layers")
            if mode != "standard":
                logger.info(f"Skipped {skipped_count} layers based on {mode} mode")
            
            if applied_count == 0:
                logger.error("ERROR: No differences were applied to Chroma!")
                raise ValueError("Difference application failed")
            
        except Exception as e:
            logger.error(f"Error applying differences: {e}")
            raise
        
        return applied_count

class LoRAExtractor:
    """Extracts LoRA from model differences using SVD with FIXED shape orientation"""
    
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
                                 and not should_skip_layer(k)]
                    
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
                                
                                # Compute difference
                                diff = tensor_a - tensor_b
                                
                                # Skip if difference is too small
                                if torch.abs(diff).max() < 1e-6:
                                    del tensor_a, tensor_b, diff
                                    continue
                                
                                # Apply mode-specific filtering
                                if mode == "add_dissimilar":
                                    relative_diff = torch.norm(diff) / (torch.norm(tensor_b) + 1e-8)
                                    if relative_diff < similarity_threshold:
                                        del tensor_a, tensor_b, diff
                                        continue
                                
                                # Convert to float32 for SVD
                                if diff.dtype == torch.bfloat16:
                                    diff = diff.float()
                                
                                # Reshape for SVD if needed
                                original_shape = diff.shape
                                if len(original_shape) > 2:
                                    diff = diff.reshape(original_shape[0], -1)
                                
                                # Perform SVD
                                try:
                                    U, S, Vt = torch.linalg.svd(diff, full_matrices=False)
                                except RuntimeError as e:
                                    if "svd_cuda" in str(e):
                                        logger.warning(f"CUDA SVD failed for {key}, using CPU")
                                        diff_cpu = diff.cpu()
                                        U, S, Vt = torch.linalg.svd(diff_cpu, full_matrices=False)
                                        U = U.to(device)
                                        S = S.to(device)
                                        Vt = Vt.to(device)
                                    else:
                                        raise
                                
                                # Truncate to rank
                                actual_rank = min(rank, len(S), U.shape[1], Vt.shape[0])
                                U_r = U[:, :actual_rank]
                                S_r = S[:actual_rank]
                                Vt_r = Vt[:actual_rank, :]
                                
                                # Create LoRA matrices with CORRECT shape orientation for Chroma
                                sqrt_S = torch.sqrt(S_r)
                                
                                # CRITICAL FIX: Chroma expects lora_down=[rank, in_features], lora_up=[out_features, rank]
                                # For a weight of shape (out_features, in_features):
                                # - U has shape (out_features, rank) after truncation
                                # - Vt has shape (rank, in_features) after truncation
                                
                                # lora_down should be (rank, in_features) - use Vt_r directly
                                lora_down = (Vt_r.T * sqrt_S.unsqueeze(0)).T  # Shape: (rank, in_features)
                                
                                # lora_up should be (out_features, rank) - use U_r directly
                                lora_up = U_r * sqrt_S.unsqueeze(0)  # Shape: (out_features, rank)
                                
                                # Verify shapes
                                if DEBUG_MODE:
                                    logger.debug(f"Layer {key}: original weight shape {original_shape}")
                                    logger.debug(f"  lora_down shape: {lora_down.shape} (should be [{actual_rank}, {original_shape[1]}])")
                                    logger.debug(f"  lora_up shape: {lora_up.shape} (should be [{original_shape[0]}, {actual_rank}])")
                                    
                                    # Test multiplication
                                    test_result = lora_up @ lora_down
                                    logger.debug(f"  Test multiplication shape: {test_result.shape} (should match {original_shape})")
                                
                                # Ensure tensors are contiguous
                                lora_down = lora_down.contiguous()
                                lora_up = lora_up.contiguous()
                                
                                # Convert key to Chroma format
                                base_name = key.replace('.weight', '')
                                chroma_base = convert_flux_key_to_chroma_format(base_name)
                                
                                # Store with Chroma naming convention
                                lora_dict[f"{chroma_base}.lora_down.weight"] = lora_down.cpu()
                                lora_dict[f"{chroma_base}.lora_up.weight"] = lora_up.cpu()
                                
                                # Add alpha value (use rank as alpha, which is common practice)
                                alpha_tensor = torch.tensor(float(actual_rank), dtype=torch.bfloat16)
                                lora_dict[f"{chroma_base}.alpha"] = alpha_tensor
                                
                                extracted_count += 1
                                
                                del tensor_a, tensor_b, diff, U, S, Vt
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                logger.warning(f"Failed to extract LoRA for {key}: {e}")
                                if DEBUG_MODE:
                                    logger.debug(traceback.format_exc())
                                continue
                        
                        # Cleanup after each chunk
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        
                        if DEBUG_MODE:
                            print_memory_usage(f"After extraction chunk {chunk_num}/{total_chunks}")
            
            logger.info(f"Extracted {extracted_count} LoRA pairs")
            
            # Add modulation layers if they're missing (experimental)
            if extracted_count < 231:  # Working Chroma LoRA has 231 layers
                logger.info(f"Note: Extracted {extracted_count} layers, working Chroma LoRAs typically have 231")
                logger.info("Missing layers might be modulation layers that Chroma doesn't use")
            
        except Exception as e:
            logger.error(f"Error during LoRA extraction: {e}")
            raise
        
        return lora_dict

def load_lora_metadata(lora_path: str) -> Dict[str, Any]:
    """Load metadata from input LoRA including trigger words"""
    metadata = {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            if hasattr(f, 'metadata'):
                metadata = f.metadata()
                if metadata is None:
                    metadata = {}
    except Exception as e:
        logger.warning(f"Could not load LoRA metadata: {e}")
    
    # Extract important metadata to preserve
    preserved_metadata = {}
    
    # Common metadata keys for trigger words
    trigger_keys = [
        "ss_tag_frequency",
        "ss_dataset_dirs", 
        "trigger_words",
        "trigger_word",
        "instance_prompt",
        "ss_instance_prompt",
        "ss_caption_tag",
        "keywords"
    ]
    
    # Preserve trigger word metadata
    for key in trigger_keys:
        if key in metadata:
            preserved_metadata[key] = metadata[key]
            value_str = str(metadata[key])
            if len(value_str) > 100:
                logger.info(f"Found trigger metadata: {key} = {value_str[:100]}...")
            else:
                logger.info(f"Found trigger metadata: {key} = {value_str}")
    
    # Also preserve training info
    training_keys = [
        "ss_base_model_version",
        "ss_training_comment",
        "ss_network_module",
        "ss_network_args",
        "ss_learning_rate",
        "ss_unet_lr",
        "ss_text_encoder_lr",
        "ss_network_dim",
        "ss_network_alpha"
    ]
    
    for key in training_keys:
        if key in metadata:
            preserved_metadata[key] = metadata[key]
    
    # Add conversion info
    preserved_metadata["chroma_converted"] = "true"
    preserved_metadata["chroma_conversion_date"] = datetime.now().isoformat()
    preserved_metadata["chroma_conversion_tool"] = "flux-chroma-lora-converter-fixed"
    preserved_metadata["original_lora"] = str(Path(lora_path).name)
    
    return preserved_metadata

def verify_output_lora(lora_path: str) -> Dict[str, Any]:
    """Verify the structure of the output LoRA"""
    verification = {
        "total_keys": 0,
        "lora_pairs": 0,
        "alpha_keys": 0,
        "sample_keys": [],
        "key_patterns": defaultdict(int),
        "ranks": [],
        "shape_test": []
    }
    
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            verification["total_keys"] = len(keys)
            verification["sample_keys"] = keys[:10]
            
            # Test a few LoRA pairs
            tested = 0
            for key in keys:
                if ".lora_down.weight" in key and tested < 3:
                    base_key = key.replace(".lora_down.weight", "")
                    up_key = f"{base_key}.lora_up.weight"
                    
                    if up_key in keys:
                        down_tensor = f.get_tensor(key)
                        up_tensor = f.get_tensor(up_key)
                        
                        verification["shape_test"].append({
                            "layer": base_key,
                            "down_shape": list(down_tensor.shape),
                            "up_shape": list(up_tensor.shape),
                            "multiplication_result": list((up_tensor @ down_tensor).shape)
                        })
                        tested += 1
                
                if ".lora_down.weight" in key:
                    verification["lora_pairs"] += 1
                    tensor = f.get_tensor(key)
                    verification["ranks"].append(tensor.shape[0])  # First dimension is rank
                elif ".alpha" in key:
                    verification["alpha_keys"] += 1
                
                # Extract key pattern
                base_key = key.split(".")[0] + "." + key.split(".")[1] if len(key.split(".")) > 1 else key.split(".")[0]
                verification["key_patterns"][base_key] += 1
        
        # Get unique ranks
        if verification["ranks"]:
            verification["unique_ranks"] = list(set(verification["ranks"]))
            verification["most_common_rank"] = max(set(verification["ranks"]), key=verification["ranks"].count)
    
    except Exception as e:
        verification["error"] = str(e)
    
    return verification

def main():
    # Set up exception handler
    setup_exception_handler()
    
    parser = argparse.ArgumentParser(description="Convert Flux Dev LoRA to Chroma LoRA")
    parser.add_argument("--flux-base", type=str, required=True, help="Path to Flux Dev base model")
    parser.add_argument("--flux-lora", type=str, required=True, help="Path to Flux Dev LoRA")
    parser.add_argument("--chroma-base", type=str, required=True, help="Path to Chroma base model")
    parser.add_argument("--output-lora", type=str, required=True, help="Output path for Chroma LoRA")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                       help="Device to use for computation")
    parser.add_argument("--rank", type=int, default=-1, 
                       help="LoRA rank for extraction (-1 for auto-detect)")
    parser.add_argument("--lora-alpha", type=float, default=1.0, 
                       help="LoRA alpha (weight) for merging")
    parser.add_argument("--chunk-size", type=int, default=50, 
                       help="Chunk size for memory-efficient processing")
    parser.add_argument("--mode", type=str, default="standard", 
                       choices=["standard", "add_dissimilar", "comparative"],
                       help="Conversion mode")
    parser.add_argument("--similarity-threshold", type=float, default=0.1,
                       help="Threshold for add_dissimilar mode (0.0-1.0)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify LoRA compatibility without conversion")
    parser.add_argument("--inspect-output", type=str, 
                       help="Inspect an existing converted LoRA file")
    
    args = parser.parse_args()
    
    # Set debug mode
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    
    # If inspect-output mode
    if args.inspect_output:
        logger.info(f"Inspecting output LoRA: {args.inspect_output}")
        verification = verify_output_lora(args.inspect_output)
        
        logger.info(f"\nOutput LoRA Structure:")
        logger.info(f"Total keys: {verification['total_keys']}")
        logger.info(f"LoRA pairs: {verification['lora_pairs']}")
        logger.info(f"Alpha keys: {verification['alpha_keys']}")
        logger.info(f"Ranks: {verification.get('unique_ranks', 'N/A')}")
        logger.info(f"Most common rank: {verification.get('most_common_rank', 'N/A')}")
        
        logger.info(f"\nKey patterns:")
        for pattern, count in sorted(verification['key_patterns'].items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {pattern}: {count}")
        
        logger.info(f"\nShape tests:")
        for test in verification.get('shape_test', []):
            logger.info(f"  Layer: {test['layer']}")
            logger.info(f"    Down shape: {test['down_shape']}")
            logger.info(f"    Up shape: {test['up_shape']}")
            logger.info(f"    Result shape: {test['multiplication_result']}")
        
        # Load and display metadata
        metadata = load_lora_metadata(args.inspect_output)
        if metadata:
            logger.info(f"\nMetadata:")
            for k, v in metadata.items():
                if isinstance(v, str) and len(v) > 100:
                    logger.info(f"  {k}: {v[:100]}...")
                else:
                    logger.info(f"  {k}: {v}")
        
        return 0
    
    # Create temp directory
    temp_dir = Path("temp_conversion")
    temp_dir.mkdir(exist_ok=True)
    
    # Auto-detect rank if needed
    if args.rank == -1:
        args.rank, rank_by_layer = detect_lora_rank(args.flux_lora)
    else:
        rank_by_layer = {}
    
    # If verify-only mode
    if args.verify_only:
        logger.info("Running in verify-only mode...")
        with safe_open(args.flux_lora, framework="pt", device="cpu") as lora_f:
            lora_state_dict = {k: lora_f.get_tensor(k) for k in lora_f.keys()}
        
        lora_pairs = extract_lora_pairs(lora_state_dict)
        
        with safe_open(args.flux_base, framework="pt", device="cpu") as flux_f:
            flux_keys = set(flux_f.keys())
        
        matched = sum(1 for base_key in lora_pairs if base_key + ".weight" in flux_keys or base_key in flux_keys)
        
        logger.info(f"\nVerification Report:")
        logger.info(f"Total LoRA pairs: {len(lora_pairs)}")
        logger.info(f"Matched pairs: {matched}")
        logger.info(f"Unmatched pairs: {len(lora_pairs) - matched}")
        logger.info(f"Match rate: {matched/len(lora_pairs)*100:.1f}%")
        
        return 0
    
    try:
        print_memory_usage("Initial")
        
        # Step 1: Merge LoRA with Flux base model
        logger.info("="*60)
        logger.info("Step 1: Merging LoRA with Flux base model...")
        logger.info("="*60)
        
        merged_flux_path = temp_dir / "merged_flux.safetensors"
        
        try:
            merged_state_dict, applied_count = LoRAMerger.merge_lora_to_model(
                args.flux_base, args.flux_lora, args.device, args.lora_alpha
            )
        except Exception as e:
            logger.error(f"Failed to merge LoRA with base model: {type(e).__name__}: {e}")
            return 1
        
        # Save merged model
        logger.info("Saving merged Flux model...")
        try:
            save_file(merged_state_dict, str(merged_flux_path))
        except Exception as e:
            logger.error(f"Failed to save merged model: {type(e).__name__}: {e}")
            return 1
        
        del merged_state_dict
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print_memory_usage("After merging")
        
        # Step 2: Compute differences
        logger.info("\n" + "="*60)
        logger.info("Step 2: Computing differences between merged and original Flux...")
        logger.info("="*60)
        try:
            differences = DifferenceComputer.compute_difference(
                str(merged_flux_path), args.flux_base, args.device, args.chunk_size
            )
        except Exception as e:
            logger.error(f"Failed to compute differences: {type(e).__name__}: {e}")
            return 1
        
        if not differences:
            logger.error("No differences found between merged and original model.")
            return 1
        
        print_memory_usage("After difference computation")
        
        # Step 3: Apply differences to Chroma
        logger.info("\n" + "="*60)
        logger.info("Step 3: Applying differences to Chroma base model...")
        logger.info("="*60)
        merged_chroma_path = temp_dir / "merged_chroma.safetensors"
        
        try:
            ChromaDifferenceApplier.apply_differences(
                args.chroma_base, differences, str(merged_chroma_path), 
                args.device, mode=args.mode, similarity_threshold=args.similarity_threshold
            )
        except Exception as e:
            logger.error(f"Failed to apply differences to Chroma: {type(e).__name__}: {e}")
            return 1
        
        del differences
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print_memory_usage("After applying differences")
        
        # Step 4: Extract LoRA from merged Chroma
        logger.info("\n" + "="*60)
        logger.info("Step 4: Extracting LoRA from merged Chroma model...")
        logger.info("="*60)
        try:
            lora_dict = LoRAExtractor.extract_lora_svd(
                str(merged_chroma_path), args.chroma_base, args.rank, args.device,
                mode=args.mode, similarity_threshold=args.similarity_threshold,
                chunk_size=args.chunk_size
            )
        except Exception as e:
            logger.error(f"Failed to extract LoRA: {type(e).__name__}: {e}")
            return 1
        
        if not lora_dict:
            logger.error("No LoRA weights extracted.")
            return 1
        
        # Verify and save
        num_triplets = len([k for k in lora_dict if '.lora_down.weight' in k])
        logger.info(f"Extracted {num_triplets} complete LoRA triplets")
        
        # Load metadata
        logger.info("Loading metadata from input LoRA...")
        metadata = load_lora_metadata(args.flux_lora)
        
        # Add extraction info
        metadata["chroma_lora_pairs"] = str(num_triplets)
        metadata["chroma_extraction_rank"] = str(args.rank)
        metadata["flux_lora_applied_count"] = str(applied_count)
        
        # Save final LoRA
        output_path = Path(args.output_lora)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Saving final LoRA with metadata...")
            save_file(lora_dict, str(output_path), metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to save output LoRA: {type(e).__name__}: {e}")
            return 1
        
        logger.info("\n" + "="*60)
        logger.info(f"Conversion complete! Chroma LoRA saved to: {output_path}")
        logger.info(f"Format: Chroma-compatible (underscores with lora_unet_ prefix)")
        logger.info(f"Extracted {num_triplets} LoRA triplets with rank {args.rank}")
        logger.info("="*60)
        
        # Verify the output
        logger.info("\nVerifying output LoRA structure...")
        verification = verify_output_lora(str(output_path))
        
        if verification.get('shape_test'):
            logger.info("Shape verification:")
            for test in verification['shape_test'][:1]:  # Show first test
                logger.info(f"  Sample layer: {test['layer']}")
                logger.info(f"    Down: {test['down_shape']} (rank, in_features)")
                logger.info(f"    Up: {test['up_shape']} (out_features, rank)")
                logger.info(f"    Result: {test['multiplication_result']}")
        
        # Cleanup temp files
        logger.info("Cleaning up temporary files...")
        try:
            if merged_flux_path.exists():
                merged_flux_path.unlink()
            if merged_chroma_path.exists():
                merged_chroma_path.unlink()
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.error("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        if DEBUG_MODE:
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1
    
    finally:
        # Final cleanup
        gc.collect()
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage("Final")

if __name__ == "__main__":
    sys.exit(main())