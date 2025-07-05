#!/usr/bin/env python3
"""
Flux to Chroma LoRA Converter - Advanced Version
With comparative interpolation and add dissimilar operations

Author: AI Assistant
License: MIT
"""

import os
import sys
import json
import struct
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from safetensors import safe_open

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for the conversion process"""
    flux_model_path: str
    chroma_model_path: str
    lora_path: str
    output_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    lora_rank: int = 64
    merge_strength: float = 1.0
    extraction_threshold: float = 1e-6
    use_memory_efficient_loading: bool = True
    chunk_size: int = 100
    save_intermediate: bool = False
    intermediate_dir: str = "./intermediate"
    debug_keys: bool = False
    # Enhanced parameters
    amplify_strength: float = 1.0
    force_extract_all: bool = False
    aggressive_mapping: bool = False
    # Advanced interpolation parameters
    use_comparative_interpolation: bool = True
    interpolation_noise_scale: float = 0.02  # Random distribution scale
    dissimilarity_threshold: float = 0.1  # Threshold for add dissimilar
    interpolation_clamp_range: float = 2.0  # Clamping range for random distribution
    adaptive_strength: bool = True  # Adjust strength based on layer similarity


class MemoryEfficientSafeOpen:
    """Memory efficient safetensors file reader"""
    def __init__(self, filename):
        self.filename = filename
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            # adjust offset by header size
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        # process float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # convert to the target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # add float8 types if available
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """Memory efficient save file"""
    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                logger.warning(f"Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    logger.info(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
            continue
        
        size = v.numel() * v.element_size()
        header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
        offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            
            byte_view = v.contiguous().view(torch.uint8) if v.dim() > 0 else v.unsqueeze(0).contiguous().view(torch.uint8)

            if v.is_cuda:
                byte_view.cpu().numpy().tofile(f)
            else:
                byte_view.numpy().tofile(f)


class FluxToChromaConverter:
    """Main converter class for Flux to Chroma LoRA conversion"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = self._get_dtype(config.dtype)
        
        # Create intermediate directory if needed
        if config.save_intermediate:
            Path(config.intermediate_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized converter with device: {self.device}, dtype: {self.dtype}")
        
        # Validate paths
        self._validate_paths()
    
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def _validate_paths(self):
        """Validate that all required files exist"""
        required_files = [
            ("Flux model", self.config.flux_model_path),
            ("Chroma model", self.config.chroma_model_path),
            ("LoRA file", self.config.lora_path)
        ]
        
        for name, path in required_files:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found at: {path}")
            logger.info(f"✓ Found {name}: {path}")

    def calculate_tensor_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate similarity between two tensors using cosine similarity"""
        if tensor1.shape != tensor2.shape:
            return 0.0
        
        # Flatten tensors
        t1_flat = tensor1.flatten().to(torch.float32)
        t2_flat = tensor2.flatten().to(torch.float32)
        
        # Calculate cosine similarity
        dot_product = torch.dot(t1_flat, t2_flat)
        norm1 = torch.norm(t1_flat)
        norm2 = torch.norm(t2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity.item()

    def comparative_interpolation(self, base: torch.Tensor, target: torch.Tensor, 
                                 reference: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Performs comparative interpolation with random distribution clamped.
        This is inspired by the developer's suggestion to interpolate differences
        with random distribution clamped.
        """
        # Calculate similarity between base and target
        similarity = self.calculate_tensor_similarity(base, target)
        
        # Calculate dissimilarity factor
        dissimilarity = 1.0 - abs(similarity)
        
        # Generate random noise with controlled distribution
        noise_shape = base.shape
        random_noise = torch.randn(noise_shape, device=base.device, dtype=base.dtype)
        
        # Scale noise based on dissimilarity and configuration
        noise_scale = self.config.interpolation_noise_scale * dissimilarity
        random_noise = random_noise * noise_scale
        
        # Clamp the random distribution
        clamp_range = self.config.interpolation_clamp_range
        random_noise = torch.clamp(random_noise, -clamp_range, clamp_range)
        
        # Calculate base difference
        diff = target - reference
        
        # Apply adaptive strength if enabled
        if self.config.adaptive_strength:
            # Adjust alpha based on dissimilarity - more dissimilar = stronger effect
            adaptive_alpha = alpha * (1.0 + dissimilarity * 0.5)
        else:
            adaptive_alpha = alpha
        
        # Perform interpolation with random perturbation
        # The random noise helps preserve texture and detail during conversion
        interpolated = base + adaptive_alpha * diff + random_noise
        
        return interpolated

    def add_dissimilar(self, tensor1: torch.Tensor, tensor2: torch.Tensor, 
                      threshold: float = None) -> torch.Tensor:
        """
        Add dissimilar operation - enhances regions where tensors are most different.
        This helps preserve unique characteristics during conversion.
        """
        if threshold is None:
            threshold = self.config.dissimilarity_threshold
        
        # Calculate element-wise difference
        diff = torch.abs(tensor1 - tensor2)
        
        # Normalize difference to [0, 1]
        diff_norm = diff / (diff.max() + 1e-8)
        
        # Create mask for dissimilar regions
        dissimilar_mask = diff_norm > threshold
        
        # Calculate enhancement factor for dissimilar regions
        enhancement = torch.where(
            dissimilar_mask,
            1.0 + diff_norm * self.config.amplify_strength,
            torch.ones_like(diff_norm)
        )
        
        # Apply enhancement
        result = tensor1 * enhancement
        
        return result

    def normalize_lora_key(self, key: str) -> str:
        """Normalize LoRA key by removing common prefixes"""
        # Remove various prefixes that might be present
        prefixes_to_remove = [
            'lora_unet_',
            'model.diffusion_model.',
            'diffusion_model.',
            'model.',
            'transformer.',
            'lora_te_',
            'lora_te1_',
            'lora_te2_',
        ]
        
        normalized_key = key
        for prefix in prefixes_to_remove:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        
        return normalized_key

    def analyze_lora_keys(self, lora_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, str], List[str]]:
        """Analyze LoRA keys to understand their structure"""
        lora_keys = sorted(lora_dict.keys())
        
        # Extract unique base keys (without .lora_down/.lora_up/.lora_A/.lora_B suffixes)
        base_keys = set()
        for key in lora_keys:
            # Normalize the key first
            normalized_key = self.normalize_lora_key(key)
            
            if '.lora_down.weight' in normalized_key:
                base_keys.add(normalized_key.replace('.lora_down.weight', ''))
            elif '.lora_up.weight' in normalized_key:
                base_keys.add(normalized_key.replace('.lora_up.weight', ''))
            elif '.lora_A.weight' in normalized_key:
                base_keys.add(normalized_key.replace('.lora_A.weight', ''))
            elif '.lora_B.weight' in normalized_key:
                base_keys.add(normalized_key.replace('.lora_B.weight', ''))
        
        logger.info(f"\nAnalyzing LoRA structure:")
        logger.info(f"Total LoRA tensors: {len(lora_keys)}")
        logger.info(f"Unique LoRA layers: {len(base_keys)}")
        
        if self.config.debug_keys and base_keys:
            logger.info("\nSample LoRA base keys (after normalization):")
            for i, key in enumerate(sorted(base_keys)[:10]):
                logger.info(f"  {key}")
        
        # Detect LoRA format
        lora_format = {}
        
        if any('.lora_down.weight' in k for k in lora_keys):
            lora_format['down_suffix'] = '.lora_down.weight'
            lora_format['up_suffix'] = '.lora_up.weight'
        else:
            lora_format['down_suffix'] = '.lora_A.weight'
            lora_format['up_suffix'] = '.lora_B.weight'
        
        # Store original keys for proper access
        lora_format['original_keys'] = lora_keys
        
        return lora_format, list(base_keys)

    def get_lora_weights_for_key(self, base_key: str, lora_dict: Dict[str, torch.Tensor], 
                                lora_format: Dict[str, str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get LoRA weights for a given base key, checking all possible prefixes"""
        possible_prefixes = ['', 'lora_unet_', 'model.diffusion_model.', 'diffusion_model.', 
                           'model.', 'transformer.']
        
        down_suffix = lora_format['down_suffix']
        up_suffix = lora_format['up_suffix']
        
        # Try each prefix
        for prefix in possible_prefixes:
            down_key = prefix + base_key + down_suffix
            up_key = prefix + base_key + up_suffix
            
            if down_key in lora_dict and up_key in lora_dict:
                return lora_dict[down_key], lora_dict[up_key]
        
        return None

    def merge_qkv_lora_weights(self, q_lora: Optional[Tuple[torch.Tensor, torch.Tensor]], 
                              k_lora: Optional[Tuple[torch.Tensor, torch.Tensor]], 
                              v_lora: Optional[Tuple[torch.Tensor, torch.Tensor]], 
                              qkv_weight: torch.Tensor,
                              alpha: float = 1.0,
                              strength: float = 1.0) -> torch.Tensor:
        """
        Merge separate Q, K, V LoRA weights into a combined QKV weight matrix.
        Enhanced with comparative interpolation.
        """
        # Get dimensions
        out_features, in_features = qkv_weight.shape
        head_dim = out_features // 3  # QKV is 3x the head dimension
        
        # Clone the original weight
        merged_weight = qkv_weight.clone()
        
        # Apply amplification if configured
        effective_strength = strength * self.config.amplify_strength
        
        # Process each component (Q, K, V)
        components = [
            (q_lora, 0, head_dim, "Q"),
            (k_lora, head_dim, 2 * head_dim, "K"),
            (v_lora, 2 * head_dim, 3 * head_dim, "V")
        ]
        
        for lora_weights, start_idx, end_idx, name in components:
            if lora_weights is not None:
                down, up = lora_weights
                
                # Calculate scale
                rank = down.shape[0]
                scale = alpha / rank
                
                # Apply LoRA: W = W + scale * up @ down * strength
                lora_delta = scale * torch.matmul(up, down) * effective_strength
                
                # Get the QKV slice
                qkv_slice = merged_weight[start_idx:end_idx, :]
                
                if self.config.use_comparative_interpolation:
                    # Use comparative interpolation for better quality
                    qkv_slice_new = self.comparative_interpolation(
                        qkv_slice, 
                        qkv_slice + lora_delta,
                        qkv_slice,
                        alpha=1.0
                    )
                    merged_weight[start_idx:end_idx, :] = qkv_slice_new
                else:
                    # Standard merge
                    merged_weight[start_idx:end_idx, :] += lora_delta
                
                if self.config.debug_keys:
                    logger.debug(f"    Applied {name} LoRA to QKV [{start_idx}:{end_idx}] with strength {effective_strength}")
        
        return merged_weight

    def enhanced_map_lora_key_to_model_key(self, lora_base_key: str, model_keys: List[str]) -> List[str]:
        """Enhanced mapping with more aggressive strategies"""
        # First normalize the key
        key = self.normalize_lora_key(lora_base_key)
        
        potential_keys = []
        
        # Direct mappings
        if f"{key}.weight" in model_keys:
            potential_keys.append(f"{key}.weight")
        
        # Standard mappings
        mappings = [
            ('single_transformer_blocks', 'double_blocks'),
            ('transformer_blocks', 'double_blocks'),
            ('attn.proj_out', 'attn.proj'),
            ('norm.linear', 'mod.lin'),
            ('proj_mlp', 'img_mlp.0'),
            ('proj_out', 'img_mlp.2'),
            ('proj_out', 'linear2'),  # Alternative mapping
            ('norm1.linear', 'img_mod.lin'),
            ('norm2.linear', 'txt_mod.lin'),
            ('norm1_context.linear', 'txt_mod.lin'),
            ('attn.to_out.0', 'attn.proj'),
        ]
        
        # Apply all mappings
        current_key = key
        for old, new in mappings:
            if old in current_key:
                mapped_key = current_key.replace(old, new)
                if f"{mapped_key}.weight" in model_keys:
                    potential_keys.append(f"{mapped_key}.weight")
                # Try cumulative replacements
                current_key = mapped_key
        
        # Try img/txt variants
        if 'attn' in key or 'mod' in key or 'mlp' in key:
            for variant in ['img_attn', 'txt_attn', 'img_mod', 'txt_mod', 'img_mlp', 'txt_mlp']:
                variant_key = key
                if 'attn' in key:
                    variant_key = key.replace('attn', variant)
                elif 'mod' in key:
                    variant_key = key.replace('mod', variant)
                elif 'mlp' in key:
                    variant_key = key.replace('mlp', variant)
                
                if f"{variant_key}.weight" in model_keys:
                    potential_keys.append(f"{variant_key}.weight")
        
        # Handle ff.net layers
        if 'ff.net' in key:
            mapped_key = key.replace('ff.net', 'img_mlp')
            if f"{mapped_key}.weight" in model_keys:
                potential_keys.append(f"{mapped_key}.weight")
            
            # Also try txt variant
            mapped_key = key.replace('ff.net', 'txt_mlp')
            if f"{mapped_key}.weight" in model_keys:
                potential_keys.append(f"{mapped_key}.weight")
        
        # Aggressive mapping: try to match by block number
        if self.config.aggressive_mapping:
            # Extract block number
            import re
            block_match = re.search(r'blocks\.(\d+)\.', key)
            if block_match:
                block_num = block_match.group(1)
                # Try all possible layer types for this block
                for layer_type in ['img_attn.proj', 'txt_attn.proj', 'img_mlp.0', 'img_mlp.2', 
                                  'txt_mlp.0', 'txt_mlp.2', 'img_mod.lin', 'txt_mod.lin']:
                    test_key = f"double_blocks.{block_num}.{layer_type}.weight"
                    if test_key in model_keys:
                        potential_keys.append(test_key)
        
        return list(set(potential_keys))  # Remove duplicates

    def merge_lora_to_model(self, model_state_dict: Dict[str, torch.Tensor], 
                           lora_state_dict: Dict[str, torch.Tensor], 
                           strength: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Merge LoRA weights into model with enhanced strategies
        """
        logger.info("Merging LoRA weights into model...")
        
        # Analyze LoRA structure
        lora_format, lora_base_keys = self.analyze_lora_keys(lora_state_dict)
        
        # Create a copy of the model
        merged_state_dict = {}
        for k, v in model_state_dict.items():
            merged_state_dict[k] = v.clone()
        
        # Group Q/K/V LoRAs by their base block
        qkv_lora_groups = {}
        regular_loras = {}
        
        for lora_base_key in lora_base_keys:
            # Check if this is a Q/K/V projection
            if any(x in lora_base_key for x in ['.to_q', '.to_k', '.to_v']):
                # Extract block identifier
                if 'single_transformer_blocks' in lora_base_key:
                    block_match = lora_base_key.split('single_transformer_blocks.')[1].split('.')[0]
                    block_key = f"single_transformer_blocks.{block_match}"
                elif 'transformer_blocks' in lora_base_key:
                    block_match = lora_base_key.split('transformer_blocks.')[1].split('.')[0]
                    block_key = f"transformer_blocks.{block_match}"
                else:
                    continue
                
                # Initialize group if needed
                if block_key not in qkv_lora_groups:
                    qkv_lora_groups[block_key] = {'q': None, 'k': None, 'v': None}
                
                # Get LoRA weights using the enhanced getter
                weights = self.get_lora_weights_for_key(lora_base_key, lora_state_dict, lora_format)
                if weights:
                    down, up = weights
                    # Store by type
                    if '.to_q' in lora_base_key:
                        qkv_lora_groups[block_key]['q'] = (down, up)
                    elif '.to_k' in lora_base_key:
                        qkv_lora_groups[block_key]['k'] = (down, up)
                    elif '.to_v' in lora_base_key:
                        qkv_lora_groups[block_key]['v'] = (down, up)
            else:
                # Regular LoRA
                regular_loras[lora_base_key] = True
        
        # Statistics
        applied_count = 0
        qkv_applied_count = 0
        regular_applied_count = 0
        
        # Apply QKV LoRAs
        logger.info(f"Processing {len(qkv_lora_groups)} QKV LoRA groups...")
        
        for block_key, qkv_group in tqdm(qkv_lora_groups.items(), desc="Applying QKV LoRAs"):
            # Map to model QKV key
            if 'single_transformer_blocks' in block_key:
                model_block = block_key.replace('single_transformer_blocks', 'double_blocks')
                qkv_keys = [
                    f"{model_block}.img_attn.qkv.weight",
                    f"{model_block}.txt_attn.qkv.weight"
                ]
            elif 'transformer_blocks' in block_key:
                model_block = block_key.replace('transformer_blocks', 'double_blocks')
                qkv_keys = [
                    f"{model_block}.img_attn.qkv.weight",
                    f"{model_block}.txt_attn.qkv.weight"
                ]
            else:
                continue
            
            # Try to apply to each possible QKV matrix
            for qkv_key in qkv_keys:
                if qkv_key in merged_state_dict:
                    try:
                        # Get alpha value
                        alpha = 1.0
                        if qkv_group['q'] is not None:
                            # Look for alpha in all possible prefixes
                            alpha_found = False
                            for prefix in ['', 'lora_unet_', 'model.diffusion_model.', 'diffusion_model.']:
                                alpha_key = f"{prefix}{block_key}.attn.to_q.alpha"
                                if alpha_key in lora_state_dict:
                                    alpha = lora_state_dict[alpha_key].item()
                                    alpha_found = True
                                    break
                            if not alpha_found:
                                alpha = qkv_group['q'][0].shape[0]
                        
                        # Convert tensors
                        qkv_weight = merged_state_dict[qkv_key].to(self.device, dtype=self.dtype)
                        
                        q_lora = None
                        k_lora = None
                        v_lora = None
                        
                        if qkv_group['q'] is not None:
                            q_lora = (qkv_group['q'][0].to(self.device, dtype=self.dtype),
                                     qkv_group['q'][1].to(self.device, dtype=self.dtype))
                        if qkv_group['k'] is not None:
                            k_lora = (qkv_group['k'][0].to(self.device, dtype=self.dtype),
                                     qkv_group['k'][1].to(self.device, dtype=self.dtype))
                        if qkv_group['v'] is not None:
                            v_lora = (qkv_group['v'][0].to(self.device, dtype=self.dtype),
                                     qkv_group['v'][1].to(self.device, dtype=self.dtype))
                        
                        # Merge QKV
                        merged_qkv = self.merge_qkv_lora_weights(
                            q_lora, k_lora, v_lora, qkv_weight, alpha, strength
                        )
                        
                        merged_state_dict[qkv_key] = merged_qkv.cpu()
                        qkv_applied_count += 1
                        
                        if self.config.debug_keys:
                            logger.debug(f"Applied QKV LoRA: {block_key} -> {qkv_key}")
                        
                    except Exception as e:
                        logger.debug(f"Failed to apply QKV LoRA to {qkv_key}: {e}")
        
        # Apply regular LoRAs with enhanced mapping and comparative interpolation
        logger.info(f"Processing {len(regular_loras)} regular LoRA layers...")
        
        for lora_base_key in tqdm(regular_loras.keys(), desc="Applying regular LoRAs"):
            # Skip if already handled as QKV
            if any(x in lora_base_key for x in ['.to_q', '.to_k', '.to_v']):
                continue
            
            # Get LoRA weights using enhanced getter
            weights = self.get_lora_weights_for_key(lora_base_key, lora_state_dict, lora_format)
            if not weights:
                continue
            
            down, up = weights
            
            # Use enhanced mapping
            model_keys = self.enhanced_map_lora_key_to_model_key(lora_base_key, list(merged_state_dict.keys()))
            
            for model_key in model_keys:
                try:
                    down_device = down.to(self.device, dtype=self.dtype)
                    up_device = up.to(self.device, dtype=self.dtype)
                    model_weight = merged_state_dict[model_key].to(self.device, dtype=self.dtype)
                    
                    # Get alpha - check all possible prefixes
                    alpha = down.shape[0]  # Default
                    for prefix in ['', 'lora_unet_', 'model.diffusion_model.', 'diffusion_model.']:
                        alpha_key = f"{prefix}{lora_base_key}.alpha"
                        if alpha_key in lora_state_dict:
                            alpha = lora_state_dict[alpha_key].item()
                            break
                    
                    # Apply LoRA with amplification
                    scale = alpha / down.shape[0]
                    effective_strength = strength * self.config.amplify_strength
                    lora_weight = scale * torch.matmul(up_device, down_device) * effective_strength
                    
                    if lora_weight.shape == model_weight.shape:
                        if self.config.use_comparative_interpolation:
                            # Use comparative interpolation
                            merged_weight = self.comparative_interpolation(
                                model_weight,
                                model_weight + lora_weight,
                                model_weight,
                                alpha=1.0
                            )
                        else:
                            # Standard merge
                            merged_weight = model_weight + lora_weight
                        
                        merged_state_dict[model_key] = merged_weight.cpu()
                        regular_applied_count += 1
                        
                        if self.config.debug_keys:
                            logger.debug(f"Applied regular LoRA: {lora_base_key} -> {model_key}")
                        break
                    
                except Exception as e:
                    logger.debug(f"Failed to apply regular LoRA {lora_base_key}: {e}")
        
        # Clean up
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        applied_count = qkv_applied_count + regular_applied_count
        
        # Report
        logger.info(f"\nLoRA Application Summary:")
        logger.info(f"  Total LoRA layers: {len(lora_base_keys)}")
        logger.info(f"  QKV groups processed: {len(qkv_lora_groups)}")
        logger.info(f"  QKV applications: {qkv_applied_count}")
        logger.info(f"  Regular applications: {regular_applied_count}")
        logger.info(f"  Total applications: {applied_count}")
        logger.info(f"  Application rate: {applied_count/len(lora_base_keys)*100:.1f}%")
        
        if self.config.amplify_strength != 1.0:
            logger.info(f"  Amplification factor: {self.config.amplify_strength}x")
        
        if applied_count == 0:
            logger.error("\nERROR: No LoRA layers were applied!")
            logger.error("The model architectures appear to be incompatible.")
        
        return merged_state_dict

    def check_differences_fp8_safe(self, dict1: Dict[str, torch.Tensor], 
                                  dict2: Dict[str, torch.Tensor]) -> bool:
        """Check if two model dicts have differences, handling FP8 tensors safely"""
        differences_found = False
        
        for key in dict1.keys():
            if key in dict2:
                tensor1 = dict1[key]
                tensor2 = dict2[key]
                
                # Skip if shapes don't match
                if tensor1.shape != tensor2.shape:
                    continue
                
                # Convert to float32 for comparison if needed
                if tensor1.dtype in [getattr(torch, 'float8_e4m3fn', None), 
                                    getattr(torch, 'float8_e5m2', None)]:
                    tensor1 = tensor1.to(torch.float32)
                    tensor2 = tensor2.to(torch.float32)
                
                # Check if tensors are different
                try:
                    if not torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-8):
                        differences_found = True
                        break
                except:
                    # If allclose fails, try element-wise comparison
                    if not torch.equal(tensor1, tensor2):
                        differences_found = True
                        break
        
        return differences_found

    def advanced_difference_merge(self, model_a_dict: Dict[str, torch.Tensor],
                                 model_b_dict: Dict[str, torch.Tensor],
                                 model_c_dict: Dict[str, torch.Tensor],
                                 alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Perform advanced difference merge using comparative interpolation and add dissimilar.
        A: Chroma base model
        B: Flux with LoRA merged
        C: Original Flux model
        """
        logger.info("Performing advanced difference merge with comparative interpolation...")
        merged_dict = {}
        
        all_keys = list(model_a_dict.keys())
        chunk_size = self.config.chunk_size
        
        # Track merge statistics
        merged_count = 0
        shape_mismatch_count = 0
        missing_key_count = 0
        dissimilar_enhanced_count = 0
        
        # Apply amplification to alpha
        effective_alpha = alpha * self.config.amplify_strength
        
        for i in tqdm(range(0, len(all_keys), chunk_size), desc="Processing chunks"):
            chunk_keys = all_keys[i:i + chunk_size]
            
            for key in chunk_keys:
                if key in model_b_dict and key in model_c_dict:
                    a = model_a_dict[key]
                    b = model_b_dict[key]
                    c = model_c_dict[key]
                    
                    if a.shape == b.shape == c.shape:
                        # Move to computation device and handle FP8
                        a_compute = a.to(self.device)
                        b_compute = b.to(self.device)
                        c_compute = c.to(self.device)
                        
                        # Convert FP8 to computation dtype
                        if hasattr(torch, 'float8_e4m3fn') and (
                            a.dtype == torch.float8_e4m3fn or 
                            b.dtype == torch.float8_e4m3fn or 
                            c.dtype == torch.float8_e4m3fn):
                            a_compute = a_compute.to(self.dtype)
                            b_compute = b_compute.to(self.dtype)
                            c_compute = c_compute.to(self.dtype)
                        
                        if self.config.use_comparative_interpolation:
                            # Use comparative interpolation
                            result = self.comparative_interpolation(
                                a_compute,  # base (Chroma)
                                b_compute,  # target (Flux with LoRA)
                                c_compute,  # reference (Original Flux)
                                alpha=effective_alpha
                            )
                            
                            # Apply add dissimilar with more aggressive detection
                            # Calculate difference magnitude instead of similarity
                            diff_magnitude = torch.norm(b_compute - c_compute) / (torch.norm(b_compute) + 1e-8)
                            if diff_magnitude > self.config.dissimilarity_threshold * 0.1:  # Much lower threshold
                                result = self.add_dissimilar(result, a_compute, threshold=self.config.dissimilarity_threshold * 0.2)
                                dissimilar_enhanced_count += 1
                        else:
                            # Standard difference merge
                            diff = b_compute - c_compute
                            result = a_compute + effective_alpha * diff
                        
                        merged_dict[key] = result.cpu()
                        merged_count += 1
                        
                        del a_compute, b_compute, c_compute, result
                    else:
                        merged_dict[key] = model_a_dict[key]
                        shape_mismatch_count += 1
                else:
                    merged_dict[key] = model_a_dict[key]
                    missing_key_count += 1
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        logger.info(f"Merge complete: {merged_count} layers merged, {shape_mismatch_count} shape mismatches, {missing_key_count} Chroma-only layers preserved")
        if self.config.use_comparative_interpolation:
            logger.info(f"Applied comparative interpolation with {dissimilar_enhanced_count} dissimilar enhancements")
        if effective_alpha != alpha:
            logger.info(f"Used amplified alpha: {effective_alpha} (base: {alpha})")
        return merged_dict

    def extract_lora_advanced(self, original_dict: Dict[str, torch.Tensor],
                             modified_dict: Dict[str, torch.Tensor],
                             rank: int = 64,
                             threshold: float = 1e-6) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA from the difference between two models using SVD
        with advanced dissimilarity handling
        """
        # Use lower threshold if force_extract_all is enabled
        effective_threshold = 1e-10 if self.config.force_extract_all else threshold
        
        logger.info(f"Extracting LoRA with rank={rank}, threshold={effective_threshold}")
        if self.config.use_comparative_interpolation:
            logger.info("Using advanced extraction with dissimilarity enhancement")
        
        lora_dict = {}
        extracted_count = 0
        below_threshold_count = 0
        dissimilar_enhanced_count = 0
        
        all_keys = list(original_dict.keys())
        chunk_size = self.config.chunk_size

        for i in tqdm(range(0, len(all_keys), chunk_size), desc="Extracting LoRA"):
            chunk_keys = all_keys[i:i + chunk_size]
            for key in chunk_keys:
                if key in modified_dict and key.endswith('.weight'):
                    original = original_dict[key].to(self.device, dtype=torch.float32)
                    modified = modified_dict[key].to(self.device, dtype=torch.float32)
                    
                    # Calculate diff and diff_norm before the conditional block
                    diff = modified - original
                    diff_norm = torch.norm(diff).item()

                    # Apply add dissimilar if configured with more aggressive detection
                    if self.config.use_comparative_interpolation:
                        # Use normalized difference as metric
                        diff_norm_ratio = diff_norm / (torch.norm(original) + 1e-8)
                        if diff_norm_ratio > self.config.dissimilarity_threshold * 0.01:  # Very low threshold
                            modified = self.add_dissimilar(modified, original, threshold=self.config.dissimilarity_threshold * 0.1)
                            # Recalculate diff and diff_norm if modified changes
                            diff = modified - original
                            diff_norm = torch.norm(diff).item()
                            dissimilar_enhanced_count += 1
                    
                    if diff_norm > effective_threshold:
                        original_shape = diff.shape
                        if diff.dim() > 2:
                            diff_2d = diff.reshape(original_shape[0], -1)
                        else:
                            diff_2d = diff
                        
                        try:
                            # Use more stable SVD computation
                            U, S, Vh = torch.linalg.svd(diff_2d, full_matrices=False)
                            
                            # Keep only top 'rank' components
                            actual_rank = min(rank, len(S))
                            U_r = U[:, :actual_rank]
                            S_r = S[:actual_rank]
                            Vh_r = Vh[:actual_rank, :]
                            
                            # Apply additional amplification to extracted values
                            amplify_factor = self.config.amplify_strength
                            
                            # Add controlled randomness if using comparative interpolation
                            if self.config.use_comparative_interpolation:
                                noise_scale = self.config.interpolation_noise_scale * 0.1  # Smaller noise for extraction
                                noise_down = torch.randn_like(Vh_r) * noise_scale
                                noise_up = torch.randn(U_r.shape[0], actual_rank, device=U_r.device, dtype=U_r.dtype) * noise_scale
                                
                                # Create properly sized LoRA matrices
                                lora_down = (Vh_r + noise_down) * amplify_factor  # Shape: [rank, in_features]
                                lora_up = (U_r @ torch.diag(torch.sqrt(S_r)) + noise_up) * amplify_factor  # Shape: [out_features, rank]
                                lora_up = lora_up @ torch.diag(torch.sqrt(S_r))  # Include singular values
                            else:
                                # Standard extraction with proper sizing
                                lora_down = Vh_r * amplify_factor  # Shape: [rank, in_features]
                                lora_up = (U_r @ torch.diag(S_r)) * amplify_factor  # Shape: [out_features, rank]
                            
                            # Store in LoRA format
                            base_key = key.replace('.weight', '')
                            lora_dict[f"{base_key}.lora_down.weight"] = lora_down.cpu().to(self.dtype)
                            lora_dict[f"{base_key}.lora_up.weight"] = lora_up.cpu().to(self.dtype)
                            
                            # Store alpha = rank (as float tensor for compatibility)
                            lora_dict[f"{base_key}.alpha"] = torch.tensor(float(actual_rank))
                            
                            extracted_count += 1
                            
                        except Exception as e:
                            logger.debug(f"SVD failed for {key}: {e}")
                    else:
                        below_threshold_count += 1
                        if self.config.debug_keys and below_threshold_count <= 5:
                            logger.debug(f"Below threshold: {key} (norm: {diff_norm:.2e})")
                    
                    del original, modified, diff
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        logger.info(f"Extracted LoRA for {extracted_count} layers ({below_threshold_count} below threshold)")
        if self.config.use_comparative_interpolation:
            logger.info(f"Enhanced {dissimilar_enhanced_count} dissimilar layers")
        if self.config.force_extract_all:
            logger.info("Force extract mode was enabled")
        
        # Calculate actual size - fix the check to look at keys, not tensor values
        total_params = sum(t.numel() for k, t in lora_dict.items() if 'alpha' not in k and isinstance(t, torch.Tensor))
        logger.info(f"Total LoRA parameters: {total_params:,}")
        
        return lora_dict

    def convert(self):
        """Main conversion workflow with advanced operations"""
        logger.info("Starting Flux to Chroma LoRA conversion (Advanced Version)...")
        logger.info("=" * 60)
        
        # Show enhanced settings
        if self.config.amplify_strength != 1.0:
            logger.info(f"Using amplification factor: {self.config.amplify_strength}x")
        if self.config.force_extract_all:
            logger.info("Force extract all differences enabled")
        if self.config.aggressive_mapping:
            logger.info("Aggressive key mapping enabled")
        if self.config.use_comparative_interpolation:
            logger.info("Comparative interpolation enabled")
            logger.info(f"  - Noise scale: {self.config.interpolation_noise_scale}")
            logger.info(f"  - Dissimilarity threshold: {self.config.dissimilarity_threshold}")
            logger.info(f"  - Clamp range: {self.config.interpolation_clamp_range}")
            if self.config.adaptive_strength:
                logger.info("  - Adaptive strength enabled")
        
        # Step 1: Load LoRA
        logger.info("\n=== Step 1: Loading LoRA ===")
        lora_dict = load_file(self.config.lora_path)
        logger.info(f"Loaded LoRA with {len(lora_dict)} tensors")
        
        # Check for model/diffusion_model prefixes
        sample_keys = list(lora_dict.keys())[:5]
        logger.info("Sample LoRA keys (checking for prefixes):")
        for key in sample_keys:
            logger.info(f"  {key}")
        
        # Step 2: Merge LoRA with Flux model
        logger.info("\n=== Step 2: Merging LoRA with Flux model ===")
        
        # Load Flux model
        logger.info("Loading Flux model...")
        flux_dict = load_file(self.config.flux_model_path)
        logger.info(f"Loaded Flux model with {len(flux_dict)} tensors")
        
        # Debug: Show sample keys if requested
        if self.config.debug_keys:
            logger.info("\nSample Flux model keys:")
            for key in sorted(flux_dict.keys())[:10]:
                logger.info(f"  {key}")
        
        # Merge LoRA
        flux_with_lora = self.merge_lora_to_model(
            flux_dict, lora_dict, self.config.merge_strength
        )
        
        # Check if merge was successful using FP8-safe comparison
        differences_found = self.check_differences_fp8_safe(flux_dict, flux_with_lora)
        
        if not differences_found:
            logger.warning("\nWARNING: No significant differences found after LoRA merge!")
            logger.warning("The conversion may produce a LoRA with minimal effect.")
            logger.warning("Continuing anyway...")
        
        del flux_dict  # Free memory
        
        # Save intermediate if requested
        if self.config.save_intermediate:
            intermediate_path = Path(self.config.intermediate_dir) / "flux_with_lora.safetensors"
            logger.info(f"Saving intermediate file: {intermediate_path}")
            save_file(flux_with_lora, str(intermediate_path))
        
        # Step 3: Advanced difference merge with Chroma
        logger.info("\n=== Step 3: Advanced difference merge with Chroma ===")
        
        # Load models
        chroma_dict = load_file(self.config.chroma_model_path)
        logger.info(f"Loaded Chroma model with {len(chroma_dict)} tensors")
        
        flux_original_dict = load_file(self.config.flux_model_path)
        logger.info(f"Reloaded original Flux model")
        
        # Perform advanced merge
        merged_chroma = self.advanced_difference_merge(
            chroma_dict, flux_with_lora, flux_original_dict, 
            alpha=self.config.merge_strength
        )
        
        # Clean up
        del flux_with_lora, flux_original_dict
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Save intermediate if requested
        if self.config.save_intermediate:
            intermediate_path = Path(self.config.intermediate_dir) / "merged_chroma.safetensors"
            logger.info(f"Saving intermediate file: {intermediate_path}")
            save_file(merged_chroma, str(intermediate_path))
        
        # Step 4: Extract LoRA from merged Chroma with advanced methods
        logger.info("\n=== Step 4: Extracting LoRA from merged Chroma (Advanced) ===")
        
        extracted_lora = self.extract_lora_advanced(
            chroma_dict, merged_chroma,
            rank=self.config.lora_rank,
            threshold=self.config.extraction_threshold
        )
        
        # Clean up
        del chroma_dict, merged_chroma
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Check extraction results
        if len(extracted_lora) == 0:
            logger.error("\nERROR: No LoRA layers were extracted!")
            logger.error("This means no significant differences were found.")
            raise RuntimeError("LoRA extraction failed - no layers extracted")
        
        # Step 5: Save final LoRA
        logger.info("\n=== Step 5: Saving final LoRA ===")
        
        # Handle output path
        output_path = Path(self.config.output_path)
        if output_path.is_dir():
            lora_filename = Path(self.config.lora_path).name
            new_filename = lora_filename.replace("_FLUX", "_CHROMA")
            if new_filename == lora_filename:
                base, ext = os.path.splitext(lora_filename)
                new_filename = f"{base}_chroma_advanced{ext}"
            
            final_output_path = output_path / new_filename
            logger.info(f"Output path is a directory. Saving to: {final_output_path}")
        else:
            final_output_path = output_path
        
        # Ensure parent directory exists
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metadata = {
            "format": "chroma-lora-advanced",
            "source": "flux-to-chroma-converter-advanced",
            "rank": str(self.config.lora_rank),
            "merge_strength": str(self.config.merge_strength),
            "amplify_strength": str(self.config.amplify_strength),
            "comparative_interpolation": str(self.config.use_comparative_interpolation),
            "interpolation_noise_scale": str(self.config.interpolation_noise_scale) if self.config.use_comparative_interpolation else "N/A",
            "dissimilarity_threshold": str(self.config.dissimilarity_threshold) if self.config.use_comparative_interpolation else "N/A",
            "original_lora": Path(self.config.lora_path).name,
            "extracted_layers": str(len(extracted_lora) // 3)
        }
        
        # Save
        if self.config.use_memory_efficient_loading:
            mem_eff_save_file(extracted_lora, str(final_output_path), metadata)
        else:
            save_file(extracted_lora, str(final_output_path), metadata)
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("CONVERSION COMPLETE (Advanced Version)")
        logger.info("=" * 60)
        logger.info(f"Output saved to: {final_output_path}")
        logger.info(f"LoRA tensors: {len(extracted_lora)}")
        logger.info(f"LoRA layers: {len(extracted_lora) // 3}")
        
        # Calculate actual size - fixed to check keys instead of values
        total_params = sum(t.numel() for k, t in extracted_lora.items() if 'alpha' not in k and isinstance(t, torch.Tensor))
        # Get dtype size
        if hasattr(self.dtype, 'itemsize'):
            dtype_size = self.dtype.itemsize
        else:
            # Fallback for older PyTorch versions
            dtype_size = torch.finfo(self.dtype).bits // 8
        size_mb = (total_params * dtype_size) / (1024 * 1024)
        logger.info(f"File size: ~{size_mb:.2f} MB")
        
        if self.config.use_comparative_interpolation:
            logger.info("\nAdvanced operations applied:")
            logger.info("✓ Comparative interpolation with random distribution")
            logger.info("✓ Add dissimilar enhancement for unique characteristics")
            logger.info("✓ Adaptive strength based on layer similarity")
        
        return extracted_lora


def main():
    parser = argparse.ArgumentParser(description="Convert Flux LoRA to Chroma LoRA - Advanced Version")
    parser.add_argument("--flux-model", required=True, help="Path to Flux UNet model")
    parser.add_argument("--chroma-model", required=True, help="Path to Chroma UNet model")
    parser.add_argument("--lora", required=True, help="Path to Flux LoRA")
    parser.add_argument("--output", required=True, help="Output path for Chroma LoRA")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"], 
                       help="Data type for computations")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank for extraction")
    parser.add_argument("--merge-strength", type=float, default=1.0, 
                       help="Strength for merging (0.0-1.0)")
    parser.add_argument("--threshold", type=float, default=1e-6, 
                       help="Threshold for LoRA extraction")
    parser.add_argument("--chunk-size", type=int, default=100, 
                       help="Number of layers to process at once")
    parser.add_argument("--save-intermediate", action="store_true", 
                       help="Save intermediate files")
    parser.add_argument("--intermediate-dir", default="./intermediate", 
                       help="Directory for intermediate files")
    parser.add_argument("--no-memory-efficient", action="store_true", 
                       help="Disable memory efficient loading")
    parser.add_argument("--debug-keys", action="store_true",
                       help="Show debug information about model keys")
    
    # Enhanced parameters
    parser.add_argument("--amplify", type=float, default=1.0,
                       help="Amplification factor for LoRA strength (1.0-3.0)")
    parser.add_argument("--force-extract-all", action="store_true",
                       help="Extract all differences, even very small ones")
    parser.add_argument("--aggressive-mapping", action="store_true",
                       help="Use aggressive key mapping strategies")
    
    # Advanced interpolation parameters
    parser.add_argument("--use-comparative-interpolation", action="store_true",
                       help="Use comparative interpolation with random distribution (recommended)")
    parser.add_argument("--interpolation-noise-scale", type=float, default=0.02,
                       help="Scale for random noise in interpolation (0.0-0.1)")
    parser.add_argument("--dissimilarity-threshold", type=float, default=0.1,
                       help="Threshold for add dissimilar operation (0.0-1.0)")
    parser.add_argument("--interpolation-clamp-range", type=float, default=2.0,
                       help="Clamping range for random distribution (1.0-5.0)")
    parser.add_argument("--adaptive-strength", action="store_true",
                       help="Enable adaptive strength based on layer similarity")
    
    args = parser.parse_args()
    
    config = ConversionConfig(
        flux_model_path=args.flux_model,
        chroma_model_path=args.chroma_model,
        lora_path=args.lora,
        output_path=args.output,
        device=args.device,
        dtype=args.dtype,
        lora_rank=args.rank,
        merge_strength=args.merge_strength,
        extraction_threshold=args.threshold,
        use_memory_efficient_loading=not args.no_memory_efficient,
        chunk_size=args.chunk_size,
        save_intermediate=args.save_intermediate,
        intermediate_dir=args.intermediate_dir,
        debug_keys=args.debug_keys,
        amplify_strength=args.amplify,
        force_extract_all=args.force_extract_all,
        aggressive_mapping=args.aggressive_mapping,
        use_comparative_interpolation=args.use_comparative_interpolation,
        interpolation_noise_scale=args.interpolation_noise_scale,
        dissimilarity_threshold=args.dissimilarity_threshold,
        interpolation_clamp_range=args.interpolation_clamp_range,
        adaptive_strength=args.adaptive_strength
    )
    
    try:
        converter = FluxToChromaConverter(config)
        converter.convert()
        logger.info("\n✓ Conversion successful!")
    except Exception as e:
        logger.error(f"\n✗ Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
