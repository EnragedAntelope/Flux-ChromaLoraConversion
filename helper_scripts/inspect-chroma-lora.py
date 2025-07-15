#!/usr/bin/env python3
"""
Standalone script to inspect Chroma LoRA files
"""

import argparse
import torch
from safetensors import safe_open
from pathlib import Path
import numpy as np
from collections import defaultdict

def inspect_lora(lora_path: str):
    """Inspect a LoRA file and show detailed information"""
    print(f"\n{'='*60}")
    print(f"Inspecting LoRA: {Path(lora_path).name}")
    print(f"File size: {Path(lora_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
    
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())
            
            # Get metadata
            metadata = {}
            if hasattr(f, 'metadata'):
                metadata = f.metadata()
                if metadata is None:
                    metadata = {}
            
            print(f"\nTotal keys: {len(keys)}")
            
            # Analyze structure
            stats = {
                "lora_down": 0,
                "lora_up": 0,
                "alpha": 0,
                "ranks": [],
                "key_patterns": defaultdict(int),
                "magnitudes": []
            }
            
            # Sample a working pair to test
            test_pair = None
            
            for key in keys:
                tensor = f.get_tensor(key)
                
                if ".lora_down.weight" in key:
                    stats["lora_down"] += 1
                    rank = min(tensor.shape)
                    stats["ranks"].append(rank)
                    stats["magnitudes"].append(torch.norm(tensor).item())
                    if not test_pair:
                        test_pair = key.replace(".lora_down.weight", "")
                elif ".lora_up.weight" in key:
                    stats["lora_up"] += 1
                elif ".alpha" in key:
                    stats["alpha"] += 1
                
                # Extract pattern
                base = key.split(".")[0]
                if "_" in base:
                    pattern = "_".join(base.split("_")[:3])
                else:
                    pattern = base
                stats["key_patterns"][pattern] += 1
            
            print(f"\nStructure:")
            print(f"  LoRA down weights: {stats['lora_down']}")
            print(f"  LoRA up weights: {stats['lora_up']}")
            print(f"  Alpha values: {stats['alpha']}")
            
            if stats["ranks"]:
                print(f"\nRanks: {sorted(set(stats['ranks']))}")
            
            if stats["magnitudes"]:
                print(f"\nWeight magnitudes:")
                print(f"  Mean: {np.mean(stats['magnitudes']):.4f}")
                print(f"  Std: {np.std(stats['magnitudes']):.4f}")
                print(f"  Min: {np.min(stats['magnitudes']):.4f}")
                print(f"  Max: {np.max(stats['magnitudes']):.4f}")
            
            print(f"\nKey patterns:")
            for pattern, count in sorted(stats["key_patterns"].items(), key=lambda x: -x[1])[:10]:
                print(f"  {pattern}: {count}")
            
            print(f"\nFirst 10 keys:")
            for key in keys[:10]:
                tensor = f.get_tensor(key)
                print(f"  {key} - shape: {tensor.shape}")
            
            # Test a LoRA pair
            if test_pair:
                print(f"\nTesting LoRA pair: {test_pair}")
                try:
                    down = f.get_tensor(f"{test_pair}.lora_down.weight")
                    up = f.get_tensor(f"{test_pair}.lora_up.weight")
                    
                    print(f"  Down shape: {down.shape}")
                    print(f"  Up shape: {up.shape}")
                    
                    # Test multiplication
                    result = up @ down
                    print(f"  Multiplication result shape: {result.shape}")
                    # FIX: Replaced Unicode checkmark with ASCII-safe text
                    print(f"  OK: LoRA multiplication works correctly")
                    
                    # Check if alpha exists
                    alpha_key = f"{test_pair}.alpha"
                    if alpha_key in keys:
                        alpha = f.get_tensor(alpha_key)
                        print(f"  Alpha value: {alpha.item() if alpha.numel() == 1 else alpha}")
                    
                except Exception as e:
                    # FIX: Replaced Unicode ballot X with ASCII-safe text
                    print(f"  ERROR: Error testing pair: {e}")
            
            # Show metadata
            if metadata:
                print(f"\nMetadata ({len(metadata)} keys):")
                
                # Look for important keys
                important_keys = [
                    "chroma_converted", "original_lora", "trigger_words",
                    "trigger_word", "instance_prompt", "ss_tag_frequency",
                    "chroma_lora_pairs", "chroma_extraction_rank"
                ]
                
                for key in important_keys:
                    if key in metadata:
                        value = metadata[key]
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                
                # Show all keys if not many
                if len(metadata) <= 20:
                    print("\nAll metadata keys:", list(metadata.keys()))
            else:
                print("\nNo metadata found")
                
    except Exception as e:
        print(f"Error reading LoRA: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Inspect Chroma LoRA file")
    parser.add_argument("lora", type=str, help="Path to LoRA file")
    
    args = parser.parse_args()
    
    if not Path(args.lora).exists():
        print(f"Error: File not found: {args.lora}")
        return 1
    
    inspect_lora(args.lora)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
