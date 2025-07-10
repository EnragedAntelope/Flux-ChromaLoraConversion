#!/usr/bin/env python3
"""
Analyze a working Chroma LoRA to understand the expected format
"""

import argparse
import torch
from safetensors import safe_open
from pathlib import Path
import numpy as np
from collections import defaultdict
import json

def analyze_chroma_lora(lora_path: str):
    """Analyze a working Chroma LoRA structure"""
    print(f"\n{'='*60}")
    print(f"Analyzing Chroma LoRA: {Path(lora_path).name}")
    print(f"File size: {Path(lora_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
    
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        
        # Get metadata
        metadata = {}
        if hasattr(f, 'metadata'):
            metadata = f.metadata()
            if metadata is None:
                metadata = {}
        
        print(f"\nTotal keys: {len(keys)}")
        
        # Analyze key structure
        key_info = {
            "lora_down": [],
            "lora_up": [],
            "alpha": [],
            "other": []
        }
        
        ranks = []
        base_keys = set()
        
        for key in keys:
            tensor = f.get_tensor(key)
            
            if ".lora_down.weight" in key:
                key_info["lora_down"].append(key)
                base_key = key.replace(".lora_down.weight", "")
                base_keys.add(base_key)
                ranks.append(tensor.shape[1] if len(tensor.shape) >= 2 else 1)
                
            elif ".lora_up.weight" in key:
                key_info["lora_up"].append(key)
                
            elif ".alpha" in key:
                key_info["alpha"].append(key)
                # Check alpha value
                alpha_val = tensor.item() if tensor.numel() == 1 else tensor[0].item()
                if alpha_val == 0:
                    print(f"WARNING: Zero alpha for {key}")
                    
            elif "lora" in key.lower():
                key_info["other"].append(key)
            else:
                key_info["other"].append(key)
        
        print(f"\nKey types:")
        print(f"  lora_down weights: {len(key_info['lora_down'])}")
        print(f"  lora_up weights: {len(key_info['lora_up'])}")
        print(f"  alpha values: {len(key_info['alpha'])}")
        print(f"  other keys: {len(key_info['other'])}")
        
        print(f"\nBase keys (unique layers): {len(base_keys)}")
        
        if ranks:
            print(f"\nRank distribution:")
            from collections import Counter
            rank_counts = Counter(ranks)
            for rank, count in sorted(rank_counts.items()):
                print(f"  Rank {rank}: {count} layers")
        
        # Show sample keys
        print(f"\nFirst 10 keys:")
        for key in keys[:10]:
            tensor = f.get_tensor(key)
            print(f"  {key} - shape: {tensor.shape}, dtype: {tensor.dtype}")
        
        # Analyze key naming patterns
        print(f"\nKey naming patterns:")
        
        # Check for dots vs underscores
        has_dots = any("." in key for key in base_keys)
        has_underscores = any("_" in key for key in base_keys)
        
        if has_dots and not has_underscores:
            print("  [OK] Uses dot notation (e.g., double_blocks.0.img_attn.proj)") # FIX: Replaced ✓ with [OK]
        elif has_underscores and not has_dots:
            print("  [FAIL] Uses underscore notation (e.g., double_blocks_0_img_attn_proj)") # FIX: Replaced ✗ with [FAIL]
        else:
            print("  [?] Mixed notation") # FIX: Replaced ? with [?]
        
        # Check key format
        sample_base = list(base_keys)[:1][0] if base_keys else None
        if sample_base:
            print(f"  Sample base key: {sample_base}")
            
        # Check for specific patterns
        patterns = {
            "double_blocks": sum(1 for k in base_keys if "double_blocks" in k),
            "single_blocks": sum(1 for k in base_keys if "single_blocks" in k),
            "img_attn": sum(1 for k in base_keys if "img_attn" in k),
            "txt_attn": sum(1 for k in base_keys if "txt_attn" in k),
            "img_mlp": sum(1 for k in base_keys if "img_mlp" in k),
            "txt_mlp": sum(1 for k in base_keys if "txt_mlp" in k),
        }
        
        print(f"\nLayer distribution:")
        for pattern, count in patterns.items():
            if count > 0:
                print(f"  {pattern}: {count}")
        
        # Check metadata
        if metadata:
            print(f"\nMetadata found: {len(metadata)} keys")
            print("Metadata keys:", list(metadata.keys())[:10])
            
            # Look for conversion info
            if "chroma_converted" in metadata:
                print(f"  [OK] Converted LoRA: {metadata.get('chroma_converted')}") # FIX: Replaced ✓ with [OK]
                print(f"  Conversion date: {metadata.get('chroma_conversion_date', 'N/A')}")
                print(f"  Original LoRA: {metadata.get('original_lora', 'N/A')}")
            
            # Look for trigger words
            trigger_found = False
            for key in ["trigger_words", "trigger_word", "instance_prompt", "ss_tag_frequency"]:
                if key in metadata:
                    print(f"  Trigger info ({key}): {str(metadata[key])[:100]}...")
                    trigger_found = True
            
            if not trigger_found:
                print("  No trigger word metadata found")
        else:
            print("\nNo metadata found")
        
        # Compare with expected structure
        print(f"\n{'='*60}")
        print("Structure Analysis:")
        print(f"{'='*60}")
        
        expected_pairs = len(base_keys)
        actual_down = len(key_info["lora_down"])
        actual_up = len(key_info["lora_up"])
        actual_alpha = len(key_info["alpha"])
        
        print(f"Expected structure: {expected_pairs} base keys × 3 (down, up, alpha) = {expected_pairs * 3} total")
        print(f"Actual structure: {actual_down} down + {actual_up} up + {actual_alpha} alpha = {actual_down + actual_up + actual_alpha} total")
        
        if actual_down == actual_up == actual_alpha == expected_pairs:
            print("[OK] Structure is complete and balanced") # FIX: Replaced ✓ with [OK]
        else:
            print("[FAIL] Structure mismatch!") # FIX: Replaced ✗ with [FAIL]
            if actual_down != expected_pairs:
                print(f"  Missing {expected_pairs - actual_down} lora_down weights")
            if actual_up != expected_pairs:
                print(f"  Missing {expected_pairs - actual_up} lora_up weights")
            if actual_alpha != expected_pairs:
                print(f"  Missing {expected_pairs - actual_alpha} alpha values")
        
        return {
            "path": lora_path,
            "total_keys": len(keys),
            "base_keys": len(base_keys),
            "ranks": list(set(ranks)) if ranks else [],
            "has_metadata": bool(metadata),
            "key_format": "dots" if has_dots and not has_underscores else "underscores" if has_underscores else "mixed",
            "sample_keys": keys[:10]
        }

def compare_loras(working_lora_info: dict, converted_lora_info: dict):
    """Compare working and converted LoRA structures"""
    print(f"\n{'='*60}")
    print("LoRA Comparison")
    print(f"{'='*60}")
    
    print(f"\nWorking LoRA: {Path(working_lora_info['path']).name}")
    print(f"  Total keys: {working_lora_info['total_keys']}")
    print(f"  Base keys: {working_lora_info['base_keys']}")
    print(f"  Ranks: {working_lora_info['ranks']}")
    print(f"  Key format: {working_lora_info['key_format']}")
    
    print(f"\nConverted LoRA: {Path(converted_lora_info['path']).name}")
    print(f"  Total keys: {converted_lora_info['total_keys']}")
    print(f"  Base keys: {converted_lora_info['base_keys']}")
    print(f"  Ranks: {converted_lora_info['ranks']}")
    print(f"  Key format: {converted_lora_info['key_format']}")
    
    # Check differences
    print(f"\nDifferences:")
    
    if working_lora_info['base_keys'] != converted_lora_info['base_keys']:
        print(f"  [FAIL] Different number of base keys: {working_lora_info['base_keys']} vs {converted_lora_info['base_keys']}") # FIX: Replaced ✗ with [FAIL]
    else:
        print(f"  [OK] Same number of base keys: {working_lora_info['base_keys']}") # FIX: Replaced ✓ with [OK]
    
    if working_lora_info['key_format'] != converted_lora_info['key_format']:
        print(f"  [FAIL] Different key format: {working_lora_info['key_format']} vs {converted_lora_info['key_format']}") # FIX: Replaced ✗ with [FAIL]
    else:
        print(f"  [OK] Same key format: {working_lora_info['key_format']}") # FIX: Replaced ✓ with [OK]
    
    # Compare sample keys
    print(f"\nSample key comparison:")
    print(f"Working LoRA first key: {working_lora_info['sample_keys'][0] if working_lora_info['sample_keys'] else 'N/A'}")
    print(f"Converted LoRA first key: {converted_lora_info['sample_keys'][0] if converted_lora_info['sample_keys'] else 'N/A'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze working Chroma LoRA structure")
    parser.add_argument("--working-lora", type=str, required=True, help="Path to working Chroma LoRA")
    parser.add_argument("--converted-lora", type=str, help="Path to converted Chroma LoRA to compare")
    parser.add_argument("--save-analysis", type=str, help="Save analysis to JSON file")
    
    args = parser.parse_args()
    
    # Analyze working LoRA
    working_info = analyze_chroma_lora(args.working_lora)
    
    # If converted LoRA provided, analyze and compare
    if args.converted_lora:
        converted_info = analyze_chroma_lora(args.converted_lora)
        compare_loras(working_info, converted_info)
    
    # Save analysis if requested
    if args.save_analysis:
        analysis = {
            "working_lora": working_info,
            "converted_lora": converted_info if args.converted_lora else None
        }
        
        with open(args.save_analysis, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {args.save_analysis}")

if __name__ == "__main__":
    main()