#!/usr/bin/env python3
"""
Diagnostic script to understand why converted Chroma LoRAs have no effect
"""

import argparse
import torch
from safetensors import safe_open
from pathlib import Path
import numpy as np
from collections import defaultdict
import json
from typing import Optional, Dict, Any

def analyze_lora_structure(lora_path: str, label: str = "LoRA"):
    """Analyze the structure and statistics of a LoRA file"""
    print(f"\n{'='*60}")
    print(f"Analyzing {label}: {Path(lora_path).name}")
    print(f"{'='*60}")
    
    stats = {
        "total_keys": 0,
        "lora_pairs": defaultdict(int),
        "key_patterns": defaultdict(int),
        "magnitude_stats": [],
        "rank_distribution": defaultdict(int),
        "layer_types": defaultdict(int),
        "sample_keys": [],
        "metadata": {}
    }
    
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = sorted(f.keys())
        stats["total_keys"] = len(keys)
        stats["sample_keys"] = keys[:10]
        
        # Get metadata
        if hasattr(f, 'metadata') and f.metadata() is not None:
            stats["metadata"] = f.metadata()
        
        # Analyze each key
        for key in keys:
            tensor = f.get_tensor(key)
            
            # Extract base layer name
            if ".lora_down.weight" in key:
                base_key = key.replace(".lora_down.weight", "")
                stats["lora_pairs"]["down"] += 1
                
                # Get magnitude
                magnitude = torch.norm(tensor).item()
                stats["magnitude_stats"].append(magnitude)
                
                # Get rank
                # Use shape[0] for rank in down weight as it's (rank, in_features)
                rank = tensor.shape[0] if len(tensor.shape) >= 1 else 1
                stats["rank_distribution"][rank] += 1
                
                # Categorize layer type
                if "double_blocks" in key:
                    stats["layer_types"]["double_blocks"] += 1
                elif "single_blocks" in key:
                    stats["layer_types"]["single_blocks"] += 1
                else:
                    stats["layer_types"]["other"] += 1
                    
            elif ".lora_up.weight" in key:
                stats["lora_pairs"]["up"] += 1
            elif ".alpha" in key:
                stats["lora_pairs"]["alpha"] += 1
                alpha_value = tensor.item() if tensor.numel() == 1 else tensor[0].item()
                if alpha_value == 0:
                    print(f"WARNING: Zero alpha value for {key}")
            
            # Extract key pattern
            parts = key.split(".")
            if len(parts) >= 2:
                pattern = f"{parts[0]}.{parts[1]}"
                stats["key_patterns"][pattern] += 1
    
    # Print analysis
    print(f"\nTotal keys: {stats['total_keys']}")
    print(f"LoRA pairs: down={stats['lora_pairs']['down']}, up={stats['lora_pairs']['up']}, alpha={stats['lora_pairs']['alpha']}")
    
    print(f"\nRank distribution:")
    for rank, count in sorted(stats["rank_distribution"].items()):
        print(f"  Rank {rank}: {count} layers")
    
    print(f"\nLayer type distribution:")
    for layer_type, count in sorted(stats["layer_types"].items()):
        print(f"  {layer_type}: {count}")
    
    if stats["magnitude_stats"]:
        print(f"\nMagnitude statistics (for lora_down weights):")
        print(f"  Mean: {np.mean(stats['magnitude_stats']):.6f}")
        print(f"  Std: {np.std(stats['magnitude_stats']):.6f}")
        print(f"  Min: {np.min(stats['magnitude_stats']):.6f}")
        print(f"  Max: {np.max(stats['magnitude_stats']):.6f}")
        print(f"  Zero/near-zero (mag < 1e-6): {sum(1 for m in stats['magnitude_stats'] if m < 1e-6)}")
    
    print(f"\nKey patterns (top 10):")
    for pattern, count in sorted(stats["key_patterns"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")
    
    print(f"\nSample keys:")
    for key in stats["sample_keys"]:
        print(f"  {key}")
    
    # Check metadata
    if stats["metadata"]:
        print(f"\nMetadata keys: {list(stats['metadata'].keys())[:10]}")
        
        # Look for trigger words
        trigger_keys = ["trigger_words", "trigger_word", "instance_prompt", "ss_tag_frequency"]
        for tk in trigger_keys:
            if tk in stats["metadata"]:
                value = stats["metadata"][tk]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {tk}: {value[:100]}...")
                else:
                    print(f"  {tk}: {value}")
    
    return stats

def compare_lora_keys(flux_lora: str, chroma_lora: str):
    """Compare keys between Flux and Chroma LoRAs"""
    print(f"\n{'='*60}")
    print("Comparing LoRA Keys")
    print(f"{'='*60}")
    
    # Load keys from both LoRAs
    with safe_open(flux_lora, framework="pt", device="cpu") as f:
        flux_keys = set(f.keys())
    
    with safe_open(chroma_lora, framework="pt", device="cpu") as f:
        chroma_keys = set(f.keys())
    
    # Extract base keys (without .lora_down/.lora_up/.alpha)
    def get_base_keys(keys):
        base_keys = set()
        for key in keys:
            base_key = key
            for suffix in [".lora_down.weight", ".lora_up.weight", ".alpha"]:
                if key.endswith(suffix):
                    base_key = key[:-len(suffix)]
                    break
            base_keys.add(base_key)
        return base_keys
    
    flux_base = get_base_keys(flux_keys)
    chroma_base = get_base_keys(chroma_keys)
    
    # Filter out text encoder keys from Flux
    flux_unet_base = {k for k in flux_base if not k.startswith("lora_te")}
    
    print(f"\nFlux LoRA base keys: {len(flux_base)} (UNet only: {len(flux_unet_base)})")
    print(f"Chroma LoRA base keys: {len(chroma_base)}")
    
    # Check if they're the same
    if flux_unet_base == chroma_base:
        print("[OK] Keys match perfectly!")
    else:
        print("[X] Keys differ!")
        
        only_flux = flux_unet_base - chroma_base
        only_chroma = chroma_base - flux_unet_base
        
        if only_flux:
            print(f"\nKeys only in Flux LoRA ({len(only_flux)}):")
            for key in sorted(list(only_flux))[:10]:
                print(f"  {key}")
        
        if only_chroma:
            print(f"\nKeys only in Chroma LoRA ({len(only_chroma)}):")
            for key in sorted(list(only_chroma))[:10]:
                print(f"  {key}")
    
    # Check key naming patterns
    print(f"\nChecking key naming patterns...")
    
    # Sample some Chroma keys
    sample_chroma = sorted(list(chroma_base))[:5]
    print(f"\nSample Chroma LoRA keys:")
    for key in sample_chroma:
        print(f"  {key}")
        # Check if this would match any Chroma model key
        if "." not in key:
            print(f"    WARNING: No dots in key - might not match Chroma model structure")

def test_lora_application(chroma_base: str, chroma_lora: str, test_layer: Optional[str] = None):
    """Test if LoRA can be applied to Chroma model"""
    print(f"\n{'='*60}")
    print("Testing LoRA Application")
    print(f"{'='*60}")
    
    # If no test layer specified, find one
    if not test_layer:
        with safe_open(chroma_lora, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".lora_down.weight" in key:
                    test_layer = key.replace(".lora_down.weight", "")
                    break
    
    if not test_layer:
        print("ERROR: No LoRA layers found!")
        return
    
    print(f"Testing with layer: {test_layer}")
    
    try:
        # Load the base weight
        base_key = test_layer + ".weight"
        with safe_open(chroma_base, framework="pt", device="cpu") as f:
            base_model_keys = f.keys()
            if base_key in base_model_keys:
                base_weight = f.get_tensor(base_key)
                print(f"[OK] Found base weight: {base_key} shape={base_weight.shape}")
            else:
                print(f"[X] Base weight not found: {base_key}")
                print("  Available keys with similar pattern:")
                similar = [k for k in base_model_keys if test_layer.split(".")[-1] in k][:5]
                for k in similar:
                    print(f"    {k}")
                return
        
        # Load LoRA weights
        with safe_open(chroma_lora, framework="pt", device="cpu") as f:
            lora_down = f.get_tensor(f"{test_layer}.lora_down.weight")
            lora_up = f.get_tensor(f"{test_layer}.lora_up.weight")
            
            print(f"[OK] Found LoRA down: shape={lora_down.shape}")
            print(f"[OK] Found LoRA up: shape={lora_up.shape}")
            
            # Check if alpha exists
            alpha_key = f"{test_layer}.alpha"
            if alpha_key in f.keys():
                alpha = f.get_tensor(alpha_key).item()
                print(f"[OK] Found alpha: {alpha}")
            else:
                print("[X] No alpha value found - using rank as fallback")
                # The rank is the first dimension of lora_down
                alpha = lora_down.shape[0]
        
        # Test application
        print(f"\nTesting LoRA application:")
        
        lora_rank = lora_down.shape[0]
        # Compute LoRA weight
        lora_weight = (lora_up @ lora_down) * (alpha / lora_rank)
        print(f"  LoRA weight shape: {lora_weight.shape}")
        print(f"  LoRA weight magnitude: {torch.norm(lora_weight).item():.6f}")
        
        # Check shapes
        if lora_weight.shape == base_weight.shape:
            print("[OK] Shapes match - LoRA can be applied")
            
            # Compute effect magnitude
            base_norm = torch.norm(base_weight).item()
            lora_norm = torch.norm(lora_weight).item()
            relative_effect = lora_norm / (base_norm + 1e-8)
            
            print(f"  Base weight norm: {base_norm:.6f}")
            print(f"  LoRA effect: {relative_effect:.2%} of base magnitude")
            
            if relative_effect < 0.001:
                print("  WARNING: LoRA effect is very small!")
        else:
            print(f"[X] Shape mismatch! Base: {base_weight.shape}, LoRA: {lora_weight.shape}")
            
    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Diagnose Chroma LoRA issues")
    parser.add_argument("--flux-lora", type=str, help="Original Flux LoRA")
    parser.add_argument("--chroma-lora", type=str, required=True, help="Converted Chroma LoRA")
    parser.add_argument("--chroma-base", type=str, help="Chroma base model for testing")
    parser.add_argument("--test-layer", type=str, help="Specific layer to test")
    parser.add_argument("--save-report", type=str, help="Save analysis to JSON file")
    
    args = parser.parse_args()
    
    report_data = {}
    
    # Analyze Chroma LoRA
    chroma_stats = analyze_lora_structure(args.chroma_lora, "Chroma LoRA")
    report_data["chroma_lora"] = chroma_stats
    
    # If Flux LoRA provided, analyze and compare
    if args.flux_lora:
        flux_stats = analyze_lora_structure(args.flux_lora, "Flux LoRA")
        report_data["flux_lora"] = flux_stats
        compare_lora_keys(args.flux_lora, args.chroma_lora)
    
    # If Chroma base provided, test application
    if args.chroma_base:
        test_lora_application(args.chroma_base, args.chroma_lora, args.test_layer)
    
    # Save report if requested
    if args.save_report:
        full_report = {
            "analysis": report_data,
            "inputs": {
                "chroma_lora_path": args.chroma_lora,
                "flux_lora_path": args.flux_lora,
                "chroma_base_path": args.chroma_base,
            }
        }
        
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.save_report}")

if __name__ == "__main__":
    main()
