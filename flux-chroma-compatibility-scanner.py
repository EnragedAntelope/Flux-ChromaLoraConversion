#!/usr/bin/env python3
"""
Flux to Chroma LoRA Compatibility Scanner v7
Analyzes Flux LoRAs for Chroma conversion compatibility.
Aligned with converter v17.0, which treats Text Encoder conversion as
experimental and disabled by default. The score is now based purely on UNet
compatibility.
"""

import argparse
import os
import sys
from pathlib import Path
from safetensors import safe_open
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import defaultdict, Counter
import json

# Key mapping patterns - ALIGNED WITH CONVERTER v17.0 for consistency
DIFFUSERS_KEY_MAPPING = {
    # Single blocks - Flux only has linear1, linear2, and norm layers
    r"single_transformer_blocks\.(\d+)\.attn\.to_q": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.attn\.to_k": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.attn\.to_v": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.attn\.to_out\.0": "single_blocks.{}.linear2",
    r"single_transformer_blocks\.(\d+)\.ff\.net\.0\.proj": "single_blocks.{}.linear1",
    r"single_transformer_blocks\.(\d+)\.ff\.net\.2": "single_blocks.{}.linear2",
    r"single_transformer_blocks\.(\d+)\.proj_mlp": "single_blocks.{}.linear1", # Alias for ff.net.0.proj
    r"single_transformer_blocks\.(\d+)\.proj_out": "single_blocks.{}.linear2",
    r"single_transformer_blocks\.(\d+)\.norm\.linear": None,  # Skip - doesn't exist in flux1-dev

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
    
    # Handle norm layers for double blocks - these don't exist in flux-dev, so skip them.
    r"transformer_blocks\.(\d+)\.norm1\.linear": None, # Skip - doesn't exist in flux1-dev
    r"transformer_blocks\.(\d+)\.norm1_context\.linear": None, # Skip - doesn't exist in flux1-dev
    
    # Diffusers-specific attention layers (these are aliases for the above, handled by the merger)
    r"transformer_blocks\.(\d+)\.attn\.add_q_proj": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn\.add_k_proj": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn\.add_v_proj": "double_blocks.{}.img_attn.qkv",
    r"transformer_blocks\.(\d+)\.attn\.to_add_out": "double_blocks.{}.img_attn.proj",
    
    # General text encoder patterns (skip these from UNet normalization)
    r"text_encoder.*": None,
    r"te_.*": None,
    r"lora_te.*": None,
    r"lora_te1.*": None,
    r"lora_te2.*": None,
    r"unet\.time_embedding.*": None,
    r"unet\.label_emb.*": None,
}

# Additional direct mappings for edge cases
DIRECT_KEY_MAPPING = {
    # These are exact replacements, not patterns
    "transformer.proj_out": None,  # Skip - doesn't exist
    "transformer.norm_out.linear": None,  # Skip - doesn't exist
}

def normalize_lora_key(key: str) -> Optional[str]:
    """
    Normalize a LoRA key to match Flux structure.
    This function is an exact copy of the one in the converter for consistency.
    """
    original_key = key
    
    # Remove .lora_A.weight, .lora_B.weight suffixes
    key = re.sub(r'\.lora_[AB]\.weight$', '', key)
    key = re.sub(r'\.lora_(up|down)\.weight$', '', key)
    key = re.sub(r'\.alpha$', '', key)
    
    # Remove prefix if present
    if key.startswith('lora_unet_'):
        # This logic handles pre-normalized keys but the converter is more robust
        key = key[len('lora_unet_'):].replace('_', '.')
        return key
    elif any(key.startswith(prefix) for prefix in ['lora_te', 'text_encoder', 'te_', 'lora_te1', 'lora_te2']):
        return None  # Skip text encoder from UNet normalization
    elif key.startswith('unet.'):
        key = key[5:]
    elif key.startswith('transformer.'):
        key = key[12:]
    elif key.startswith('model.'):
        key = key[6:]
    elif key.startswith('diffusion_model.'):
        key = key[16:]
    
    # Check direct mappings first
    if key in DIRECT_KEY_MAPPING:
        return DIRECT_KEY_MAPPING[key]
    
    # Try pattern-based mappings
    for pattern, replacement in DIFFUSERS_KEY_MAPPING.items():
        match = re.match(pattern, key)
        if match:
            if replacement is None:
                return None
            else:
                return replacement.format(*match.groups())
    
    # If no mapping found, return the key as-is for the analyzer to classify
    return key

def is_modulation_layer(key: str) -> bool:
    """Check if a key is a true modulation layer (exists in Flux but not Chroma)"""
    modulation_keywords = [
        "_mod.", ".mod.", "_mod_", ".modulation.",
        "mod_out", "norm_out", "scale_shift",
        "mod.lin", "modulated", "norm_k.", "norm_q.",
        "img_mod", "txt_mod", "vector_in",
        "guidance_in", "timestep_embedder"
    ]
    
    for mod in modulation_keywords:
        if mod in key:
            if "norm_added" in key:
                continue
            return True
    
    return False

def detect_lora_format(keys: List[str]) -> str:
    """Detect the LoRA format based on key patterns"""
    if any(".lora_up." in k or ".lora_down." in k for k in keys):
        return "diffusers"
    elif any(".lora_A." in k or ".lora_B." in k for k in keys):
        return "kohya"
    return "unknown"

def get_base_key(key: str) -> str:
    """Extract base key from LoRA key"""
    base = re.sub(r'\.lora_[AB]\.weight$', '', key)
    base = re.sub(r'\.lora_(up|down)\.weight$', '', base)
    base = re.sub(r'\.alpha$', '', base)
    return base

def analyze_lora_keys(keys: List[str]) -> Dict[str, Any]:
    """Analyze LoRA keys for patterns and structure"""
    analysis: Dict[str, Any] = {
        "total_keys": len(keys),
        "unet_keys": 0,
        "text_encoder_keys": 0,
        "format": "unknown",
        "layer_types": Counter(),
        "block_distribution": Counter(),
    }
    
    analysis["format"] = detect_lora_format(keys)
    
    base_keys = set()
    for key in keys:
        if any(te in key for te in ["text_encoder", "lora_te", "te_", "lora_te1", "lora_te2"]):
            analysis["text_encoder_keys"] += 1
        else:
            analysis["unet_keys"] += 1
        
        base = get_base_key(key)
        if not base.endswith((".weight", ".alpha")):
            base_keys.add(base)
    
    for base in base_keys:
        if "attn" in base:
            analysis["layer_types"]["attention"] += 1
        elif "ff" in base or "mlp" in base:
            analysis["layer_types"]["feedforward"] += 1
        elif "norm" in base:
            analysis["layer_types"]["normalization"] += 1
        
        if "single_transformer_blocks" in base:
            analysis["block_distribution"]["single_blocks"] += 1
        elif "transformer_blocks" in base:
            analysis["block_distribution"]["double_blocks"] += 1
            
    analysis["unique_base_keys"] = len(base_keys)
    return analysis

def check_single_lora_compatibility(lora_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Check compatibility of a single LoRA file"""
    result: Dict[str, Any] = {
        "path": lora_path,
        "compatible": False,
        "score": 0.0,
        "issues": [],
        "warnings": [],
        "stats": {},
        "conversion_stats": {
            "total_unet_pairs": 0,
            "convertible_unet_pairs": 0,
            "experimental_text_encoder_pairs": 0,
            "will_accumulate": 0,
            "will_skip_modulation": 0,
            "will_skip_unsupported": 0,
        }
    }
    
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            metadata = f.metadata() or {}
            analysis = analyze_lora_keys(keys)
            result["stats"] = analysis
            
            trigger_words = [f"{k}: {v}" for k, v in metadata.items() if any(w in k.lower() for w in ["trigger", "prompt", "keyword", "instance"])]
            if trigger_words:
                result["trigger_words"] = trigger_words
            
            result["file_size_mb"] = round(os.path.getsize(lora_path) / (1024 * 1024), 2)
            
            base_keys = set()
            normalized_mapping = {}
            target_layer_groups = defaultdict(list)
            
            # Count total pairs first
            total_pairs = len({get_base_key(k) for k in keys if not k.endswith('.alpha')})

            for key in keys:
                if "alpha" in key or not any(x in key for x in ["lora_down", "lora_up", "lora_A", "lora_B"]):
                    continue
                
                base = get_base_key(key)
                if base in base_keys:
                    continue
                base_keys.add(base)
                
                # Check for T5 text encoder keys first (lora_te1). These are now experimental.
                if base.startswith("lora_te1_"):
                    result["conversion_stats"]["experimental_text_encoder_pairs"] += 1
                    continue
                
                # Skip other TE keys (like CLIP/lora_te2) which are not used by Chroma
                if any(te in base for te in ["text_encoder", "lora_te2", "te_"]):
                    result["conversion_stats"]["will_skip_unsupported"] += 1
                    continue
                
                # This is a UNet key
                result["conversion_stats"]["total_unet_pairs"] += 1
                normalized = normalize_lora_key(base)
                
                if normalized is None:
                    result["conversion_stats"]["will_skip_unsupported"] += 1
                elif is_modulation_layer(normalized):
                    result["conversion_stats"]["will_skip_modulation"] += 1
                else:
                    result["conversion_stats"]["convertible_unet_pairs"] += 1
                    normalized_mapping[base] = normalized
                    target_layer_groups[normalized].append(base)
            
            result["conversion_stats"]["will_accumulate"] = sum(1 for sources in target_layer_groups.values() if len(sources) > 1)
            
            # --- Scoring and Recommendation Logic ---
            conv_stats = result["conversion_stats"]
            total_unet_pairs = conv_stats["total_unet_pairs"]
            
            if total_unet_pairs > 0:
                # Score is based purely on UNet conversion quality
                conversion_rate = conv_stats["convertible_unet_pairs"] / total_unet_pairs
                result["score"] = conversion_rate * 100
                result["compatible"] = conversion_rate >= 0.5
                if conversion_rate >= 0.7:
                    result["recommendation"] = "✅ Good candidate for conversion."
                elif conversion_rate >= 0.5:
                    result["recommendation"] = "⚠️ Fair candidate. May have partial effect."
                else:
                    result["recommendation"] = "❌ Poor candidate. Most of the LoRA will be lost."
            else:
                # No UNet layers found
                result["score"] = 0.0
                result["compatible"] = False
                if conv_stats["experimental_text_encoder_pairs"] > 0:
                    result["recommendation"] = "❌ Not recommended (Text Encoder only, conversion is experimental and likely to fail)."
                    result["issues"].append("This appears to be a Text Encoder-only LoRA.")
                else:
                    result["recommendation"] = "❌ Not recommended for conversion."
                    result["issues"].append("No convertible UNet or Text Encoder layers found.")

            # Add warnings based on what will be skipped
            if conv_stats["will_skip_modulation"] > 0:
                pct = conv_stats["will_skip_modulation"] / total_unet_pairs * 100 if total_unet_pairs > 0 else 0
                result["warnings"].append(f"{pct:.0f}% of UNet layers are modulation (incompatible)")
            
            if conv_stats["will_skip_unsupported"] > 0:
                pct = conv_stats["will_skip_unsupported"] / total_unet_pairs * 100 if total_unet_pairs > 0 else 0
                result["warnings"].append(f"{pct:.0f}% of UNet layers are unsupported or non-T5 TE (will be skipped)")

            # Find rank
            for key in keys:
                if ".lora_down.weight" in key or ".lora_A.weight" in key:
                    result["rank"] = min(f.get_tensor(key).shape)
                    break
                
    except Exception as e:
        result["issues"].append(f"Error analyzing file: {str(e)}")
        result["compatible"] = False
        result["score"] = 0.0
        result["recommendation"] = "❌ Error during analysis."

    return result

def scan_directory(directory: str, detailed: bool) -> List[Dict[str, Any]]:
    """Scan directory for LoRA files and rank by compatibility"""
    results = []
    
    lora_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.safetensors')]
    
    print(f"Found {len(lora_files)} LoRA files, analyzing...")
    
    for lora_path in lora_files:
        print(f"  -> Analyzing: {os.path.basename(lora_path)}")
        result = check_single_lora_compatibility(lora_path, verbose=detailed)
        results.append(result)
    
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results

def format_compatibility_report(result: Dict[str, Any], detailed: bool = False) -> str:
    """Format compatibility analysis as readable report"""
    lines = ["="*70, f"LoRA: {os.path.basename(result['path'])}", "="*70]
    
    score = result.get("score", 0)
    lines.append(f"UNet Compatibility Score: {score:.1f}%")
    lines.append(f"Recommendation: {result.get('recommendation', 'N/A')}")
    lines.append(f"File Size: {result.get('file_size_mb', 0)} MB | Rank: {result.get('rank', 'N/A')} | Format: {result['stats'].get('format', 'unknown')}")
    
    conv_stats = result.get("conversion_stats", {})
    total_unet_pairs = conv_stats.get("total_unet_pairs", 0)
    if total_unet_pairs > 0:
        lines.append("\nUNet Conversion Potential:")
        lines.append(f"  - Convertible Pairs: {conv_stats['convertible_unet_pairs']} / {total_unet_pairs}")
        if conv_stats.get("will_accumulate", 0) > 0:
            lines.append(f"  - Layers to Accumulate: {conv_stats['will_accumulate']}")
        
        skipped_items = []
        if conv_stats.get("will_skip_modulation", 0) > 0:
            skipped_items.append(f"Modulation ({conv_stats['will_skip_modulation']})")
        if conv_stats.get("will_skip_unsupported", 0) > 0:
            skipped_items.append(f"Unsupported ({conv_stats['will_skip_unsupported']})")
        if skipped_items:
            lines.append(f"  - Skipped Pairs: " + ", ".join(skipped_items))

    if conv_stats.get("experimental_text_encoder_pairs", 0) > 0:
        lines.append("\nText Encoder:")
        lines.append(f"  - Experimental T5 TE Pairs: {conv_stats['experimental_text_encoder_pairs']} (Conversion is opt-in and may fail)")

    if detailed:
        stats = result.get("stats", {})
        lines.append("\nDetailed Structure:")
        lines.append(f"  - Total Keys: {stats.get('total_keys', 0)} (UNet: {stats.get('unet_keys', 0)}, TE: {stats.get('text_encoder_keys', 0)})")
        lines.append(f"  - Unique Modules: {stats.get('unique_base_keys', 0)}")
        if stats.get("block_distribution"):
            dist_str = ", ".join([f"{k.replace('_blocks', '')}: {v}" for k, v in stats["block_distribution"].items()])
            lines.append(f"  - Block Distribution: {dist_str}")

    if result.get("trigger_words"):
        lines.append("\nTrigger Words Found:")
        lines.extend([f"  - {trigger}" for trigger in result["trigger_words"]])
    
    if result.get("issues"):
        lines.append("\nIssues:")
        lines.extend([f"  - {issue}" for issue in result["issues"]])
    
    if result.get("warnings"):
        lines.append("\nWarnings:")
        lines.extend([f"  - {warning}" for warning in result["warnings"]])
        
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Analyze Flux LoRA compatibility with Chroma (v7, for converter v17.0+)")
    parser.add_argument("--lora", help="Path to single LoRA file to analyze")
    parser.add_argument("--scan-dir", help="Directory to scan for LoRA files")
    parser.add_argument("--min-score", type=float, help="Only output LoRAs with a compatibility score at or above this threshold (e.g., 70).")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top results to show when scanning. Ignored if --min-score is set.")
    parser.add_argument("--save-report", help="Save detailed report to a text file")
    parser.add_argument("--json", help="Save detailed report as a JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show more detailed analysis in the console report")
    
    args = parser.parse_args()
    
    if not args.lora and not args.scan_dir:
        parser.error("Either --lora or --scan-dir must be specified")
    
    if args.min_score is not None and not (0 <= args.min_score <= 100):
        parser.error("--min-score must be between 0 and 100")
        
    results = []
    all_results = []

    if args.lora:
        results = [check_single_lora_compatibility(args.lora, verbose=args.detailed)]
        all_results = results
    else:
        all_results = scan_directory(args.scan_dir, args.detailed)
        if args.min_score is not None:
            results = [res for res in all_results if res.get('score', 0) >= args.min_score]
            print(f"\n--- Found {len(results)} of {len(all_results)} LoRAs with score >= {args.min_score}% ---")
        else:
            results = all_results[:args.top_n]
            print(f"\n--- Top {len(results)} LoRA Compatibility Results ---")

    # Output to console
    for result in results:
        print(format_compatibility_report(result, detailed=args.detailed))
        print()
    
    # Save text report if requested
    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            f.write(f"Flux to Chroma LoRA Compatibility Report\n")
            f.write(f"Scanned Directory: {os.path.abspath(args.scan_dir) if args.scan_dir else os.path.abspath(args.lora)}\n")
            f.write(f"Total LoRAs Analyzed: {len(all_results)}\n")
            f.write(f"Displaying {len(results)} results.\n\n")
            for result in results:
                f.write(format_compatibility_report(result, detailed=True))
                f.write("\n\n")
        print(f"Text report saved to: {args.save_report}")
    
    # Save JSON report if requested
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"JSON report saved to: {args.json}")
    
    if not results or not results[0].get("compatible"):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
