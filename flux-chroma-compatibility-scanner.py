#!/usr/bin/env python3
"""
Flux to Chroma LoRA Compatibility Scanner v3
Comprehensive analysis based on complete understanding of architecture
"""

import argparse
from safetensors import safe_open
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import traceback
import re

def normalize_lora_key_for_analysis(key: str) -> Tuple[str, str]:
    """
    Normalize LoRA key and determine what it targets
    Returns (normalized_key, target_type)
    """
    # Remove prefixes
    normalized = key
    for prefix in ["lora_unet_", "lora.", "transformer.", "model.", "diffusion_model.", "unet."]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    
    # Remove LoRA suffixes
    for suffix in [".lora_A.weight", ".lora_B.weight", ".lora_up.weight", ".lora_down.weight", 
                   ".alpha", ".weight", ".bias"]:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    
    # Determine target type
    target_type = "unknown"
    
    # Check for known non-existent layers
    if "proj_out" in normalized and "single" in normalized:
        target_type = "nonexistent_single_proj_out"
    elif "norm.linear" in normalized and "single" in normalized:
        target_type = "nonexistent_norm_linear"
    # Modulation layers
    elif any(mod in normalized.lower() for mod in ["modulation", "mod_lin", "_mod", ".mod"]):
        target_type = "modulation"
    # Single block patterns
    elif "single" in normalized:
        if any(x in normalized for x in ["to_q", "to_k", "to_v"]):
            target_type = "single_attention_qkv"
        elif "to_out" in normalized or "proj_mlp" in normalized:
            target_type = "single_output"
        elif "ff.net" in normalized:
            target_type = "single_ff"
        elif "linear1" in normalized:
            target_type = "single_linear1"
        elif "linear2" in normalized:
            target_type = "single_linear2"
    # Double block patterns
    elif "double" in normalized or ("transformer_blocks" in normalized and "single" not in normalized):
        if "img_attn" in normalized or ("attn" in normalized and "context" not in normalized):
            if any(x in normalized for x in ["to_q", "to_k", "to_v"]):
                target_type = "double_img_attention_qkv"
            else:
                target_type = "double_img_attention"
        elif "txt_attn" in normalized or "attn_context" in normalized:
            if any(x in normalized for x in ["to_q", "to_k", "to_v"]):
                target_type = "double_txt_attention_qkv"
            else:
                target_type = "double_txt_attention"
        elif "img_mlp" in normalized:
            target_type = "double_img_mlp"
        elif "txt_mlp" in normalized or "ff_context" in normalized:
            target_type = "double_txt_mlp"
    
    return normalized, target_type

def detect_naming_style(keys: List[str]) -> str:
    """Detect the LoRA naming style with more precision"""
    style_votes = defaultdict(int)
    
    for key in keys[:50]:  # Check more keys
        if "transformer.single_transformer_blocks" in key:
            style_votes["diffusers"] += 1
        elif "transformer.transformer_blocks" in key and "single" not in key:
            style_votes["diffusers"] += 1
        elif "_blocks_" in key and ("lora_unet_" in key or key.count("_") > 3):
            style_votes["kohya"] += 1
        elif "double_blocks." in key or "single_blocks." in key:
            style_votes["standard"] += 1
    
    if style_votes:
        return max(style_votes.items(), key=lambda x: x[1])[0]
    return "unknown"

def analyze_lora_structure(lora_path: str) -> Dict:
    """Comprehensive LoRA structure analysis"""
    
    analysis = {
        "path": str(lora_path),
        "filename": Path(lora_path).name,
        "file_size_mb": Path(lora_path).stat().st_size / (1024 * 1024),
        "total_keys": 0,
        "unet_keys": 0,
        "text_encoder_keys": 0,
        "naming_style": "unknown",
        "layer_distribution": defaultdict(int),
        "target_types": defaultdict(int),
        "issues": [],
        "warnings": [],
        "rank": None,
        "metadata": {},
        "trigger_words": None,
        "convertible_stats": {
            "total_pairs": 0,
            "convertible": 0,
            "will_accumulate": 0,
            "will_skip_modulation": 0,
            "will_skip_nonexistent": 0
        }
    }
    
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            # Get metadata
            if hasattr(f, 'metadata') and f.metadata():
                metadata = f.metadata()
                analysis["metadata"] = metadata
                
                # Look for trigger words
                trigger_keys = ["trigger_words", "trigger_word", "instance_prompt", 
                               "ss_instance_prompt", "prompt", "activation_text", 
                               "modelspec.trigger_phrase"]
                for key in trigger_keys:
                    if key in metadata and metadata[key]:
                        analysis["trigger_words"] = str(metadata[key])
                        break
                
                # Get rank
                if "ss_network_dim" in metadata:
                    analysis["rank"] = int(metadata["ss_network_dim"])
            
            # Analyze keys
            keys = list(f.keys())
            analysis["total_keys"] = len(keys)
            
            # Detect naming style
            analysis["naming_style"] = detect_naming_style(keys)
            
            # Group keys by base layer
            base_layers = defaultdict(list)
            ranks_found = []
            
            for key in keys:
                # Skip text encoder
                if any(te in key for te in ["lora_te", "text_encoder", "text_model"]):
                    analysis["text_encoder_keys"] += 1
                    continue
                
                analysis["unet_keys"] += 1
                
                # Extract base layer name
                base_key = key
                for suffix in [".lora_A.weight", ".lora_B.weight", ".lora_up.weight", 
                             ".lora_down.weight", ".alpha"]:
                    if base_key.endswith(suffix):
                        base_key = base_key[:-len(suffix)]
                        # Check rank
                        if ("lora_down" in key or "lora_A" in key) and analysis["rank"] is None:
                            tensor = f.get_tensor(key)
                            ranks_found.append(min(tensor.shape))
                        break
                
                base_layers[base_key].append(key)
                
                # Analyze layer type
                normalized, target_type = normalize_lora_key_for_analysis(key)
                analysis["target_types"][target_type] += 1
                
                # Track layer distribution
                if "double_blocks" in normalized or "double_blocks" in key:
                    analysis["layer_distribution"]["double_blocks"] += 1
                elif "single_blocks" in normalized or "single_transformer_blocks" in normalized:
                    analysis["layer_distribution"]["single_blocks"] += 1
                else:
                    analysis["layer_distribution"]["other"] += 1
            
            # Auto-detect rank from most common
            if ranks_found and analysis["rank"] is None:
                from collections import Counter
                rank_counts = Counter(ranks_found)
                analysis["rank"] = rank_counts.most_common(1)[0][0]
            
            # Count complete pairs and analyze convertibility
            for base_key, related_keys in base_layers.items():
                has_down = any("down" in k or "lora_A" in k for k in related_keys)
                has_up = any("up" in k or "lora_B" in k for k in related_keys)
                
                if has_down and has_up:
                    analysis["convertible_stats"]["total_pairs"] += 1
                    
                    normalized, target_type = normalize_lora_key_for_analysis(base_key)
                    
                    # Check if it's modulation
                    if target_type == "modulation":
                        analysis["convertible_stats"]["will_skip_modulation"] += 1
                    # Check if it's non-existent
                    elif target_type in ["nonexistent_single_proj_out", "nonexistent_norm_linear"]:
                        analysis["convertible_stats"]["will_skip_nonexistent"] += 1
                    # Check if it will accumulate
                    elif target_type in ["single_attention_qkv", "double_img_attention_qkv", 
                                       "double_txt_attention_qkv"]:
                        analysis["convertible_stats"]["will_accumulate"] += 1
                        analysis["convertible_stats"]["convertible"] += 1
                    else:
                        analysis["convertible_stats"]["convertible"] += 1
            
            # Identify issues
            if analysis["unet_keys"] == 0:
                analysis["issues"].append("No UNet keys found - text encoder only LoRA")
            
            if analysis["convertible_stats"]["convertible"] == 0:
                analysis["issues"].append("No convertible layer pairs found")
            
            # Add warnings
            if analysis["convertible_stats"]["will_skip_nonexistent"] > 0:
                pct = (analysis["convertible_stats"]["will_skip_nonexistent"] / 
                      analysis["convertible_stats"]["total_pairs"] * 100)
                analysis["warnings"].append(f"{pct:.1f}% of layers target non-existent Flux layers")
            
            if analysis["convertible_stats"]["will_skip_modulation"] > 0:
                pct = (analysis["convertible_stats"]["will_skip_modulation"] / 
                      analysis["convertible_stats"]["total_pairs"] * 100)
                analysis["warnings"].append(f"{pct:.1f}% of layers are modulation (will be skipped)")
            
    except Exception as e:
        analysis["issues"].append(f"Error reading file: {str(e)}")
        
    return analysis

def calculate_quality_score(analysis: Dict) -> Tuple[float, List[str]]:
    """Calculate conversion quality score based on comprehensive analysis"""
    
    score = 100.0
    factors = []
    
    # Check if UNet LoRA
    if analysis["unet_keys"] == 0:
        return 0.0, ["Text encoder only LoRA - cannot convert to Chroma"]
    
    # Check convertible layers
    if analysis["convertible_stats"]["convertible"] == 0:
        return 0.0, ["No convertible layer pairs found"]
    
    # Calculate conversion coverage
    total_pairs = analysis["convertible_stats"]["total_pairs"]
    convertible = analysis["convertible_stats"]["convertible"]
    if total_pairs > 0:
        coverage = convertible / total_pairs
        if coverage < 0.5:
            score -= 30
            factors.append(f"Low conversion coverage ({coverage:.1%})")
        elif coverage < 0.8:
            score -= 15
            factors.append(f"Moderate conversion coverage ({coverage:.1%})")
        else:
            factors.append(f"High conversion coverage ({coverage:.1%})")
    
    # File size analysis
    if analysis["file_size_mb"] < 5:
        score -= 20
        factors.append(f"Very small file ({analysis['file_size_mb']:.1f}MB) - limited effect")
    elif analysis["file_size_mb"] > 500:
        score -= 10
        factors.append(f"Large file ({analysis['file_size_mb']:.1f}MB) - may be slow to convert")
    
    # Naming style
    if analysis["naming_style"] == "standard":
        factors.append("Standard naming (optimal)")
    elif analysis["naming_style"] in ["diffusers", "kohya"]:
        factors.append(f"{analysis['naming_style'].capitalize()} naming (fully supported)")
    else:
        score -= 10
        factors.append("Unknown naming format")
    
    # Layer distribution
    has_double = analysis["layer_distribution"]["double_blocks"] > 0
    has_single = analysis["layer_distribution"]["single_blocks"] > 0
    
    if has_double and has_single:
        factors.append("Full model training (double + single blocks)")
    elif has_single and not has_double:
        score -= 10
        factors.append("Single blocks only (partial training)")
    elif has_double and not has_single:
        score -= 15
        factors.append("Double blocks only (limited training)")
    
    # Accumulation
    if analysis["convertible_stats"]["will_accumulate"] > 0:
        factors.append(f"{analysis['convertible_stats']['will_accumulate']} layers will use smart accumulation")
    
    # Non-existent layers
    if analysis["convertible_stats"]["will_skip_nonexistent"] > 10:
        score -= 20
        factors.append(f"{analysis['convertible_stats']['will_skip_nonexistent']} layers target non-existent architecture")
    
    # Rank analysis
    if analysis["rank"]:
        if analysis["rank"] < 4:
            score -= 20
            factors.append(f"Very low rank ({analysis['rank']}) - minimal effect")
        elif analysis["rank"] > 256:
            score -= 10
            factors.append(f"Very high rank ({analysis['rank']}) - may be slow")
        else:
            factors.append(f"Rank {analysis['rank']}")
    
    # Trigger words
    if analysis["trigger_words"]:
        factors.append(f"Has trigger: '{analysis['trigger_words'][:50]}'")
    
    score = max(0, min(100, score))
    
    return score, factors

def format_analysis_report(analysis: Dict, score: float, factors: List[str]) -> str:
    """Format a comprehensive analysis report"""
    
    report = []
    report.append(f"\n{'='*70}")
    report.append(f"LoRA: {analysis['filename']}")
    report.append(f"{'='*70}")
    report.append(f"Conversion Quality Score: {score:.0f}%")
    report.append(f"File Size: {analysis['file_size_mb']:.1f} MB")
    report.append(f"Naming Style: {analysis['naming_style']}")
    
    if analysis["rank"]:
        report.append(f"Rank: {analysis['rank']}")
    
    report.append(f"\nStructure Analysis:")
    report.append(f"  Total keys: {analysis['total_keys']}")
    report.append(f"  UNet keys: {analysis['unet_keys']}")
    report.append(f"  Text encoder keys: {analysis['text_encoder_keys']}")
    
    report.append(f"\nConversion Statistics:")
    stats = analysis["convertible_stats"]
    report.append(f"  Total LoRA pairs: {stats['total_pairs']}")
    report.append(f"  Convertible pairs: {stats['convertible']} ({stats['convertible']/stats['total_pairs']*100:.1f}%)" if stats['total_pairs'] > 0 else "  Convertible pairs: 0")
    report.append(f"  Will accumulate: {stats['will_accumulate']}")
    report.append(f"  Will skip (modulation): {stats['will_skip_modulation']}")
    report.append(f"  Will skip (non-existent): {stats['will_skip_nonexistent']}")
    
    # Layer distribution
    if analysis["layer_distribution"]:
        report.append(f"\nLayer Distribution:")
        for layer_type, count in analysis["layer_distribution"].items():
            if count > 0:
                report.append(f"  {layer_type}: {count} keys")
    
    # Target types breakdown
    if len(analysis["target_types"]) > 1:
        report.append(f"\nTarget Layer Types:")
        sorted_types = sorted(analysis["target_types"].items(), key=lambda x: -x[1])
        for target_type, count in sorted_types[:10]:
            report.append(f"  {target_type}: {count}")
    
    if analysis["trigger_words"]:
        report.append(f"\nTrigger Words: {analysis['trigger_words']}")
    
    report.append(f"\nQuality Factors:")
    for factor in factors:
        report.append(f"  • {factor}")
    
    if analysis["warnings"]:
        report.append(f"\nWarnings:")
        for warning in analysis["warnings"]:
            report.append(f"  ⚠ {warning}")
    
    if analysis["issues"]:
        report.append(f"\nIssues:")
        for issue in analysis["issues"]:
            report.append(f"  ❌ {issue}")
    
    # Recommendation
    report.append(f"\nRecommendation:")
    if score >= 90:
        report.append("  ✅ Excellent for conversion - minimal information loss")
    elif score >= 70:
        report.append("  ✓ Good for conversion - should work well")
    elif score >= 50:
        report.append("  ⚡ Fair for conversion - will work with some limitations")
    else:
        report.append("  ❌ Poor for conversion - significant limitations expected")
    
    return "\n".join(report)

def scan_directory(directory: str, min_score: float = 0.0) -> List[Tuple[Dict, float, List[str]]]:
    """Scan directory for LoRA files and analyze them"""
    
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    lora_files = list(directory_path.glob("*.safetensors"))
    
    if not lora_files:
        print(f"No .safetensors files found in {directory}")
        return []
    
    print(f"\nFound {len(lora_files)} .safetensors files")
    print("Analyzing compatibility...\n")
    
    results = []
    
    for lora_file in lora_files:
        try:
            # Skip very large files (likely base models)
            if lora_file.stat().st_size > 2 * 1024**3:  # 2GB
                continue
                
            analysis = analyze_lora_structure(str(lora_file))
            score, factors = calculate_quality_score(analysis)
            
            if score >= min_score:
                results.append((analysis, score, factors))
                
        except Exception as e:
            print(f"Error analyzing {lora_file.name}: {e}")
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def print_summary(results: List[Tuple[Dict, float, List[str]]], top_n: int = 10):
    """Print summary of scan results"""
    
    print(f"\n{'='*80}")
    print("COMPATIBILITY SCAN SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTotal LoRAs analyzed: {len(results)}")
    
    # Group by score ranges
    excellent = sum(1 for _, score, _ in results if score >= 90)
    good = sum(1 for _, score, _ in results if 70 <= score < 90)
    fair = sum(1 for _, score, _ in results if 50 <= score < 70)
    poor = sum(1 for _, score, _ in results if score < 50)
    
    print(f"\nScore Distribution:")
    print(f"  Excellent (90-100%): {excellent}")
    print(f"  Good (70-89%): {good}")
    print(f"  Fair (50-69%): {fair}")
    print(f"  Poor (0-49%): {poor}")
    
    # Show top results
    print(f"\nTop {min(top_n, len(results))} LoRAs for Chroma conversion:")
    print(f"{'Score':<8} {'Rank':<6} {'Style':<10} {'Coverage':<10} {'Size':<8} {'Filename'}")
    print("-" * 80)
    
    for analysis, score, _ in results[:top_n]:
        rank_str = str(analysis["rank"]) if analysis["rank"] else "?"
        style = analysis["naming_style"][:10]
        stats = analysis["convertible_stats"]
        coverage = f"{stats['convertible']}/{stats['total_pairs']}" if stats['total_pairs'] > 0 else "0/0"
        size = f"{analysis['file_size_mb']:.1f}MB"
        
        print(f"{score:>6.0f}%  {rank_str:<6} {style:<10} {coverage:<10} {size:<8} {analysis['filename'][:35]}")

def main():
    parser = argparse.ArgumentParser(description="Scan Flux LoRAs for Chroma compatibility v3")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lora", type=str, help="Single LoRA file to analyze")
    group.add_argument("--scan-dir", type=str, help="Directory to scan for LoRA files")
    
    parser.add_argument("--save-report", type=str, help="Save detailed report to file")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score to include")
    parser.add_argument("--json", action="store_true", help="Output analysis as JSON")
    
    args = parser.parse_args()
    
    if args.lora:
        # Single LoRA analysis
        try:
            analysis = analyze_lora_structure(args.lora)
            score, factors = calculate_quality_score(analysis)
            
            if args.json:
                output = {
                    "analysis": analysis,
                    "score": score,
                    "factors": factors
                }
                # Convert defaultdicts to regular dicts for JSON
                output["analysis"]["layer_distribution"] = dict(output["analysis"]["layer_distribution"])
                output["analysis"]["target_types"] = dict(output["analysis"]["target_types"])
                print(json.dumps(output, indent=2))
            else:
                report = format_analysis_report(analysis, score, factors)
                print(report)
            
            if args.save_report and not args.json:
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\nReport saved to: {args.save_report}")
                
        except Exception as e:
            print(f"Error analyzing LoRA: {e}")
            traceback.print_exc()
    
    else:
        # Directory scan
        results = scan_directory(args.scan_dir, args.min_score)
        
        if results:
            print_summary(results, args.top_n)
            
            if args.save_report:
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    f.write(f"Flux to Chroma Compatibility Scan Report v3\n")
                    f.write(f"Directory: {args.scan_dir}\n")
                    f.write(f"Total LoRAs analyzed: {len(results)}\n")
                    f.write("="*80 + "\n")
                    
                    for analysis, score, factors in results:
                        report = format_analysis_report(analysis, score, factors)
                        f.write(report + "\n")
                
                print(f"\nDetailed report saved to: {args.save_report}")
        else:
            print("No LoRA files found matching criteria.")

if __name__ == "__main__":
    main()
