#!/usr/bin/env python3
"""
Flux to Chroma LoRA Compatibility Scanner
Analyzes Flux LoRAs to predict conversion success based on empirical findings
"""

import argparse
from safetensors import safe_open
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

def analyze_lora_structure(lora_path: str) -> Dict:
    """Analyze LoRA structure to understand training configuration"""
    
    analysis = {
        "path": str(lora_path),
        "filename": Path(lora_path).name,
        "file_size_mb": Path(lora_path).stat().st_size / (1024 * 1024),
        "total_keys": 0,
        "unet_keys": 0,
        "text_encoder_keys": 0,
        "layer_counts": defaultdict(int),
        "has_double_blocks": False,
        "has_single_blocks": False,
        "modulation_layers": 0,
        "rank": None,
        "training_config": {},
        "metadata": {},
        "warnings": []
    }
    
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            # Get metadata
            if hasattr(f, 'metadata') and f.metadata():
                metadata = f.metadata()
                analysis["metadata"] = metadata
                
                # Extract training configuration
                if "ss_network_args" in metadata:
                    try:
                        import json
                        network_args = json.loads(metadata["ss_network_args"])
                        if "train_blocks" in network_args:
                            analysis["training_config"]["train_blocks"] = network_args["train_blocks"]
                        if "train_double_block_indices" in network_args:
                            analysis["training_config"]["double_block_indices"] = network_args["train_double_block_indices"]
                        if "train_single_block_indices" in network_args:
                            analysis["training_config"]["single_block_indices"] = network_args["train_single_block_indices"]
                    except:
                        pass
                
                # Get rank info
                if "ss_network_dim" in metadata:
                    analysis["rank"] = int(metadata["ss_network_dim"])
            
            # Analyze keys
            keys = list(f.keys())
            analysis["total_keys"] = len(keys)
            
            for key in keys:
                # Count text encoder vs UNet
                if any(te in key for te in ["lora_te", "text_encoder", "text_model"]):
                    analysis["text_encoder_keys"] += 1
                else:
                    analysis["unet_keys"] += 1
                    
                    # Analyze layer types
                    if "double_blocks" in key:
                        analysis["has_double_blocks"] = True
                        analysis["layer_counts"]["double_blocks"] += 1
                        
                        # Check specific components
                        if "img_attn" in key:
                            analysis["layer_counts"]["double_img_attn"] += 1
                        elif "txt_attn" in key:
                            analysis["layer_counts"]["double_txt_attn"] += 1
                        elif "img_mlp" in key:
                            analysis["layer_counts"]["double_img_mlp"] += 1
                        elif "txt_mlp" in key:
                            analysis["layer_counts"]["double_txt_mlp"] += 1
                        elif "img_mod" in key or "txt_mod" in key:
                            analysis["modulation_layers"] += 1
                            
                    elif "single_blocks" in key:
                        analysis["has_single_blocks"] = True
                        analysis["layer_counts"]["single_blocks"] += 1
                        
                        if "linear1" in key:
                            analysis["layer_counts"]["single_linear1"] += 1
                        elif "linear2" in key:
                            analysis["layer_counts"]["single_linear2"] += 1
                        elif "modulation" in key:
                            analysis["modulation_layers"] += 1
                
                # Detect rank from tensor shape if not in metadata
                if analysis["rank"] is None and ".lora_down.weight" in key:
                    tensor = f.get_tensor(key)
                    analysis["rank"] = min(tensor.shape)
                    
    except Exception as e:
        analysis["warnings"].append(f"Error reading file: {str(e)}")
        
    return analysis

def calculate_compatibility_score(analysis: Dict) -> Tuple[float, List[str]]:
    """Calculate compatibility score based on empirical findings"""
    
    score = 100.0
    reasons = []
    
    # Check if it's a text encoder only LoRA
    if analysis["unet_keys"] == 0:
        return 0.0, ["Text encoder only LoRA - cannot convert to Chroma"]
    
    # File size check (empirical: working LoRAs are typically 150-450MB)
    if analysis["file_size_mb"] < 50:
        score -= 20
        reasons.append(f"Very small file size ({analysis['file_size_mb']:.1f}MB) - may be incomplete")
    elif analysis["file_size_mb"] > 1000:
        score -= 50
        reasons.append(f"Very large file size ({analysis['file_size_mb']:.1f}MB) - may not be a LoRA")
    
    # Training configuration analysis
    train_config = analysis["training_config"]
    
    # Best case: trained on both blocks (like working Chroma LoRAs)
    if analysis["has_double_blocks"] and analysis["has_single_blocks"]:
        reasons.append("Trained on both double and single blocks (optimal)")
    
    # Good case: single blocks only (like carriefisher)
    elif analysis["has_single_blocks"] and not analysis["has_double_blocks"]:
        if train_config.get("train_blocks") == "single":
            score -= 10
            reasons.append("Trained on single blocks only (will work but limited)")
        else:
            score -= 15
            reasons.append("Only has single blocks (partial training)")
    
    # Bad case: double blocks only
    elif analysis["has_double_blocks"] and not analysis["has_single_blocks"]:
        score -= 30
        reasons.append("Only has double blocks (incomplete for Chroma)")
    
    # Modulation layers (will be skipped but not a problem)
    if analysis["modulation_layers"] > 0:
        mod_ratio = analysis["modulation_layers"] / analysis["unet_keys"]
        if mod_ratio > 0.3:
            score -= 5
            reasons.append(f"High ratio of modulation layers ({mod_ratio:.1%}) - will be skipped")
    
    # Rank analysis
    if analysis["rank"]:
        if analysis["rank"] < 16:
            score -= 10
            reasons.append(f"Low rank ({analysis['rank']}) - may have limited effect")
        elif analysis["rank"] > 128:
            score -= 5
            reasons.append(f"High rank ({analysis['rank']}) - may be memory intensive")
    
    # Layer count analysis
    expected_double_layers = 152 * 3  # Based on working LoRAs
    expected_single_layers = 76 * 3
    
    if analysis["has_double_blocks"]:
        double_ratio = analysis["layer_counts"]["double_blocks"] / expected_double_layers
        if double_ratio < 0.5:
            score -= 20
            reasons.append(f"Incomplete double blocks ({double_ratio:.1%} of expected)")
    
    if analysis["has_single_blocks"]:
        single_ratio = analysis["layer_counts"]["single_blocks"] / expected_single_layers
        if single_ratio < 0.5:
            score -= 20
            reasons.append(f"Incomplete single blocks ({single_ratio:.1%} of expected)")
    
    # Ensure score doesn't go below 0
    score = max(0, score)
    
    return score, reasons

def format_lora_report(analysis: Dict, score: float, reasons: List[str]) -> str:
    """Format a detailed report for a single LoRA"""
    
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"LoRA: {analysis['filename']}")
    report.append(f"{'='*60}")
    report.append(f"Compatibility Score: {score:.1f}%")
    report.append(f"File Size: {analysis['file_size_mb']:.1f} MB")
    
    if analysis["rank"]:
        report.append(f"Rank: {analysis['rank']}")
    
    report.append(f"\nStructure:")
    report.append(f"  Total keys: {analysis['total_keys']}")
    report.append(f"  UNet keys: {analysis['unet_keys']}")
    report.append(f"  Text encoder keys: {analysis['text_encoder_keys']}")
    
    if analysis["has_double_blocks"]:
        report.append(f"  Double blocks: {analysis['layer_counts']['double_blocks']} keys")
    if analysis["has_single_blocks"]:
        report.append(f"  Single blocks: {analysis['layer_counts']['single_blocks']} keys")
    if analysis["modulation_layers"] > 0:
        report.append(f"  Modulation layers: {analysis['modulation_layers']} (will be skipped)")
    
    if analysis["training_config"]:
        report.append(f"\nTraining Configuration:")
        for key, value in analysis["training_config"].items():
            report.append(f"  {key}: {value}")
    
    report.append(f"\nCompatibility Analysis:")
    for reason in reasons:
        report.append(f"  • {reason}")
    
    if analysis["warnings"]:
        report.append(f"\nWarnings:")
        for warning in analysis["warnings"]:
            report.append(f"  ⚠ {warning}")
    
    # Recommendation
    report.append(f"\nRecommendation:")
    if score >= 90:
        report.append("  ✅ Excellent candidate for conversion")
    elif score >= 70:
        report.append("  ✓ Good candidate - should convert successfully")
    elif score >= 50:
        report.append("  ⚡ Fair candidate - may have limitations")
    else:
        report.append("  ❌ Poor candidate - conversion not recommended")
    
    return "\n".join(report)

def scan_directory(directory: str, top_n: int = 10, min_score: float = 50.0) -> List[Tuple[Dict, float, List[str]]]:
    """Scan directory for LoRA files and analyze compatibility"""
    
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Find all safetensors files
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
            score, reasons = calculate_compatibility_score(analysis)
            
            if score >= min_score:
                results.append((analysis, score, reasons))
                
        except Exception as e:
            print(f"Error analyzing {lora_file.name}: {e}")
            if args.debug:
                traceback.print_exc()
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def print_summary(results: List[Tuple[Dict, float, List[str]]], top_n: int = 10):
    """Print summary of scan results"""
    
    print(f"\n{'='*80}")
    print("COMPATIBILITY SCAN SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTotal compatible LoRAs found: {len(results)}")
    
    # Group by score ranges
    excellent = sum(1 for _, score, _ in results if score >= 90)
    good = sum(1 for _, score, _ in results if 70 <= score < 90)
    fair = sum(1 for _, score, _ in results if 50 <= score < 70)
    
    print(f"\nScore Distribution:")
    print(f"  Excellent (90-100%): {excellent}")
    print(f"  Good (70-89%): {good}")
    print(f"  Fair (50-69%): {fair}")
    
    # Show top results
    print(f"\nTop {min(top_n, len(results))} LoRAs for Chroma conversion:")
    print(f"{'Rank':<6} {'Score':<8} {'Rank':<6} {'Blocks':<20} {'Filename'}")
    print("-" * 80)
    
    for i, (analysis, score, _) in enumerate(results[:top_n], 1):
        blocks = []
        if analysis["has_double_blocks"]:
            blocks.append("double")
        if analysis["has_single_blocks"]:
            blocks.append("single")
        blocks_str = "+".join(blocks) if blocks else "none"
        
        rank_str = str(analysis["rank"]) if analysis["rank"] else "?"
        
        print(f"{i:<6} {score:>6.1f}%  {rank_str:<6} {blocks_str:<20} {analysis['filename']}")

def main():
    global args
    
    parser = argparse.ArgumentParser(description="Scan Flux LoRAs for Chroma compatibility")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lora", type=str, help="Single LoRA file to analyze")
    group.add_argument("--scan-dir", type=str, help="Directory to scan for LoRA files")
    
    # Output options
    parser.add_argument("--save-report", type=str, help="Save detailed report to file")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--min-score", type=float, default=50.0, help="Minimum score to include")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    args = parser.parse_args()
    
    if args.lora:
        # Single LoRA analysis
        try:
            analysis = analyze_lora_structure(args.lora)
            score, reasons = calculate_compatibility_score(analysis)
            
            report = format_lora_report(analysis, score, reasons)
            print(report)
            
            if args.save_report:
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\nReport saved to: {args.save_report}")
                
        except Exception as e:
            print(f"Error analyzing LoRA: {e}")
            if args.debug:
                traceback.print_exc()
    
    else:
        # Directory scan
        results = scan_directory(args.scan_dir, args.top_n, args.min_score)
        
        if results:
            print_summary(results, args.top_n)
            
            if args.save_report:
                # Create detailed report
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    f.write(f"Flux to Chroma Compatibility Scan Report\n")
                    f.write(f"Directory: {args.scan_dir}\n")
                    f.write(f"Total LoRAs analyzed: {len(results)}\n")
                    f.write("="*80 + "\n")
                    
                    for analysis, score, reasons in results:
                        report = format_lora_report(analysis, score, reasons)
                        f.write(report + "\n")
                
                print(f"\nDetailed report saved to: {args.save_report}")
        else:
            print("No compatible LoRA files found.")

if __name__ == "__main__":
    main()