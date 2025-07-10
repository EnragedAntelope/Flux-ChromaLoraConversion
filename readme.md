# Flux to Chroma LoRA Converter

Convert Flux Dev LoRAs to work with the Chroma model, enabling you to use your favorite Flux LoRAs with this exciting new model variant.

## 🎯 Overview

Chroma is a heavily modified variant based on Flux Schnell with fewer parameters (8.9B vs 12B). This tool converts Flux Dev LoRAs to be compatible with Chroma using an advanced add-difference merge technique.

### Key Features
- ✅ Converts Flux Dev LoRAs to Chroma-compatible format
- ✅ Preserves trigger words and metadata
- ✅ Memory-efficient chunk processing
- ✅ Automatic rank detection
- ✅ Multiple conversion quality modes
- ✅ Compatibility pre-scanning

## 📋 Requirements

- Python 3.8+
- PyTorch with CUDA support (or CPU mode)
- 16GB+ RAM recommended (24GB+ for GPU mode)
- Required packages:
  ```bash
  pip install torch safetensors numpy tqdm
  ```

## 🚀 Quick Start

### Basic Conversion
```bash
python flux-chroma-converter.py \
    --flux-base "path/to/flux1-dev.safetensors" \
    --flux-lora "path/to/your-flux-lora.safetensors" \
    --chroma-base "path/to/chroma-model.safetensors" \
    --output-lora "your-chroma-lora.safetensors"
```

### Windows Example
```cmd
python flux-chroma-converter.py ^
    --flux-base "D:\models\flux1-dev.safetensors" ^
    --flux-lora "D:\loras\my-character.safetensors" ^
    --chroma-base "D:\models\chroma-unlocked.safetensors" ^
    --output-lora "my-character-chroma.safetensors"
```

## 🛠️ Main Converter Options

### Required Arguments
- `--flux-base`: Path to Flux Dev base model (12B parameters)
- `--flux-lora`: Path to Flux Dev LoRA to convert
- `--chroma-base`: Path to Chroma base model
- `--output-lora`: Output path for converted Chroma LoRA

### Optional Arguments
- `--device`: Computation device (`cuda` or `cpu`, default: cuda)
- `--rank`: LoRA rank for extraction (default: -1 for auto-detect)
- `--lora-alpha`: LoRA merge strength (default: 1.0)
- `--chunk-size`: Memory chunk size (default: 50, reduce if OOM)
- `--mode`: Conversion mode - `standard`, `add_dissimilar`, or `comparative`
- `--similarity-threshold`: Threshold for add_dissimilar mode (0.0-1.0)
- `--debug`: Enable detailed debug output
- `--verify-only`: Test compatibility without conversion
- `--inspect-output`: Inspect a converted LoRA file

### Memory Management
If you encounter out-of-memory errors:
```bash
# Use CPU mode (slower but uses system RAM)
python flux-chroma-converter.py ... --device cpu

# Reduce chunk size
python flux-chroma-converter.py ... --chunk-size 10

# Use system with more RAM/VRAM
```

## 🔍 Compatibility Scanner

Check if your Flux LoRAs are suitable for conversion before processing:

### Scan Single LoRA
```bash
python flux-chroma-compatibility-scanner.py \
    --lora "path/to/flux-lora.safetensors"
```

### Scan Entire Directory
```bash
python flux-chroma-compatibility-scanner.py \
    --scan-dir "D:\models\loras\flux" \
    --top-n 20 \
    --save-report "compatibility_report.txt"
```

### Scanner Options
- `--lora`: Single LoRA file to analyze
- `--scan-dir`: Directory to scan for LoRAs
- `--top-n`: Number of top results to show (default: 10)
- `--min-score`: Minimum compatibility score (default: 50.0)
- `--save-report`: Save detailed report to file
- `--debug`: Show error details

### Compatibility Scoring
- **90-100%**: Excellent - Full conversion expected
- **70-89%**: Good - Should work with minor limitations
- **50-69%**: Fair - Partial compatibility, some features missing
- **Below 50%**: Poor - Not recommended for conversion

## 📊 Additional Tools

### LoRA Inspector
Examine the structure of any LoRA file:
```bash
python inspect-chroma-lora.py "path/to/lora.safetensors"
```

### LoRA Structure Analyzer
Compare working Chroma LoRAs with converted ones:
```bash
python analyze_working_chroma_lora.py \
    --working-lora "known-good-chroma-lora.safetensors" \
    --converted-lora "your-converted-lora.safetensors"
```

### Diagnostic Tool
Diagnose issues with converted LoRAs:
```bash
python diagnose-chroma-lora.py \
    --chroma-lora "converted-lora.safetensors" \
    --chroma-base "chroma-model.safetensors" \
    --flux-lora "original-flux-lora.safetensors"
```

## 💡 Understanding the Conversion Process

### How It Works
1. **Merge**: LoRA weights are merged into Flux Dev base model
2. **Difference**: Computes the difference between merged and original models
3. **Apply**: Transfers differences to Chroma base model
4. **Extract**: Extracts new LoRA from the modified Chroma model

### What Gets Converted
- ✅ Attention layers (img_attn, txt_attn)
- ✅ MLP layers (img_mlp, txt_mlp)
- ✅ Linear layers in single blocks
- ✅ Trigger words and metadata
- ❌ Modulation layers (not used by Chroma)
- ❌ Text encoder weights

### Expected Results
- Converted LoRAs typically work best at strength 1.0-1.5
- Character/style LoRAs maintain good likeness
- Some fine details may differ from original Flux output
- Single-block-only LoRAs (76 layers) work but with limitations
- Full LoRAs (231 layers) provide best results

## 🐛 Troubleshooting

### Common Issues

**Out of Memory**
```
Error: CUDA out of memory
Solution: Use --device cpu or --chunk-size 10
```

**No Effect on Output**
```
Issue: LoRA loads but doesn't affect image
Check: If compatibility scanner indicated high compatibility. Also check ComfyUI/other tool error messages during lora load.
```

**Missing Layers Warning**
```
Warning: lora key not loaded: modulation_lin
This is normal - Chroma doesn't use modulation layers
```

### Debugging Steps
1. Run compatibility scanner first
2. Use `--debug` flag for detailed output
3. Inspect converted LoRA structure
4. Test with known working LoRAs
5. Try different conversion modes

## 📈 Best Practices

1. **Pre-scan LoRAs**: Use compatibility scanner before conversion
2. **Start with Standard Mode**: Try other modes only if needed
3. **Test at Multiple Strengths**: Converted LoRAs may need 1.0-1.5x strength
4. **Keep Original Files**: Always preserve original Flux LoRAs
5. **Monitor Memory**: Close other applications during conversion
6. **Batch Processing**: Convert multiple LoRAs sequentially, not in parallel

## 🔬 Technical Details

### Architecture Differences
- **Flux Dev**: 12B parameters with modulation layers
- **Chroma**: 8.9B parameters, based on Flux Schnell architecture
- **Missing**: 3.3B modulation layer parameters

### LoRA Structure
- Flux format: Various naming conventions
- Chroma format: `lora_unet_` prefix with underscores
- Shape orientation: Critical for proper matrix multiplication

### Supported Training Configurations
- ✅ Full training (double + single blocks)
- ✅ Single blocks only
- ⚠️ Double blocks only (limited compatibility)
- ❌ Text encoder only

## 📝 License

This tool is provided as-is for research and personal use. Please respect the licenses of the models and LoRAs you convert.

## Help Improve This!

PRs are welcome if you identify any bugs or potential enhancements. 
