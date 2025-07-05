# Flux to Chroma LoRA Converter

**⚠️ BETA SOFTWARE - Use with realistic expectations**

A Python script that attempts to convert Flux Dev LoRAs to work with the Chroma model. This tool performs architectural mapping and difference extraction to adapt LoRAs between these different model architectures.

## ⚠️ Important Limitations

**Please read this section carefully before using the tool.**

### Architectural Reality
- **Flux Dev** (12B parameters) vs **Chroma** (8.9B parameters, based on Flux Schnell)
- **Expected success rate: ~15-25%** of LoRA layers will successfully convert
- **Fundamental incompatibilities exist** due to different architectures:
  - Flux has separate Q/K/V projections, Chroma uses combined QKV matrices  
  - Different transformer block structures
  - Many Flux layers simply don't exist in Chroma
  - Parameter count mismatch (25% fewer parameters in Chroma)

### What This Means
- ✅ **Some effect may be preserved** - basic style/character elements might transfer
- ❌ **Full fidelity is impossible** - expect weaker results than original LoRA
- ❌ **Complex LoRAs may not work** - intricate style adjustments likely won't survive conversion
- ⚠️ **Results vary significantly** by LoRA type and training approach

## Better Alternatives

1. **🎯 Train directly on Chroma** - Best results, full compatibility
2. **🔄 Use Flux Schnell LoRAs** - Higher compatibility (Chroma is Schnell-based)
3. **📚 Retrain with Chroma base model** - Time investment but guaranteed compatibility

## Installation

### Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install safetensors numpy tqdm
```

### System Requirements
- **GPU**: 8GB+ VRAM recommended (can run on CPU but very slow)
- **RAM**: 16GB+ system RAM
- **Storage**: ~20GB free space for model files and intermediates

## Usage

### Basic Conversion

```bash
python flux-chroma-converter-enhanced.py \
  --flux-model /path/to/flux_dev.safetensors \
  --chroma-model /path/to/chroma.safetensors \
  --lora /path/to/your_flux_lora.safetensors \
  --output /path/to/output_chroma_lora.safetensors
```

### Enhanced Conversion (Recommended)

```bash
python flux-chroma-converter-enhanced.py \
  --flux-model /path/to/flux_dev.safetensors \
  --chroma-model /path/to/chroma.safetensors \
  --lora /path/to/your_flux_lora.safetensors \
  --output ./converted_loras/ \
  --amplify 1.5 \
  --aggressive-mapping \
  --rank 64 \
  --merge-strength 1.0 \
  --save-intermediate \
  --debug-keys
```

### Weak LoRA Recovery

For LoRAs that barely transfer:
```bash
python flux-chroma-converter-enhanced.py \
  --flux-model /path/to/flux_dev.safetensors \
  --chroma-model /path/to/chroma.safetensors \
  --lora /path/to/weak_lora.safetensors \
  --output ./recovered_lora.safetensors \
  --amplify 2.0 \
  --force-extract-all \
  --aggressive-mapping \
  --threshold 1e-8
```

## Command Line Options

### Required Parameters
- `--flux-model`: Path to Flux Dev UNet model file
- `--chroma-model`: Path to Chroma UNet model file  
- `--lora`: Path to Flux LoRA to convert
- `--output`: Output path (file or directory)

### Core Settings
- `--device cuda|cpu`: Processing device (default: cuda)
- `--dtype float32|float16|bfloat16`: Computation precision (default: bfloat16)
- `--rank N`: Output LoRA rank for extraction (default: 64, try 32-128)
  - Higher = more detail, larger file; Lower = cleaner, smaller file
  - Input LoRA rank is auto-detected during merging
- `--merge-strength 0.0-2.0`: Merge intensity (default: 1.0)

### Enhancement Options
- `--amplify 1.0-3.0`: Amplify weak LoRA effects (default: 1.0, try 1.5-2.0)
- `--aggressive-mapping`: Try more mapping strategies
- `--force-extract-all`: Extract even tiny differences
- `--threshold 1e-6`: Difference threshold (lower = more sensitive)

### Debugging & Storage
- `--debug-keys`: Show detailed key mapping information
- `--save-intermediate`: Save intermediate conversion steps
- `--intermediate-dir ./path`: Where to save intermediates
- `--chunk-size N`: Memory management (default: 100)

## Example Workflows

### 1. Quick Test Conversion
```bash
# Basic conversion to see if LoRA has any compatibility
python flux-chroma-converter-enhanced.py \
  --flux-model flux_dev.safetensors \
  --chroma-model chroma.safetensors \
  --lora character_lora.safetensors \
  --output test_conversion.safetensors \
  --debug-keys
```

### 2. Character LoRA Conversion  
```bash
# Character LoRAs often work better with amplification
python flux-chroma-converter-enhanced.py \
  --flux-model flux_dev.safetensors \
  --chroma-model chroma.safetensors \
  --lora anime_character.safetensors \
  --output converted_character.safetensors \
  --amplify 1.5 \
  --aggressive-mapping \
  --rank 64
```

### 3. Style LoRA Conversion
```bash
# Style LoRAs may need higher amplification
python flux-chroma-converter-enhanced.py \
  --flux-model flux_dev.safetensors \
  --chroma-model chroma.safetensors \
  --lora art_style.safetensors \
  --output converted_style.safetensors \
  --amplify 2.0 \
  --force-extract-all \
  --aggressive-mapping \
  --threshold 1e-7
```

### 4. Batch Processing Directory
```bash
# Process multiple LoRAs to a directory
for lora in /path/to/flux_loras/*.safetensors; do
    python flux-chroma-converter-enhanced.py \
      --flux-model flux_dev.safetensors \
      --chroma-model chroma.safetensors \
      --lora "$lora" \
      --output ./converted_loras/ \
      --amplify 1.5 \
      --aggressive-mapping
done
```

## Understanding the Output

### Console Messages
```
✓ Found Flux model: flux_dev.safetensors
✓ Found Chroma model: chroma.safetensors  
✓ Found LoRA file: my_lora.safetensors

Analyzing LoRA structure:
Total LoRA tensors: 144
Unique LoRA layers: 48

LoRA Application Summary:
  Total LoRA layers: 48
  QKV applications: 8
  Regular applications: 12
  Total applications: 20
  Application rate: 41.7%   ← Your success rate

Extracted LoRA for 15 layers (5 below threshold)
```

### Success Indicators
- ✅ **Application rate >20%**: Good conversion candidate
- ⚠️ **Application rate 10-20%**: Partial conversion, may still be useful
- ❌ **Application rate <10%**: Likely incompatible LoRA

## Troubleshooting

### Low Application Rate (<15%)
```bash
# Try more aggressive settings
--amplify 2.0 --aggressive-mapping --force-extract-all --threshold 1e-8
```

### Out of Memory Errors
```bash
# Reduce memory usage
--chunk-size 50 --dtype float16 --device cpu
```

### No LoRA Layers Extracted
```bash
# Lower the extraction threshold
--threshold 1e-8 --force-extract-all
```

### Weak Effects in Generated Images
```bash
# Increase amplification
--amplify 2.0
# or try higher merge strength
--merge-strength 1.5
```

## File Structure

When using `--save-intermediate`, you'll get:
```
intermediate/
├── flux_with_lora.safetensors     # Flux + original LoRA merged
└── merged_chroma.safetensors      # Chroma with differences applied
```

## Model File Requirements

### Flux Dev Model
- Standard Flux Dev UNet file (usually ~12GB)
- Format: `.safetensors`
- Common names: `flux_dev.safetensors`, `flux1-dev.safetensors`

### Chroma Model  
- Chroma UNet file (usually ~8-9GB)
- Format: `.safetensors`
- Should be the base model, not a fine-tuned version

### LoRA File
- Flux-trained LoRA in `.safetensors` format
- Any rank/training style supported
- Character and style LoRAs work better than concept LoRAs

## Expected Results by LoRA Type

| LoRA Type | Conversion Success | Recommended Settings |
|-----------|-------------------|---------------------|
| **Character** | 🟡 Moderate (20-40%) | `--amplify 1.5 --aggressive-mapping` |
| **Art Style** | 🟡 Moderate (15-30%) | `--amplify 2.0 --force-extract-all` |
| **Photography** | 🔴 Poor (5-15%) | `--amplify 2.5 --threshold 1e-8` |
| **Pose/Composition** | 🔴 Very Poor (<10%) | Not recommended |
| **Concept/Object** | 🟡 Variable (10-30%) | `--aggressive-mapping` |

## Validation

After conversion, test the LoRA:

1. **Load in your preferred interface** (ComfyUI, Automatic1111, etc.)
2. **Start with low strength** (0.3-0.5) and increase gradually
3. **Compare with original** using same prompt/settings
4. **Check for artifacts** - conversion issues may cause visual problems

## Contributing

This is experimental software. Improvements welcome:

- Better architectural mapping strategies
- Memory optimization
- Support for additional model formats
- Validation and testing tools

## License

MIT License - Use at your own risk

## Disclaimer

This tool is provided as-is for experimental purposes. The fundamental architectural differences between Flux Dev and Chroma mean that perfect conversions are impossible. Results will vary significantly based on:

- LoRA training methodology
- Target concepts/styles  
- Model architecture differences
- Conversion parameters used

**For production use, training LoRAs directly on Chroma is strongly recommended.**

---

**🔗 Need Help?**
- Check the troubleshooting section above
- Review the console output for specific error messages  
- Consider training directly on Chroma for best results