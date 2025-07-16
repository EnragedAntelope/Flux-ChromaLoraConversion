# Flux to Chroma LoRA Converter & Scanner

This repository contains a set of tools to convert LoRAs made for the Flux model to be compatible with Chroma, a heavily modified and more efficient variant. It includes a powerful converter and an essential pre-flight compatibility scanner.

## 🎯 Overview

Chroma is a streamlined model variant based on Flux, with a different architecture and fewer parameters. Standard Flux LoRAs are not directly compatible. This converter uses an advanced "apply-difference-extract" method to rebuild the LoRA's influence within the Chroma architecture.

### Key Features
- **High-Fidelity UNet Conversion**: Intelligently merges and rebuilds UNet layers, handling various LoRA formats (Kohya, Diffusers) and accumulating weights from multiple components (e.g., Q, K, V projections) into single layers.
- **⚠️ Experimental Text Encoder Conversion**: Includes an *optional, experimental* feature to convert T5 Text Encoder (`lora_te1`) weights. **This is disabled by default as these keys may not map to Chroma loras.**
- **Compatibility Pre-Scanner**: Analyze your LoRAs *before* conversion to get a detailed report and a UNet compatibility score, saving you time and effort.
- **Batch Processing**: Convert an entire folder of LoRAs with a single command.
- **Metadata Preservation**: Automatically carries over trigger words and other metadata from the original LoRA to the converted version.
- **Memory Efficiency**: Uses chunk-based processing and CPU fallbacks to handle large models on systems with limited VRAM.
- **Auto-Detection**: Automatically detects the LoRA's rank and naming style.

## 📋 Requirements

- Python 3.8+
- PyTorch (CPU or CUDA version)
- 16GB+ of System RAM is recommended for CPU mode.
- 24GB+ of VRAM is recommended for GPU mode.

Install the required packages using pip:
```bash
pip install torch safetensors numpy tqdm psutil gputil
```
*(Note: `psutil` and `gputil` are optional but recommended for memory monitoring in debug mode).*

---

## ⭐ Important: Use the Pruned Flux Model

For the conversion process, you **must** use a Flux base model. It is **strongly recommended** to use the pruned version to save disk space and reduce load times, as it is functionally identical for this process.

- **Model:** `flux1-dev-pruned.safetensors`
- **Download Link:** [Hugging Face - silveroxides/pruned-models](https://huggingface.co/silveroxides/pruned-models/blob/main/flux1-dev-pruned.safetensors)

---

## 🚀 Quick Start Guide

The recommended workflow is to first scan your LoRAs for compatibility and then convert the best candidates.

### Step 1: Scan LoRAs for Compatibility

Before converting, use the scanner to see which of your LoRAs are good candidates. This tool analyzes the LoRA's structure and gives it a score based on how many of its **UNet layers** can be converted.

**Scan an entire directory and get a report:**
```bash
python flux-chroma-compatibility-scanner.py ^
    --scan-dir "D:\path\to\your\flux-loras" ^
    --min-score 70 ^
    --save-report "compatibility_report.txt"
```

**Interpreting the UNet Score:**
- **90-100% (Excellent):** Ideal candidate. Most or all of the LoRA's visual effect should transfer.
- **70-89% (Good):** High chance of success. The LoRA should work well, though some minor aspects might be lost.
- **50-69% (Fair):** Partial conversion. The LoRA might work for broad styles but may lose specific details.
- **< 50% (Poor):** Not recommended. The LoRA likely targets layers that don't exist in Chroma.

### Step 2: Convert Your LoRA

Once you've identified a good candidate, use the converter.

**Basic Single-File Conversion (Recommended for most LoRAs):**
```cmd
python flux-chroma-converter.py ^
    --flux-base "D:\models\flux1-dev-pruned.safetensors" ^
    --chroma-base "D:\models\chroma-unlocked.safetensors" ^
    --flux-lora "D:\flux-loras\my-character.safetensors"
```
*(The output will be automatically named `my-character-chroma.safetensors` in the same folder. This command correctly skips the broken Text Encoder conversion).*

**Batch Conversion:**
```bash
python flux-chroma-converter.py  ^
    --flux-base "/path/to/models/flux1-dev-pruned.safetensors"  ^
    --chroma-base "/path/to/models/chroma-unlocked.safetensors"  ^
    --lora-folder "/path/to/flux-loras/"
```
*(This will convert all `.safetensors` files in the folder, appending `-chroma` to each filename).*

## 🛠️ Command-Line Options

### Compatibility Scanner (`flux-chroma-compatibility-scanner.py`)

- `--lora`: Path to a single LoRA file to analyze.
- `--scan-dir`: Path to a directory to scan for LoRAs.
- `--min-score`: (Optional) Only show LoRAs with a UNet score at or above this value (0-100).
- `--top-n`: (Optional) Show the top N results when scanning a directory. Default is 10.
- `--save-report`: (Optional) Save the full, detailed report to a text file.
- `--json`: (Optional) Save the results as a JSON file.
- `--detailed`: (Optional) Show more structural details in the console output.

### Converter (`flux-chroma-converter.py`)

**Required Arguments:**
- `--flux-base`: Path to the Flux base model (e.g., `flux1-dev-pruned.safetensors`).
- `--chroma-base`: Path to the Chroma base model.
- `--flux-lora` OR `--lora-folder`: Path to the input LoRA file or a folder of LoRAs.

**Optional Arguments:**
- `--output-lora`: Path for the converted LoRA. If omitted, the output is auto-named. (Ignored in batch mode).
- `--device`: The device to use for processing. `cuda` for GPU, `cpu` for system RAM. (Default: `cuda`).
- `--rank`: The rank for the extracted LoRA. Use `-1` to auto-detect from the original LoRA's UNet layers. (Default: -1).
- `--lora-alpha`: The strength to use when merging the LoRA into the Flux model. (Default: 1.0).
- `--chunk-size`: Number of layers to process at once. Lower this if you get out-of-memory errors. (Default: 50).
- `--enable-text-encoder-conversion`: **(EXPERIMENTAL)** Use this flag to *attempt* to convert text encoder weights. This is disabled by default and is known to produce keys that do not remap accurately.
- `--analyze-only`: Run the analysis portion without performing the conversion.
- `--skip-validation`: Skips the final validation step that checks LoRA shape compatibility.
- `--debug`: Enable verbose logging for troubleshooting.

## 💡 Understanding the Process

### How It Works
The conversion is a four-step process designed to translate a LoRA's influence from one model architecture to another:
1.  **Apply LoRA**: The original Flux LoRA is merged into the Flux base model at full strength.
2.  **Extract Difference**: The script calculates the precise difference (the "delta") between the original Flux model and the LoRA-merged version.
3.  **Apply Difference**: This "delta" is then applied to the Chroma base model, effectively transferring the LoRA's changes.
4.  **Extract LoRA**: A brand new, Chroma-native LoRA is extracted from the modified Chroma model using SVD (Singular Value Decomposition).

### What Gets Converted
- ✅ **UNet Layers**: All compatible attention and feed-forward/MLP layers in both single and double blocks are converted. This is the primary function and works well.
- ✅ **Metadata**: Trigger words, tags, and other information are preserved.
- ⚠️ **T5 Text Encoder (`lora_te1`)**: Conversion is **experimental and disabled by default**. The generated keys are currently **not loaded correctly by ComfyUI**. You can force the conversion with a flag, but it is not recommended for production use.
- ❌ **Modulation Layers**: These layers exist in Flux but not in Chroma. They are safely skipped. You may see "key not loaded" warnings for these in ComfyUI, which is normal and expected.
- ❌ **CLIP-L Text Encoder (`lora_te2`)**: These weights are not used by the standard Chroma workflow and are safely skipped.

## 🐛 Troubleshooting

**Problem: My character/style LoRA doesn't work or has a weak effect.**
- **Reason:** Many character and style LoRAs rely heavily on the Text Encoder. The conversion for these layers is currently experimental and known to fail in ComfyUI. The UNet portion of the LoRA is likely converted correctly, but the crucial text-based influence is lost. There is currently no fix for this, but you can try increasing strength to see if fidelity improves.

**Problem: CUDA Out of Memory error.**
- **Solution 1:** Rerun the command with `--device cpu`. This will be slower but will use your system's RAM instead of VRAM.
- **Solution 2:** If you want to use your GPU, lower the processing chunk size with `--chunk-size 20` or `--chunk-size 10`.

**Problem: The conversion fails with an error.**
- Rerun the command with the `--debug` flag to get a detailed error log. This can help identify if it's a file path issue, a corrupted input file, or a bug.

## 📈 Best Practices

1.  **Scan First, Convert Later**: Always use the `flux-chroma-compatibility-scanner.py` script first to check which LoRAs have a high UNet compatibility score.
2.  **Use Batch Mode**: For converting multiple files, the `--lora-folder` argument is much more efficient than running the script manually for each file.
3.  **Start with Default Settings**: The default settings (which skip TE conversion) are recommended for the best results.
4.  **Test at Different Strengths**: A converted LoRA might have a slightly different "feel." Test it in your image generator at strengths from `0.8` to `1.2` to find the new sweet spot.
5.  **Keep Your Originals**: Never delete your original Flux LoRAs.

## 📝 License

This tool is provided under the MIT License. You are free to use, modify, and distribute it. Please respect the licenses of the models and LoRAs you are converting.
```
