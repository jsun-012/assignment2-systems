# CS336 Transformer Model Benchmarking

This directory contains scripts for benchmarking the CS336 Transformer model implementation.

## Files

- `benchmark_script.py` - Main benchmarking script
- `example_benchmark.py` - Example usage with different configurations
- `BENCHMARK_README.md` - This documentation file

## Model Specifications

The benchmarking script is configured to work with the following model specifications:

| Size | d_model | d_ff | num_layers | num_heads |
|------|---------|------|------------|-----------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

**Note**: All models use a vocabulary size of 10,000 and a batch size of 4 by default.

## Quick Start

### Basic Usage

```bash
# Run a basic benchmark with default parameters
python cs336_systems/benchmark_script.py

# Run with custom hyperparameters
python cs336_systems/benchmark_script.py \
    --num_layers 12 \
    --d_model 768 \
    --num_heads 12 \
    --batch_size 32 \
    --num_steps 100
```

### Command Line Arguments

#### Model Hyperparameters
- `--vocab_size` (default: 10000) - Vocabulary size
- `--context_length` (default: 1024) - Context length
- `--d_model` (default: 768) - Model dimension
- `--num_layers` (default: 12) - Number of transformer layers
- `--num_heads` (default: 12) - Number of attention heads
- `--d_ff` (default: 3072) - Feed-forward dimension
- `--rope_theta` (default: 10000.0) - RoPE theta parameter

#### Benchmarking Parameters
- `--batch_size` (default: 4) - Batch size for benchmarking
- `--num_warmup` (default: 10) - Number of warm-up steps
- `--num_steps` (default: 100) - Number of benchmark steps
- `--benchmark_type` (choices: "forward", "both", default: "both") - Type of benchmark

#### Output Options
- `--output` - Output file to save results (JSON format)
- `--device` - Device to use (cuda/cpu, auto-detects if not specified)

## Examples

### 1. Small Model Benchmark (Forward Pass Only)

```bash
python cs336_systems/benchmark_script.py \
    --num_layers 12 \
    --d_model 768 \
    --num_heads 12 \
    --d_ff 3072 \
    --batch_size 4 \
    --benchmark_type forward \
    --num_steps 50 \
    --output small_model_forward.json
```

### 2. Medium Model Benchmark (Forward + Backward)

```bash
python cs336_systems/benchmark_script.py \
    --num_layers 24 \
    --d_model 1024 \
    --num_heads 16 \
    --d_ff 4096 \
    --batch_size 4 \
    --benchmark_type both \
    --num_steps 100 \
    --output medium_model_both.json
```

### 3. Large Model Benchmark (CPU)

```bash
python cs336_systems/benchmark_script.py \
    --num_layers 36 \
    --d_model 1280 \
    --num_heads 20 \
    --d_ff 5120 \
    --batch_size 4 \
    --device cpu \
    --num_steps 20 \
    --output large_model_cpu.json
```

### 4. XL Model Benchmark (GPU)

```bash
python cs336_systems/benchmark_script.py \
    --num_layers 48 \
    --d_model 1600 \
    --num_heads 25 \
    --d_ff 6400 \
    --batch_size 4 \
    --num_steps 50 \
    --output xl_model_gpu.json
```

## Running Example Benchmarks

### Run All Benchmarks

To run all example benchmarks at once:

```bash
python cs336_systems/example_benchmark.py
```

This will run four different benchmark configurations:
1. Small model (12 layers) - forward pass only
2. Medium model (24 layers) - forward + backward
3. Large model (36 layers) - CPU only
4. XL model (48 layers) - GPU

### Run Specific Model Size

To run a benchmark for a specific model size:

```bash
# Run small model benchmark
python cs336_systems/example_benchmark.py --model-size small

# Run medium model with forward pass only
python cs336_systems/example_benchmark.py --model-size medium --benchmark-type forward

# Run large model on CPU with custom parameters
python cs336_systems/example_benchmark.py --model-size large --device cpu --num-steps 50

# Run XL model with custom output file
python cs336_systems/example_benchmark.py --model-size xl --output my_xl_results.json
```

### Available Model Sizes

- `small` - 12 layers, d_model=768, d_ff=3072, num_heads=12
- `medium` - 24 layers, d_model=1024, d_ff=4096, num_heads=16
- `large` - 36 layers, d_model=1280, d_ff=5120, num_heads=20
- `xl` - 48 layers, d_model=1600, d_ff=6400, num_heads=25
- `2.7B` - 32 layers, d_model=2560, d_ff=10240, num_heads=32

### Command Line Options

- `--model-size` - Model size to benchmark (default: all)
- `--benchmark-type` - Type of benchmark: "forward" or "both" (default: both)
- `--device` - Device to use: "cuda" or "cpu" (default: auto-detect)
- `--num-steps` - Number of benchmark steps (default: 100)
- `--num-warmup` - Number of warm-up steps (default: 10)
- `--output` - Output file name (default: auto-generated)
- `--dtype` - Data type for model: "float32", "float16", or "bfloat16" (default: float32)

## Output Format

The benchmark script outputs results in two formats:

### Console Output
```
============================================================
BENCHMARK RESULTS
============================================================
Model Parameters: 124,439,808
Device: cuda
Benchmark Type: both
Batch Size: 32
Context Length: 1024
Warm-up Steps: 10
Benchmark Steps: 100

Timing Results (ms):
  Average: 45.23 ± 2.15
  Min:     42.10
  Max:     51.30

Throughput: 707.27 samples/second (forward + backward)
============================================================
```

### JSON Output
When using the `--output` flag, results are saved in JSON format:

```json
{
  "avg_time_ms": 45.23,
  "std_time_ms": 2.15,
  "min_time_ms": 42.10,
  "max_time_ms": 51.30,
  "hyperparams": {
    "vocab_size": 10000,
    "context_length": 1024,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": 10000.0,
    "learning_rate": 0.0001,
    "weight_decay": 0.01
  },
  "batch_size": 32,
  "context_length": 1024,
  "num_warmup": 10,
  "num_steps": 100,
  "benchmark_type": "both",
  "device": "cuda",
  "model_params": 124439808
}
```

## Features

### ✅ Implemented Features

1. **Model Initialization**: Supports all hyperparameters for the BasicsTransformerLM
2. **Random Batch Generation**: Uses the existing `get_batch` function from cs336-basics
3. **Warm-up Steps**: Configurable number of warm-up steps before timing
4. **High-Resolution Timing**: Uses Python's `timeit` module for accurate measurements
5. **Forward/Backward Benchmarking**: Supports both forward-only and full training benchmarks
6. **Statistical Analysis**: Provides mean, std, min, max timing statistics
7. **Throughput Calculation**: Calculates samples per second
8. **Device Support**: Works on both CPU and CUDA
9. **JSON Output**: Saves detailed results for further analysis
10. **Command Line Interface**: Full argument parsing with help text

### 🔧 Technical Details

- **Warm-up**: Runs full forward + backward passes to stabilize GPU memory allocation and cuDNN kernels
- **Timing**: Uses `timeit.Timer` with `repeat()` for multiple measurements
- **Memory Management**: Properly handles GPU memory with synchronization
- **Error Handling**: Graceful error handling with informative messages
- **Modular Design**: Clean class-based architecture for easy extension

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model size
2. **Import Errors**: Ensure cs336-basics is properly installed and in the path
3. **Slow Performance**: Increase warm-up steps for more stable measurements

### Performance Tips

1. **Warm-up**: Always use warm-up steps (10-20) for stable GPU measurements
2. **Batch Size**: Larger batch sizes generally provide better GPU utilization
3. **Context Length**: Longer sequences increase memory usage and computation time
4. **Model Size**: Larger models require more memory and computation

## Dependencies

- torch
- numpy
- cs336-basics (local package)
- einops
- einx
- jaxtyping

All dependencies should be available through the project's `pyproject.toml`. 