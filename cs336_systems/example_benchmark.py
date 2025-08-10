#!/usr/bin/env python3
"""
Example usage of the benchmarking script.

This script demonstrates how to use the benchmark_script.py with different
model configurations and benchmarking parameters.
"""

import subprocess
import sys
import os
import argparse

# Model specifications from the assignment
MODEL_SPECS = {
    'small': {
        'num_layers': 12,
        'd_model': 768,
        'num_heads': 12,
        'd_ff': 3072,
        'description': 'Small model (12 layers)'
    },
    'medium': {
        'num_layers': 24,
        'd_model': 1024,
        'num_heads': 16,
        'd_ff': 4096,
        'description': 'Medium model (24 layers)'
    },
    'large': {
        'num_layers': 36,
        'd_model': 1280,
        'num_heads': 20,
        'd_ff': 5120,
        'description': 'Large model (36 layers)'
    },
    'xl': {
        'num_layers': 48,
        'd_model': 1600,
        'num_heads': 25,
        'd_ff': 6400,
        'description': 'XL model (48 layers)'
    },
    '2.7B': {
        'num_layers': 32,
        'd_model': 2560,
        'num_heads': 32,
        'd_ff': 10240,
        'description': '2.7B model (32 layers)'
    }
}

def run_single_benchmark(model_size: str, benchmark_type: str = "both", device: str = None, 
                        num_steps: int = 10, num_warmup: int = 5, output_file: str = None, nvtx: bool = False, 
                        dtype: str = "float32"):
    """Run a single benchmark for a specific model size."""
    
    if model_size not in MODEL_SPECS:
        print(f"Error: Unknown model size '{model_size}'")
        print(f"Available model sizes: {', '.join(MODEL_SPECS.keys())}")
        return False
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_dir, "benchmark_script.py")
    
    # Get model specifications
    specs = MODEL_SPECS[model_size]
    
    # Set default output file if not provided
    if output_file is None:
        output_file = f"{model_size}_model_{benchmark_type}.json"
    
    print(f"\nRunning {specs['description']} benchmark:")
    print(f"  Benchmark type: {benchmark_type}")
    print(f"  Device: {device or 'auto'}")
    print(f"  Steps: {num_steps}, Warm-up: {num_warmup}")
    print(f"  Output: {output_file}")
    print("-" * 50)
    
    # Build command
    cmd = [
        sys.executable, benchmark_script,
        "--num_layers", str(specs['num_layers']),
        "--d_model", str(specs['d_model']),
        "--num_heads", str(specs['num_heads']),
        "--d_ff", str(specs['d_ff']),
        "--batch_size", "4",  # Fixed batch size per assignment
        "--context_length", "512",  # Fixed context length per assignment
        "--num_warmup", str(num_warmup),
        "--num_steps", str(num_steps),
        "--benchmark_type", benchmark_type,
        "--output", output_file,
        "--dtype", dtype
    ]
    
    # Add device if specified
    if device:
        cmd.extend(["--device", device])
    # Add nvtx if specified
    if nvtx:
        cmd.append("--nvtx")
    # Add dtype if specified
    if dtype:
        cmd.extend(["--dtype", dtype])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ {model_size} model benchmark completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {model_size} model benchmark: {e}")
        return False

def run_all_benchmarks(nvtx: bool = False, dtype: str = "float32"):
    """Run all benchmark configurations."""
    print("CS336 Transformer Model Benchmarking Examples")
    print("=" * 60)
    
    # Run all model sizes with different configurations
    benchmarks = [
        ('small', 'forward', None, 50, 5, 'small_model_forward.json'),
        ('medium', 'both', None, 100, 10, 'medium_model_both.json'),
        ('large', 'both', 'cpu', 20, 3, 'large_model_cpu.json'),
        ('xl', 'both', None, 30, 5, 'xl_model_gpu.json')
    ]
    
    success_count = 0
    total_count = len(benchmarks)
    
    for model_size, benchmark_type, device, num_steps, num_warmup, output_file in benchmarks:
        if run_single_benchmark(model_size, benchmark_type, device, num_steps, num_warmup, output_file, nvtx=nvtx, dtype=dtype):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Benchmarking completed: {success_count}/{total_count} successful")
    print("Check the generated JSON files for detailed results.")
    print("\nModel configurations used:")
    for size, specs in MODEL_SPECS.items():
        print(f"{size:>6}: {specs['num_layers']:2d} layers, d_model={specs['d_model']:4d}, "
              f"d_ff={specs['d_ff']:5d}, num_heads={specs['num_heads']:2d}")

def main():
    parser = argparse.ArgumentParser(description="Run CS336 Transformer Model Benchmarks")
    parser.add_argument("--model-size", choices=list(MODEL_SPECS.keys()) + ['all'], 
                       default='all', help="Model size to benchmark (default: all)")
    parser.add_argument("--benchmark-type", choices=["forward", "both"], 
                       default="both", help="Type of benchmark (default: both)")
    parser.add_argument("--device", choices=["cuda", "cpu"], 
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--num-steps", type=int, default=10, 
                       help="Number of benchmark steps (default: 10)")
    parser.add_argument("--num-warmup", type=int, default=5, 
                       help="Number of warm-up steps (default: 5)")
    parser.add_argument("--output", type=str, 
                       help="Output file name (default: auto-generated)")
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX CUDA profiling ranges")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for model (default: float32)")
    
    args = parser.parse_args()
    
    if args.model_size == 'all':
        run_all_benchmarks(nvtx=args.nvtx, dtype=args.dtype)
    else:
        success = run_single_benchmark(
            model_size=args.model_size,
            benchmark_type=args.benchmark_type,
            device=args.device,
            num_steps=args.num_steps,
            num_warmup=args.num_warmup,
            output_file=args.output,
            nvtx=args.nvtx,
            dtype=args.dtype
        )
        
        if success:
            print(f"\n✓ {args.model_size} model benchmark completed!")
        else:
            print(f"\n✗ {args.model_size} model benchmark failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
