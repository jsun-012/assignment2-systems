#!/usr/bin/env python3
"""
Simple script to inspect data types during forward and backward passes with autocast.
This integrates with your existing benchmark setup.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_script import ModelBenchmarker
from dtype_inspector import inspect_dtypes


def inspect_with_benchmark_setup():
    """Inspect data types using your existing benchmark setup."""
    
    print("Setting up benchmarker...")
    benchmarker = ModelBenchmarker(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_nvtx=False,
        dtype=torch.float16
    )
    
    # Use the same hyperparams as in your small model
    hyperparams = {
        'vocab_size': 10000,
        'context_length': 512,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'd_ff': 3072,
        'rope_theta': 10000.0,
        'learning_rate': 0.0001,
        'weight_decay': 0.01
    }
    
    print("Initializing model...")
    benchmarker.initialize_model(hyperparams)
    
    # Generate a batch
    batch_size = 4
    context_length = 64
    x, y = benchmarker.generate_random_batch(batch_size, context_length)
    
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Target shape: {y.shape}, dtype: {y.dtype}")
    
    print("\n" + "="*80)
    print("INSPECTING DATA TYPES WITH AUTOCAST")
    print("="*80)
    
    # Inspect with autocast (exactly like your benchmark)
    with inspect_dtypes(benchmarker.model) as inspector:
        # Forward pass with autocast
        with torch.autocast(device_type=benchmarker.device, dtype=benchmarker.dtype):
            logits = benchmarker.model(x)
        
        # Loss computation
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass with autocast
        with torch.autocast(device_type=benchmarker.device, dtype=benchmarker.dtype):
            benchmarker.optimizer.zero_grad()
            loss.backward()
            benchmarker.optimizer.step()
        
        # Print results
        inspector.print_summary()
        
        # Save results
        inspector.save_summary('dtype_inspection_results.json')
        
        print(f"\nDetailed results saved to: dtype_inspection_results.json")


if __name__ == "__main__":
    print("Data Type Inspection with Benchmark Setup")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")
    
    inspect_with_benchmark_setup() 