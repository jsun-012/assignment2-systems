#!/usr/bin/env python3
"""
Benchmarking script for CS336 Transformer model.

This script performs end-to-end benchmarking of forward and backward passes
in the transformer model with configurable hyperparameters.
"""

import argparse
import timeit
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import json
import sys
import os
from contextlib import contextmanager

# Try to import NVTX for CUDA profiling
try:
    import torch.cuda.nvtx as nvtx_mod
    NVTX_IMPORTED = True
except ImportError:
    NVTX_IMPORTED = False

# Add the cs336-basics package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cs336-basics'))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch

@contextmanager
def nullcontext(*args, **kwargs):
    yield

class NVTXRangeWrapper:
    def __init__(self, enabled: bool):
        self.enabled = enabled and NVTX_IMPORTED

    def range(self, msg: str):
        if self.enabled:
            return nvtx_mod.range(msg)
        else:
            return nullcontext()

class ModelBenchmarker:
    """Benchmarking class for transformer models."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", use_nvtx: bool = False, dtype: torch.dtype = torch.float32):
        self.device = device
        self.model = None
        self.optimizer = None
        self.nvtx = NVTXRangeWrapper(use_nvtx)
        self.dtype = dtype
        
    def initialize_model(self, hyperparams: Dict[str, Any]) -> None:
        """Initialize the model with given hyperparameters."""
        with self.nvtx.range("Initialize Model"):
            print(f"Initializing model with hyperparameters: {hyperparams}")
            
            self.model = BasicsTransformerLM(
                vocab_size=hyperparams.get('vocab_size', 10000),
                context_length=hyperparams.get('context_length', 1024),
                d_model=hyperparams.get('d_model', 768),
                num_layers=hyperparams.get('num_layers', 12),
                num_heads=hyperparams.get('num_heads', 12),
                d_ff=hyperparams.get('d_ff', 3072),
                rope_theta=hyperparams.get('rope_theta', 10000.0)
            ).to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=hyperparams.get('learning_rate', 1e-4),
                weight_decay=hyperparams.get('weight_decay', 0.01)
            )
            
            print(f"Model initialized with {self.model.get_num_params():,} parameters")
            print(f"Model device: {self.device}")
            
            # Print layer information with data types
            self.print_layer_info()
    
    def print_layer_info(self) -> None:
        """Print layer information with data types."""
        print("\n" + "="*60)
        print("LAYER INFORMATION")
        print("="*60)
        
        # Print main components
        print(f"Token Embeddings: {self.model.token_embeddings.weight.dtype}")
        # print(f"Positional Encoder: {self.model.positional_encoder.rope_theta.dtype}")
        print(f"Final Layer Norm: {self.model.ln_final.weight.dtype}")
        print(f"LM Head: {self.model.lm_head.weight.dtype}")
        
        # Print transformer layers
        print(f"\nTransformer Layers ({len(self.model.layers)} layers):")
        for i, layer in enumerate(self.model.layers):
            print(f"  Layer {i}:")
            print(f"    Attention:")
            print(f"      Q Projection: {layer.attn.q_proj.weight.dtype}")
            print(f"      K Projection: {layer.attn.k_proj.weight.dtype}")
            print(f"      V Projection: {layer.attn.v_proj.weight.dtype}")
            print(f"      Output Projection: {layer.attn.output_proj.weight.dtype}")
            # print(f"      Positional Encoder: {layer.attn.positional_encoder.rope_theta.dtype}")
            print(f"    Feed-Forward:")
            print(f"      W1: {layer.ffn.w1.weight.dtype}")
            print(f"      W2: {layer.ffn.w2.weight.dtype}")
            print(f"      W3: {layer.ffn.w3.weight.dtype}")
            print(f"    Layer Norms:")
            print(f"      LN1: {layer.ln1.weight.dtype}")
            print(f"      LN2: {layer.ln2.weight.dtype}")
            if i < len(self.model.layers) - 1:  # Don't print separator for last layer
                print()
        
        print("="*60)
        
    def generate_random_batch(self, batch_size: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a random batch of data for benchmarking."""
        # Create a dummy dataset (random token IDs)
        vocab_size = 10000  # Default vocab size
        dataset_size = batch_size * context_length * 10  # Ensure we have enough data
        dataset = np.random.randint(0, vocab_size, size=dataset_size, dtype=np.int64)
        
        # Generate batch using the existing get_batch function
        x, y = get_batch(dataset, batch_size, context_length, self.device)
        return x, y
    
    def warm_up(self, batch_size: int, context_length: int, num_warmup: int) -> None:
        """Run warm-up steps to stabilize performance measurements."""
        print(f"Running {num_warmup} warm-up steps...")
        with self.nvtx.range("Warm-up"):
            for i in range(num_warmup):
                with self.nvtx.range(f"Warm-up Step {i+1}"):
                    x, y = self.generate_random_batch(batch_size, context_length)
                    
                    # Forward pass
                    with self.nvtx.range("Forward"):
                        with torch.autocast(device_type=self.device, dtype=self.dtype):
                            logits = self.model(x)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # Backward pass
                    with self.nvtx.range("Backward"):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    
                    if (i + 1) % max(1, num_warmup // 10) == 0:
                        print(f"  Warm-up step {i + 1}/{num_warmup}")
                
                    # Synchronize GPU if using CUDA
                    if self.device == "cuda":
                        torch.cuda.synchronize()
            
            print("Warm-up completed.")
    
    def benchmark_forward_only(self, batch_size: int, context_length: int, num_steps: int) -> Dict[str, float]:
        """Benchmark forward pass only."""
        print(f"Benchmarking forward pass only for {num_steps} steps...")

        times = []
        with self.nvtx.range("Benchmark Forward Only"):
            for i in range(num_steps):
                start = timeit.default_timer()
                x, _ = self.generate_random_batch(batch_size, context_length)
                with torch.no_grad():
                    with self.nvtx.range("Forward"):
                        with torch.autocast(device_type=self.device, dtype=self.dtype):
                            logits = self.model(x)
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                end = timeit.default_timer()
                times.append((end - start) * 1000)  # ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'all_times_ms': times
        }

    def benchmark_forward_backward(self, batch_size: int, context_length: int, num_steps: int) -> Dict[str, float]:
        """Benchmark both forward and backward passes."""
        print(f"Benchmarking forward + backward pass for {num_steps} steps...")

        times = []
        with self.nvtx.range("Benchmark Forward+Backward"):
            for i in range(num_steps):
                start = timeit.default_timer()
                x, y = self.generate_random_batch(batch_size, context_length)
                # Forward pass
                with self.nvtx.range("Forward"):
                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        logits = self.model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                # Backward pass
                with self.nvtx.range("Backward"):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = timeit.default_timer()
                times.append((end - start) * 1000)  # ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'all_times_ms': times
        }

    def run_benchmark(self, 
                     hyperparams: Dict[str, Any],
                     batch_size: int = 32,
                     context_length: int = 1024,
                     num_warmup: int = 10,
                     num_steps: int = 100,
                     benchmark_type: str = "both") -> Dict[str, Any]:
        """Run the complete benchmarking process."""
        with self.nvtx.range("Run Benchmark"):
            # Initialize model
            self.initialize_model(hyperparams)
            
            # Warm up
            self.warm_up(batch_size, context_length, num_warmup)
            
            # Run benchmark
            if benchmark_type == "forward":
                results = self.benchmark_forward_only(batch_size, context_length, num_steps)
            elif benchmark_type == "both":
                results = self.benchmark_forward_backward(batch_size, context_length, num_steps)
            else:
                raise ValueError(f"Unknown benchmark type: {benchmark_type}")
            
            # Add metadata
            results.update({
                'hyperparams': hyperparams,
                'batch_size': batch_size,
                'context_length': context_length,
                'num_warmup': num_warmup,
                'num_steps': num_steps,
                'benchmark_type': benchmark_type,
                'device': self.device,
                'model_params': self.model.get_num_params()
            })
            
            return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print benchmark results in a formatted way."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        print(f"Model Parameters: {results['model_params']:,}")
        print(f"Device: {results['device']}")
        print(f"Benchmark Type: {results['benchmark_type']}")
        print(f"Batch Size: {results['batch_size']}")
        print(f"Context Length: {results['context_length']}")
        print(f"Warm-up Steps: {results['num_warmup']}")
        print(f"Benchmark Steps: {results['num_steps']}")
        
        print(f"\nTiming Results (ms):")
        print(f"  Average: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f}")
        print(f"  Min:     {results['min_time_ms']:.2f}")
        print(f"  Max:     {results['max_time_ms']:.2f}")
        
        # Calculate throughput
        if results['benchmark_type'] == "forward":
            throughput = results['batch_size'] / (results['avg_time_ms'] / 1000)
            print(f"\nThroughput: {throughput:.2f} samples/second")
        else:
            throughput = results['batch_size'] / (results['avg_time_ms'] / 1000)
            print(f"\nThroughput: {throughput:.2f} samples/second (forward + backward)")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark CS336 Transformer Model")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")
    
    # Benchmarking parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--num_warmup", type=int, default=10, help="Number of warm-up steps")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--benchmark_type", choices=["forward", "both"], default="both", 
                       help="Type of benchmark: 'forward' for forward pass only, 'both' for forward + backward")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file to save results (JSON format)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX CUDA profiling ranges")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], 
                       help="Data type for model (default: float32)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Prepare hyperparameters
    hyperparams = {
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta,
        'learning_rate': 1e-4,
        'weight_decay': 0.01
    }
    
    # Convert dtype string to torch.dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Create benchmarker and run
    benchmarker = ModelBenchmarker(device=device, use_nvtx=args.nvtx, dtype=dtype)
    
    try:
        results = benchmarker.run_benchmark(
            hyperparams=hyperparams,
            batch_size=args.batch_size,
            context_length=args.context_length,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            benchmark_type=args.benchmark_type
        )
        
        # Print results
        benchmarker.print_results(results)
        
        # Save results if output file specified
        if args.output:
            # Remove the 'all_times_ms' list to keep the file smaller
            results_to_save = {k: v for k, v in results.items() if k != 'all_times_ms'}
            with open(args.output, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise


if __name__ == "__main__":
    main()
