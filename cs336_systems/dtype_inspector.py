import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import json
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class LayerDtypeInfo:
    """Information about data types in a layer."""
    layer_name: str
    layer_type: str
    input_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    weight_dtype: Optional[str] = None
    grad_dtype: Optional[str] = None
    forward_called: bool = False
    backward_called: bool = False


class DtypeInspector:
    """Utility to inspect data types during forward and backward passes with autocast."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_info: Dict[str, LayerDtypeInfo] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up hooks for all layers in the model."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                layer_type = type(module).__name__
                self.layer_info[name] = LayerDtypeInfo(
                    layer_name=name,
                    layer_type=layer_type
                )
                
                # Register forward hook
                hook = module.register_forward_hook(
                    self._create_forward_hook(name)
                )
                self.hooks.append(hook)
                
                # Register backward hook
                hook = module.register_backward_hook(
                    self._create_backward_hook(name)
                )
                self.hooks.append(hook)
    
    def _create_forward_hook(self, layer_name: str) -> Callable:
        """Create a forward hook for a specific layer."""
        def forward_hook(module, input, output):
            info = self.layer_info[layer_name]
            info.forward_called = True
            
            # Record input dtype
            if input and len(input) > 0:
                info.input_dtype = str(input[0].dtype)
            
            # Record output dtype
            if isinstance(output, torch.Tensor):
                info.output_dtype = str(output.dtype)
            elif isinstance(output, (list, tuple)):
                if len(output) > 0 and isinstance(output[0], torch.Tensor):
                    info.output_dtype = str(output[0].dtype)
            
            # Record weight dtype if available
            if hasattr(module, 'weight') and module.weight is not None:
                info.weight_dtype = str(module.weight.dtype)
        
        return forward_hook
    
    def _create_backward_hook(self, layer_name: str) -> Callable:
        """Create a backward hook for a specific layer."""
        def backward_hook(module, grad_input, grad_output):
            info = self.layer_info[layer_name]
            info.backward_called = True
            
            # Record gradient dtype
            if grad_output and len(grad_output) > 0:
                info.grad_dtype = str(grad_output[0].dtype)
        
        return backward_hook
    
    def reset(self):
        """Reset all layer information."""
        for info in self.layer_info.values():
            info.input_dtype = None
            info.output_dtype = None
            info.weight_dtype = None
            info.grad_dtype = None
            info.forward_called = False
            info.backward_called = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all layer data types."""
        summary = {
            'layers': [asdict(info) for info in self.layer_info.values()],
            'statistics': self._compute_statistics()
        }
        return summary
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics about data type usage."""
        stats = {
            'total_layers': len(self.layer_info),
            'layers_with_forward': sum(1 for info in self.layer_info.values() if info.forward_called),
            'layers_with_backward': sum(1 for info in self.layer_info.values() if info.backward_called),
            'dtype_counts': defaultdict(int)
        }
        
        for info in self.layer_info.values():
            if info.output_dtype:
                stats['dtype_counts'][info.output_dtype] += 1
            if info.weight_dtype:
                stats['dtype_counts'][info.weight_dtype] += 1
            if info.grad_dtype:
                stats['dtype_counts'][info.grad_dtype] += 1
        
        return stats
    
    def print_summary(self):
        """Print a formatted summary of data types."""
        print("\n" + "="*80)
        print("DATA TYPE INSPECTION SUMMARY")
        print("="*80)
        
        stats = self._compute_statistics()
        print(f"Total layers: {stats['total_layers']}")
        print(f"Layers with forward pass: {stats['layers_with_forward']}")
        print(f"Layers with backward pass: {stats['layers_with_backward']}")
        
        print(f"\nData type distribution:")
        for dtype, count in stats['dtype_counts'].items():
            print(f"  {dtype}: {count} occurrences")
        
        print(f"\nDetailed layer information:")
        for name, info in self.layer_info.items():
            if info.forward_called or info.backward_called:
                print(f"\n  {name} ({info.layer_type}):")
                if info.input_dtype:
                    print(f"    Input: {info.input_dtype}")
                if info.output_dtype:
                    print(f"    Output: {info.output_dtype}")
                if info.weight_dtype:
                    print(f"    Weight: {info.weight_dtype}")
                if info.grad_dtype:
                    print(f"    Gradient: {info.grad_dtype}")
    
    def save_summary(self, filename: str):
        """Save the summary to a JSON file."""
        summary = self.get_summary()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Data type summary saved to {filename}")
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


@contextmanager
def inspect_dtypes(model: nn.Module):
    """Context manager for inspecting data types during model execution."""
    inspector = DtypeInspector(model)
    try:
        yield inspector
    finally:
        inspector.cleanup()


def inspect_model_with_autocast(model: nn.Module, 
                               input_tensor: torch.Tensor,
                               dtype: torch.dtype = torch.float16,
                               device: str = "cuda") -> Dict[str, Any]:
    """
    Inspect data types during forward and backward pass with autocast.
    
    Args:
        model: The model to inspect
        input_tensor: Input tensor for the model
        dtype: Autocast dtype (typically torch.float16 for mixed precision)
        device: Device to run on
    
    Returns:
        Dictionary containing data type information
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    with inspect_dtypes(model) as inspector:
        # Forward pass with autocast
        with torch.autocast(device_type=device, dtype=dtype):
            output = model(input_tensor)
        
        # Create a dummy loss for backward pass
        if output.dim() > 1:
            target = torch.randint(0, output.size(-1), (output.size(0),), device=device)
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target)
        else:
            loss = output.sum()
        
        # Backward pass with autocast
        with torch.autocast(device_type=device, dtype=dtype):
            loss.backward()
        
        # Get summary
        summary = inspector.get_summary()
        inspector.print_summary()
        
        return summary


def compare_dtypes_with_without_autocast(model: nn.Module,
                                        input_tensor: torch.Tensor,
                                        device: str = "cuda") -> Dict[str, Any]:
    """
    Compare data types with and without autocast.
    
    Args:
        model: The model to inspect
        input_tensor: Input tensor for the model
        device: Device to run on
    
    Returns:
        Dictionary containing comparison results
    """
    results = {}
    
    # Without autocast
    print("\n" + "="*50)
    print("WITHOUT AUTOCAST")
    print("="*50)
    model_copy = type(model)(**model.config).to(device)
    model_copy.load_state_dict(model.state_dict())
    
    with inspect_dtypes(model_copy) as inspector:
        output = model_copy(input_tensor)
        if output.dim() > 1:
            target = torch.randint(0, output.size(-1), (output.size(0),), device=device)
            loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target)
        else:
            loss = output.sum()
        loss.backward()
        
        results['without_autocast'] = inspector.get_summary()
    
    # With autocast
    print("\n" + "="*50)
    print("WITH AUTOCAST (float16)")
    print("="*50)
    model_copy2 = type(model)(**model.config).to(device)
    model_copy2.load_state_dict(model.state_dict())
    
    with inspect_dtypes(model_copy2) as inspector:
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = model_copy2(input_tensor)
            if output.dim() > 1:
                target = torch.randint(0, output.size(-1), (output.size(0),), device=device)
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target)
            else:
                loss = output.sum()
            loss.backward()
        
        results['with_autocast'] = inspector.get_summary()
    
    return results 