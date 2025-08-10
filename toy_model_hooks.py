import torch
import torch.nn as nn
from typing import Dict, Any

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
        
        # Register hooks for all layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for all layers."""
        self.hooks = []
        
        # Register hooks for each layer
        layers = {
            'fc1': self.fc1,
            'ln': self.ln, 
            'fc2': self.fc2,
            'relu': self.relu
        }
        
        for name, layer in layers.items():
            # Forward hook
            def make_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        print(f"Forward {layer_name}: input_dtype={input[0].dtype}, output_dtype={output.dtype}")
                    else:
                        print(f"Forward {layer_name}: input_dtype={input[0].dtype}, output_dtype=multiple")
                return forward_hook
            
            # Backward hook
            def make_backward_hook(layer_name):
                def backward_hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        print(f"Backward {layer_name}: grad_output_dtype={grad_output[0].dtype}")
                        if grad_input[0] is not None:
                            print(f"  -> grad_input_dtype={grad_input[0].dtype}")
                return backward_hook
            
            # Register the hooks
            forward_hook = make_forward_hook(name)
            backward_hook = make_backward_hook(name)
            
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
            
            self.hooks.append((layer, forward_hook, backward_hook))
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for layer, forward_hook, backward_hook in self.hooks:
            layer._forward_hooks.pop(id(forward_hook), None)
            layer._backward_hooks.pop(id(backward_hook), None)
        self.hooks.clear()

def test_with_cuda_autocast():
    """Test specifically with CUDA autocast if available."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA autocast test")
        return
    
    print("\n" + "=" * 60)
    print("TESTING WITH CUDA AUTO-CASTING")
    print("=" * 60)
    
    # Create model on CUDA
    model = ToyModel(in_features=5, out_features=3).cuda()
    model.train()
    
    # Create input data on CUDA
    x = torch.randn(2, 5, dtype=torch.float32).cuda()
    target = torch.randn(2, 3, dtype=torch.float32).cuda()
    
    print(f"Input dtype: {x.dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    print()
    
    # Test with CUDA autocast
    print("--- CUDA Autocast float16 ---")
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
    
    print(f"Output dtype: {output.dtype}")
    print(f"Loss dtype: {loss.dtype}")

if __name__ == "__main__":    
    # Test with CUDA if available
    test_with_cuda_autocast()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60) 