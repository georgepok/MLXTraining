def count_parameters_mlx(model):
    """Count parameters in MLX model"""
    total = 0
    trainable = 0
    
    def count_params(module):
        nonlocal total, trainable
        
        # Count parameters in current module
        if hasattr(module, 'parameters'):
            for param in module.parameters():
                if hasattr(param, 'size'):
                    param_count = param.size
                    total += param_count
                    # In MLX, we consider LoRA parameters as trainable
                    if hasattr(module, 'lora_A') and (param is module.lora_A or param is module.lora_B):
                        trainable += param_count
                    elif hasattr(module, 'weight') and param is module.weight and not hasattr(module, '_frozen_weight'):
                        trainable += param_count
                    elif hasattr(module, 'bias') and param is module.bias and not hasattr(module, '_frozen_bias'):
                        trainable += param_count
        
        # Recursively count in child modules
        if hasattr(module, 'children'):
            for child in module.children():
                count_params(child)
        
        # Handle lists of modules (like transformer blocks)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, 'parameters'):
                        count_params(item)
    
    count_params(model)
    return total, trainable