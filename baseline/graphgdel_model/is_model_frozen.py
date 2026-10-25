import torch.nn as nn

def is_model_frozen(model: nn.Module) -> bool:
    """
    Checks if the model's parameters are frozen (i.e., no parameters have requires_grad=True).
    """
    return not any(param.requires_grad for param in model.parameters())