import torch.nn as nn

def freeze_model(model: nn.Module):
    """
    Freezes the parameters of the model so they will not be updated during training.
    """
    for param in model.parameters():
        param.requires_grad = False
