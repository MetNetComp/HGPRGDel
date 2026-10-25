import torch
from .DNN import DNN

def initialize_model_DNN(gene_vocab_size, smiles_vocab_size, max_gene_len, max_smiles_len, use_cpu=0):
    """
    Initializes the DNN models

    Parameters:
    - gene_vocab_size (int): Size of the gene vocabulary.
    - smiles_vocab_size (int): Size of the SMILES vocabulary.

    Returns:
    - model (CombinedModel): The combined model with GeneLSTM and MetaLSTM components.
    - device (torch.device): The device the model is loaded onto (CPU or GPU).
    """
    model = DNN(
        gene_vocab_size=gene_vocab_size,
        smiles_vocab_size=smiles_vocab_size,
        max_gene_len = max_gene_len,
        max_smiles_len = max_smiles_len)

    # Set the device
    if use_cpu == 1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if GPU is available and print device information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        print(f'Model is on: {device} ({gpu_name})')
    else:
        print('Model is on:', device)

    # Return model and device for further use
    return model, device
