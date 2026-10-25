import torch
from .GeneLSTM import GeneLSTM
from .MetaLSTM import MetaLSTM
from .CombinedModel_DeepGDel import CombinedModel_DeepGDel
from .freeze_model import freeze_model
from .is_model_frozen import is_model_frozen

def initialize_model(gene_vocab_size, smiles_vocab_size, vocab_embedding_dim, lstm_hidden_dim, freeze_gene=False, freeze_smiles=False, use_cpu=0):
    """
    Initializes the models, optionally freezes one of them, and prepares the model for training.

    Parameters:
    - gene_vocab_size (int): Size of the gene vocabulary.
    - smiles_vocab_size (int): Size of the SMILES vocabulary.
    - vocab_embedding_dim (int): The embedding dimension for the vocabulary.
    - lstm_hidden_dim (int): The number of hidden units in the LSTM layers.
    - freeze_gene (bool): Whether to freeze the GeneLSTM model (default: False).
    - freeze_smiles (bool): Whether to freeze the MetaLSTM model (default: False).

    Returns:
    - model (CombinedModel): The combined model with GeneLSTM and MetaLSTM components.
    - device (torch.device): The device the model is loaded onto (CPU or GPU).
    """
    
    # Initialize models
    gene_lstm = GeneLSTM(gene_vocab_size, vocab_embedding_dim, lstm_hidden_dim)
    smiles_lstm = MetaLSTM(smiles_vocab_size, vocab_embedding_dim, lstm_hidden_dim)
    model = CombinedModel_DeepGDel(gene_lstm, smiles_lstm, lstm_hidden_dim)

    # Optionally freeze models
    if freeze_gene:
        freeze_model(gene_lstm)
    if freeze_smiles:
        freeze_model(smiles_lstm)

    # Check and print which model is frozen
    if is_model_frozen(gene_lstm):
        print("GeneLSTM is frozen.")
    else:
        print("GeneLSTM is trainable.")

    if is_model_frozen(smiles_lstm):
        print("MetaLSTM is frozen.")
    else:
        print("MetaLSTM is trainable.")

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