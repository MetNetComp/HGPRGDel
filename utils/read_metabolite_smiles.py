import pandas as pd

def read_metabolite_smiles(file_path):
    """Read metabolite SMILES from CSV file."""
    try:
        df = pd.read_csv(file_path)
        # A metabolite with no known structure leaves the field empty, which pandas
        # reads as NaN. Callers tokenize these strings and take max(len(...)) over
        # them, so a float would raise; keep the column textual. utils/data_module.py
        # already does the same thing.
        smiles_features = df['Smiles'].fillna('').astype(str).tolist()  # column is 'Smiles', not 'SMILES'
        metabolite_names = df['Metabolite'].astype(str).tolist()
        return metabolite_names, smiles_features
    except Exception as e:
        print(f"Error reading metabolite SMILES: {e}")
        return [], []