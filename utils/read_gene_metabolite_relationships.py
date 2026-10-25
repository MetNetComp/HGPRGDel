import os
import pandas as pd

def read_gene_metabolite_relationships(folder_path):
    relationships = {}

    # Iterate over files in the given folder
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            
            # Check if required columns are present
            if 'Gene' not in df.columns or 'Deleted' not in df.columns:
                raise ValueError(f"File {file} does not contain required columns 'Gene' and 'Deleted'")
            
            # Extract metabolite name from the filename
            meta_name = file.split('.')[0]
            
            # Extract gene names and deletion status
            gene_names = df['Gene'].values
            labels = df['Deleted'].values
            
            # Store the relationship in the dictionary
            relationships[meta_name] = (gene_names, labels)
    
    return relationships