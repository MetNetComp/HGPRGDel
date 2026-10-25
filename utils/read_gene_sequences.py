import os
import pandas as pd

def read_gene_sequences(folder_path):  # Accept folder_path as argument
    gene_sequences = []
    gene_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            gene_name = filename.replace('.csv', '')
            file_path = os.path.join(folder_path, filename)
            
            try:
                df = pd.read_csv(file_path)
                if 'AA Sequence' in df.columns and len(df) > 0:
                    sequence = df['AA Sequence'].iloc[0]
                    if sequence and isinstance(sequence, str):  # Only add non-empty sequences
                        gene_sequences.append(sequence)
                        gene_names.append(gene_name)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return gene_names, gene_sequences
