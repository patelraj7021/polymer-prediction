import wandb
import pandas as pd
import torch
import utils
import pickle



main_data = pd.read_csv("train.csv")
dataset_1 = pd.read_csv("dataset1.csv")
dataset_3 = pd.read_csv("dataset3.csv")
dataset_4 = pd.read_csv("dataset4.csv")

# Concatenate all datasets and handle conflicts by duplicating rows
def merge_with_row_duplication(dataframes, on='SMILES'):
    """
    Merge dataframes by duplicating rows when there are conflicting values
    instead of adding column suffixes.
    """
    # Concatenate all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Group by SMILES and handle conflicts
    result_rows = []
    
    for smiles, group in combined.groupby(on):
        if len(group) == 1:
            # No conflicts for this SMILES
            result_rows.append(group.iloc[0])
        else:
            # Multiple rows for same SMILES - check for conflicts
            # Get all unique combinations of non-null values
            unique_combinations = group.drop_duplicates()
            
            if len(unique_combinations) == 1:
                # All rows are identical, keep one
                result_rows.append(unique_combinations.iloc[0])
            else:
                # Different values exist, keep all unique combinations
                for _, row in unique_combinations.iterrows():
                    result_rows.append(row)
    
    return pd.DataFrame(result_rows).reset_index(drop=True)

# Apply the custom merge function
data = merge_with_row_duplication([main_data, dataset_1, dataset_3, dataset_4])

y_true = torch.tensor(data[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy())
torch.save(y_true, "y_true.pt")

data['SMILES'] = data['SMILES'].apply(lambda s: utils.make_smile_canonical(s))
tokens, char_index_map = utils.character_tokenizer(data['SMILES'])
torch.save(tokens, "tokens.pt")
with open("char_index_map.pkl", "wb") as f:
    pickle.dump(char_index_map, f)

run = wandb.init(entity='patelraj7021-team', project="polymer-prediction", job_type="upload-data")

artifact = wandb.Artifact("polymer-data", type="dataset")
artifact.add_file("y_true.pt")
artifact.add_file("tokens.pt")
artifact.add_file("char_index_map.pkl")
run.log_artifact(artifact)
run.finish()