import torch
import torch.nn as nn
import math
import pandas as pd
import torchinfo
import utils
from torch.utils.data import TensorDataset, DataLoader
import nn_modules



def train_test_split(x_tensor, y_tensor, test_size=0.2):
    indices = torch.randperm(len(x_tensor))
    split_idx = int(len(x_tensor) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return (
        x_tensor[train_indices], y_tensor[train_indices], 
        x_tensor[test_indices], y_tensor[test_indices]
    )

    
def train_model(model, tokens, y_true, epochs, lr, batch_size=128):
    model.train()
    
    (train_tokens, train_y_true, 
     validation_tokens, validation_y_true) = train_test_split(tokens, y_true, test_size=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train in normalized space
    train_y_true, max_abs_vals = utils.normalize(train_y_true)
    
    # Create DataLoader for automatic batching and shuffling
    train_dataset = TensorDataset(train_tokens, train_y_true)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    loss_fn = utils.wMAE
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_tokens, batch_y_true in train_dataloader:
            batch_y_true = batch_y_true * max_abs_vals  # undo normalization
            
            optimizer.zero_grad()
            batch_y_pred = model(batch_tokens) * max_abs_vals  # undo normalization
            loss = loss_fn(batch_y_pred, batch_y_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        
        # Validation
        with torch.no_grad():
            validation_y_pred = model(validation_tokens) * max_abs_vals # undo normalization
            validation_wMAE = utils.wMAE(validation_y_pred, validation_y_true)
        
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}, "
            f"Validation wMAE: {validation_wMAE.item():.6f}"
        )
    return


if __name__ == "__main__":
    
    # fixed?
    num_heads = 2
    num_layers = 4
    
    # tunable transformer hyperparameters
    d_query = 128
    d_key = d_query
    d_value = 128
    
    # tunable predictor hyperparameter d_pred
    # pred here refers to the FF predictor network after the transformer encoder
    d_pred = 128 
    num_pred_layers = 4
    d_out = 5 # fixed - number of output features ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    pred_layer_sizes = [d_pred] * num_pred_layers + [d_out]
    
    # fixed hyperparameters
    dropout = 0.1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # need to add the remaining data
    data = pd.read_csv("train.csv")
    
    y_true = torch.tensor(data[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()).to(device)
    tokens, d_model, max_seq_length = utils.character_tokenizer(data['SMILES'])
    tokens = tokens.to(device)
    d_ff = 4 * d_model # taken from original paper
    
    transformer_predictor = nn_modules.TransformerPredictor(
        d_model, d_query, d_key, d_value, d_ff, num_heads, num_layers, \
        max_seq_length, dropout, pred_layer_sizes
    ).to(device)
    
    # torchinfo.summary(transformer_predictor, input_data=tokens[:10], device=device)
    
    # any higher batch size causes 0 loss sometimes? 
    train_model(transformer_predictor, tokens, y_true, epochs=100, lr=0.0001, batch_size=128)
    