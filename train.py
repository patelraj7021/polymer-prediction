import torch
import torch.nn as nn
import math
import pandas as pd
import torchinfo
import utils
from torch.utils.data import TensorDataset, DataLoader
import nn_modules
import wandb



def train_test_split(x_tensor, y_tensor, test_size=0.2):
    indices = torch.randperm(len(x_tensor))
    split_idx = int(len(x_tensor) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return (
        x_tensor[train_indices], y_tensor[train_indices], 
        x_tensor[test_indices], y_tensor[test_indices]
    )

    
def train_model(model, train_dataloader, validation_dataloader, optimizer, loss_fn, config):
    wandb.watch(model, log="all", log_freq=100, criterion=loss_fn)
    
    for epoch in range(config['epochs']):
        
        # Train
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_tokens, batch_y_true in train_dataloader:
            # might need to train in normalized space?
            batch_y_true = batch_y_true * config['max_abs_vals']  # undo normalization
            
            optimizer.zero_grad()
            batch_y_pred = model(batch_tokens) * config['max_abs_vals']  # undo normalization
            loss = loss_fn(batch_y_pred, batch_y_true)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            validation_losses = []
            for val_tokens, val_y_true in validation_dataloader:
                val_y_pred = model(val_tokens) * config['max_abs_vals']  # undo normalization
                val_loss = loss_fn(val_y_pred, val_y_true)
                validation_losses.append(val_loss.item())
            
            avg_validation_loss = sum(validation_losses) / len(validation_losses)
        
        if avg_validation_loss < 0.00001 or avg_train_loss < 0.00001:
            # this indicates something broke
            break
        
        wandb.log({
            "train_loss": avg_train_loss,
            "validation_wMAE": avg_validation_loss
        }, step=epoch+1)
        
        print(
            f"Epoch {epoch+1}/{config['epochs']}, Training Loss: {avg_train_loss:.6f}, "
            f"Validation wMAE: {avg_validation_loss:.6f}"
        )
        
    return


def make(config):
    # Load data
    y_true = torch.load("y_true.pt").to(config['device'])
    tokens = torch.load("tokens.pt").to(config['device'])
    d_model = tokens.shape[-1]
    max_seq_length = tokens.shape[-2]
    d_ff = 4 * d_model  # taken from original paper
    
    # Create model
    pred_layer_sizes = [config['d_pred']] * config['num_pred_layers'] + [config['d_out']]
    model = nn_modules.TransformerPredictor(
        d_model, config['d_query'], config['d_key'], config['d_value'], d_ff, 
        config['num_heads'], config['num_layers'], max_seq_length, 
        config['dropout'], pred_layer_sizes
    ).to(config['device'])
    
    # Split data
    (train_tokens, train_y_true, 
     validation_tokens, validation_y_true) = train_test_split(tokens, y_true, test_size=0.2)
    
    # Normalize training data
    train_y_true_normalized, max_abs_vals = utils.normalize(train_y_true)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_tokens, train_y_true_normalized)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    validation_dataset = TensorDataset(validation_tokens, validation_y_true)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = utils.wMAE
    
    # Store normalization values in config for later use
    config['max_abs_vals'] = max_abs_vals
    
    return model, train_dataloader, validation_dataloader, optimizer, loss_fn, config


if __name__ == "__main__":
    
    # Create configuration dictionary
    config = {
        # Fixed hyperparameters
        'd_out': 5,  # fixed - number of output features ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        'dropout': 0.1,
        
        # Tunable transformer hyperparameters
        # d_query and d_key are set to be the same here
        'num_heads': 2,
        'num_layers': 2,  # 4 might be too much?
        'd_query': 64,
        'd_value': 64,
        
        # Tunable predictor hyperparameters
        'd_pred': 64,
        'num_pred_layers': 3,
        
        # Training hyperparameters
        'lr': 0.00001,
        'batch_size': 128,  # any higher batch size causes 0 loss sometimes?
        'epochs': 10,
        
        'model_type': 'transformer_predictor',
        'loss_type': 'wMAE',
        
        # Device
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    # Set d_key equal to d_query
    config['d_key'] = config['d_query']
    
    with wandb.init(entity='patelraj7021-team', project="polymer-prediction", 
               job_type="single-training", config=config) as run:
    
        # Create model and training components
        model, train_dataloader, validation_dataloader, optimizer, loss_fn, config = make(config)
        
        # Optional: Print model summary
        # torchinfo.summary(model, input_data=next(iter(train_dataloader))[0][:10], device=config['device'])
        
        # Train the model
        train_model(model, train_dataloader, validation_dataloader, optimizer, loss_fn, config)
        
        torch.onnx.export(model, next(iter(validation_dataloader))[0], "model.onnx")
        wandb.save("model.onnx")