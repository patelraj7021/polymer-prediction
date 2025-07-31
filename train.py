import torch
import torch.nn as nn
import math
import pandas as pd
import torchinfo
import utils
from torch.utils.data import TensorDataset, DataLoader
import nn_modules
import wandb


class Config:
    def __init__(self, d_out, d_qkv, d_pred, num_heads, num_layers, num_pred_layers,
                 dropout, lr, batch_size, epochs, optimizer, device, model_type, loss_type):
        self.d_out = d_out
        self.dropout = dropout
        self.d_qkv = d_qkv
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_pred = d_pred
        self.num_pred_layers = num_pred_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.loss_type = loss_type


def train_test_split(x_tensor, y_tensor, test_size=0.2):
    indices = torch.randperm(len(x_tensor))
    split_idx = int(len(x_tensor) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return (
        x_tensor[train_indices], y_tensor[train_indices], 
        x_tensor[test_indices], y_tensor[test_indices]
    )

    
def train_model(model, train_dataloader, validation_dataloader, optimizer, loss_fn, config, max_abs_vals):
    # wandb.watch(model, log="all", log_freq=100, criterion=loss_fn)
    broke = False
    for epoch in range(config.epochs):
        
        # Train
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_tokens, batch_y_true in train_dataloader:
            # since we're training in normalized space
            batch_y_true = batch_y_true * max_abs_vals  # undo normalization
            
            nan_mask = torch.isnan(batch_y_true)
            
            optimizer.zero_grad()
            batch_y_pred = model(batch_tokens) * max_abs_vals  # undo normalization
            # helps avoid numerical instability when calculating gradients
            # doesn't affect loss calculation because nan counts
            # are based on batch_y_true
            batch_y_pred[nan_mask] = 0
            loss = loss_fn(batch_y_pred, batch_y_true)
            loss.backward()
            optimizer.step()
            
            # if any(torch.sum(~torch.isnan(batch_y_true), dim=0) == 0):
            #     print(num_batches)
            #     print(epoch)
            #     print('flag')
            #     print(loss.item())
            #     print(batch_y_true)
            #     print(batch_y_pred)
            
            if loss.item() < 0.00001:
                # indicates something broke
                print(num_batches)
                print(epoch)
                print(torch.sum(~torch.isnan(batch_y_true), dim=0))
                print(loss.item())
                print(batch_y_true)
                print(batch_y_pred)
                broke = True
                break
            
            total_loss += loss.item()
            num_batches += 1
        
        if broke:
            # to avoid wasting time on cases where the model breaks
            # indicated by loss being 0
            break
        else:
            # Calculate average training loss
            avg_train_loss = total_loss / num_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_losses = 0
                num_batches = 0
                for val_tokens, val_y_true in validation_dataloader:
                    val_y_pred = model(val_tokens) * max_abs_vals  # undo normalization
                    val_loss = loss_fn(val_y_pred, val_y_true)
                    val_losses += val_loss.item()
                    num_batches += 1

                avg_validation_loss = val_losses / num_batches
            
            if avg_validation_loss < 0.00001 or avg_train_loss < 0.00001:
                # this indicates something broke
                break
            
            wandb.log({
                "train_loss": avg_train_loss,
                "validation_wMAE": avg_validation_loss
            }, step=epoch+1)
            
            
            
            print(
                f"Epoch {epoch+1}/{config.epochs}, Training Loss: {avg_train_loss:.6f}, "
                f"Validation wMAE: {avg_validation_loss:.6f}"
            )
        
    return


def make(config):
    # Load data
    device = torch.device(config.device)
    y_true = torch.load("y_true.pt").to(device)
    tokens = torch.load("tokens.pt").to(device)
    d_model = tokens.shape[-1]
    max_seq_length = tokens.shape[-2]
    d_ff = 4 * d_model  # taken from original paper
    
    # Create model
    pred_layer_sizes = [config.d_pred] * config.num_pred_layers + [config.d_out]
    model = nn_modules.TransformerPredictor(
        d_model, config.d_qkv, config.d_qkv, config.d_qkv, d_ff, 
        config.num_heads, config.num_layers, max_seq_length, 
        config.dropout, pred_layer_sizes
    ).to(device)
    
    # Split data
    (train_tokens, train_y_true, 
     validation_tokens, validation_y_true) = train_test_split(tokens, y_true, test_size=0.2)
    
    # Normalize training data
    train_y_true_normalized, max_abs_vals = utils.normalize(train_y_true)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_tokens, train_y_true_normalized)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    validation_dataset = TensorDataset(validation_tokens, validation_y_true)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create optimizer and loss function
    optimizer_dict = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'RMSprop': torch.optim.RMSprop,
        'SGD': torch.optim.SGD
    }
    optimizer = optimizer_dict[config.optimizer](model.parameters(), lr=config.lr)
    loss_fn = utils.wMAE
    
    return model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals


def sweep_wrapper(config=None):
    
    with wandb.init(job_type="sweep-training", config=config) as run:
        
        config = wandb.config
        
        # Create model and training components
        model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals = make(config)
        
        # Train the model
        train_model(model, train_dataloader, validation_dataloader, 
                    optimizer, loss_fn, config, max_abs_vals)
        
        torch.onnx.export(model, next(iter(validation_dataloader))[0], "model.onnx")
        wandb.save("model.onnx")


def test_config():
    # for checking memory usage
    config = Config(
        d_out=5,
        d_qkv=128,
        d_pred=256,
        num_heads=4,
        num_layers=4,
        num_pred_layers=4,
        dropout=0.05,
        lr=0.0001,
        batch_size=256,
        epochs=10,
        optimizer='AdamW',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type='transformer_predictor',
        loss_type='wMAE'
    )
    model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals = make(config)
    torchinfo.summary(model, input_data=next(iter(validation_dataloader))[0])
    return
    


if __name__ == "__main__":
    
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'validation_wMAE',
            'goal': 'minimize'
        },
        'parameters': {
            'd_out': {
                'value': 5
            },
            'd_qkv': { # d_query = d_key = d_value
                'value': 64
            },
            'd_pred': {
                'value': 64
            },
            'num_heads': {
                'value': 2
            },
            'num_layers': {
                'value': 2
            },
            'num_pred_layers': {
                'value': 2
            },
            'dropout': {
                'value': 0.05
            },
            'lr': {
                'value': 0.00001
            },
            'batch_size': {
                'value': 32
            },
            'epochs': {
                'value': 200
            },
            'optimizer': {
                'value': 'AdamW'
            },
            'device': {
                'value': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'model_type': {
                'value': 'transformer_predictor'
            },
            'loss_type': {
                'value': 'wMAE'
            }
        }
    }
    # need to optimize positional encoding, try CNN arch
    # need to vary adamw hyperparameters
    # sweep_id = wandb.sweep(sweep_config, entity='patelraj7021-team', project="polymer-prediction")
    # wandb.agent(sweep_id, function=sweep_wrapper, count=50)
    
    
    test_config()
