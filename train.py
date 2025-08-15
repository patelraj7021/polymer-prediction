import torch
import torch.nn as nn
import math
import pandas as pd
import torchinfo
import utils
from torch.utils.data import TensorDataset, DataLoader
import nn_modules
import wandb
import pickle


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

    
def train_model(model, train_dataloader, validation_dataloader, optimizer, loss_fn, config, max_abs_vals, n_i,
                track_best=False):
    # wandb.watch(model, log="all", log_freq=100, criterion=loss_fn)
    broke = False
    last_best_loss = float('inf')
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
            loss = loss_fn(batch_y_pred, batch_y_true, n_i)
            loss.backward()
            optimizer.step()
            
            if loss.item() < 0.00001:
                # indicates something broke
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
                    val_loss = loss_fn(val_y_pred, val_y_true, n_i)
                    val_losses += val_loss.item()
                    num_batches += 1

                avg_validation_loss = val_losses / num_batches
            
            if avg_validation_loss < 0.00001 or avg_train_loss < 0.00001:
                # this indicates something broke
                break
            
            if track_best:
                if avg_validation_loss < last_best_loss:
                    last_best_loss = avg_validation_loss
                    torch.save(model.state_dict(), "best_model.pt")
            
            wandb.log({
                "train_loss": avg_train_loss,
                "validation_wMAE": avg_validation_loss
            }, step=epoch+1)
            
            
            
            print(
                f"Epoch {epoch+1}/{config.epochs}, Training Loss: {avg_train_loss:.6f}, "
                f"Validation wMAE: {avg_validation_loss:.6f}"
            )
    if track_best:
        model.load_state_dict(torch.load("best_model.pt"))
    return model


def make(config):
    # Load data
    device = torch.device(config.device)
    y_true = torch.load("y_true.pt").to(device)
    tokens = torch.load("tokens.pt").to(device)
    d_model = tokens.shape[-1]
    max_seq_length = tokens.shape[-2]
    d_ff = 4 * d_model  # taken from original paper
    
    n_i = torch.sum(~torch.isnan(y_true), dim=0)
    
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
    
    return model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals, n_i


def sweep_wrapper(config=None):
    
    with wandb.init(job_type="sweep-training", config=config) as run:
        
        config = wandb.config
        
        # Create model and training components
        model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals, n_i = make(config)
        
        # Train the model
        train_model(model, train_dataloader, validation_dataloader, 
                    optimizer, loss_fn, config, max_abs_vals, n_i)
        
        torch.onnx.export(model, next(iter(validation_dataloader))[0], "model.onnx")
        wandb.save("model.onnx")


def test_config():
    # for checking memory usage
    config = Config(
        d_out=5,
        d_qkv=256,
        d_pred=256,
        num_heads=8,
        num_layers=8,
        num_pred_layers=8,
        dropout=0.05,
        lr=0.0001,
        batch_size=128,
        epochs=10,
        optimizer='AdamW',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type='transformer_predictor',
        loss_type='wMAE'
    )
    model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals = make(config)
    torchinfo.summary(model, input_data=next(iter(validation_dataloader))[0])
    train_model(model, train_dataloader, validation_dataloader, 
                optimizer, loss_fn, config, max_abs_vals)
    return
    

def final_train(config):
    model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals = make(config)
    best_model = train_model(model, train_dataloader, validation_dataloader, 
                             optimizer, loss_fn, config, max_abs_vals, track_best=True)
    
    return best_model, max_abs_vals


def pred_test(model, max_abs_vals, test_loc, config):
    test_df = pd.read_csv(test_loc)
    
    test_df['SMILES'] = test_df['SMILES'].apply(lambda s: utils.make_smile_canonical(s))
    
    # Check if SMILES length is greater than max_seq_length
    # If so, truncate the SMILES string
    max_allowed_length = model.max_seq_length - 2
    test_df['SMILES'] = test_df['SMILES'].apply(
        lambda x: x[:max_allowed_length] if len(x) > max_allowed_length else x)
    
    with open("char_index_map.pkl", "rb") as f:
        char_index_map = pickle.load(f)
    test_tokens = utils.tokens_from_charmap(test_df['SMILES'], char_index_map, model.max_seq_length)
    test_tokens = test_tokens.to(config.device)
    
    # Create dataset and dataloader for batch processing
    test_dataset = TensorDataset(test_tokens)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Collect predictions from all batches
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch_tokens in test_dataloader:
            batch_tokens = batch_tokens[0]  # TensorDataset returns a tuple
            batch_predictions = model(batch_tokens)
            batch_predictions = batch_predictions * max_abs_vals
            all_predictions.append(batch_predictions.cpu())
    
    # Concatenate all batch predictions
    test_predictions = torch.cat(all_predictions, dim=0).detach().numpy()
    test_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = test_predictions
    return test_df


def kaggle_submission():
    config = Config(
        d_out=5,
        d_qkv=256,
        d_pred=256,
        num_heads=8,
        num_layers=8,
        num_pred_layers=8,
        dropout=0.05,
        lr=0.0001,
        batch_size=256,
        epochs=1,
        optimizer='AdamW',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type='transformer_predictor',
        loss_type='wMAE'
    )
    model, train_dataloader, validation_dataloader, optimizer, loss_fn, max_abs_vals = make(config)
    best_model = train_model(model, train_dataloader, validation_dataloader, 
                             optimizer, loss_fn, config, max_abs_vals, track_best=True)
    test_loc = '../input/neurips-open-polymer-prediction-2025/test.csv'
    test_df = pred_test(best_model, max_abs_vals, test_loc, config)
    test_df.to_csv("submission.csv", index=False)
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
                'values': [32, 64, 128, 256]
            },
            'd_pred': {
                'values': [32, 64, 128, 256]
            },
            'num_heads': {
                'values': [2, 4, 8]
            },
            'num_layers': {
                'values': [2, 4, 8]
            },
            'num_pred_layers': {
                'values': [2, 4, 8]
            },
            'dropout': {
                'value': 0.05
            },
            'lr': {
                'value': 1e-4
            },
            'batch_size': { # max at 256 due to memory constraints
                'values': [32, 64, 128]
            },
            'epochs': {
                'value': 50
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
    # need to optimize positional encoding (try relative - https://arxiv.org/pdf/2312.17044v4)
    # try CNN arch
    # need to vary adamw hyperparameters
    # need to fix wMAE? technically it's supposed to use 
    # the full training/validation set for calculating n_i, property weights
    sweep_id = wandb.sweep(sweep_config, entity='patelraj7021-team', project="polymer-prediction")
    wandb.agent(sweep_id, function=sweep_wrapper, count=50)
    
    
    # test_config()
