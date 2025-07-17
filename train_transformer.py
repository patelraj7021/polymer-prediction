import torch
import torch.nn as nn
import math
import pandas as pd
import torchinfo
import utils



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
   
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
   
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # need to do this assertion to avoid error
        assert isinstance(self.pe, torch.Tensor)
        return x + self.pe[:, :x.size(1)]
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_embedded = self.dropout(self.positional_encoding(src))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        output = torch.reshape(enc_output, (enc_output.shape[0], enc_output.shape[1] * enc_output.shape[2]))
        return output


class FFPredictor(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(FFPredictor, self).__init__()
        
        # Create the layer dimensions list
        all_layer_sizes = [input_size] + layer_sizes
        
        # Create the linear layers dynamically
        self.layers = nn.ModuleList([
            nn.Linear(all_layer_sizes[i], all_layer_sizes[i + 1])
            for i in range(len(all_layer_sizes) - 1)
        ])
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Process through all layers except the last one with ReLU activation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU to all layers except the last one
            if i < len(self.layers) - 1:
                x = self.relu(x)
        
        return x
     
     
class TransformerPredictor(nn.Module):
    # d_model is the dimension of the model's embedding
    # num_heads is the number of attention heads
    # num_layers is the number of layers in the encoder
    # d_ff is the dimension of the feedforward network in the encoder
    # max_seq_length is the maximum sequence length
    # dropout is the dropout rate
    # layer_sizes is a list of hidden layer sizes and output size for the predictor
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, layer_sizes):
        super(TransformerPredictor, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        
        # Calculate input size for the FFPredictor
        input_size = d_model * max_seq_length
        self.ff_predictor = FFPredictor(input_size, layer_sizes)
        
    def forward(self, src):
        enc_output = self.encoder(src)  
        pred_output = self.ff_predictor(enc_output)
        return pred_output
    

def loss_function(pred, target):
    # change this so that it matters how many values are nan in a row
    # also need to train in normalized space but predict and evaluate MAE in original space
    # there's a weighting function on the website
    diff = torch.abs(pred - target)
    loss = torch.nanmean(diff)
    return loss


def train_test_split(x_tensor, y_tensor, test_size=0.2):
    indices = torch.randperm(len(x_tensor))
    split_idx = int(len(x_tensor) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return x_tensor[train_indices], y_tensor[train_indices], x_tensor[test_indices], y_tensor[test_indices]

    
def train_model(model, tokens, y_true, epochs, lr):
    model.train()
    
    train_tokens, train_y_true, validation_tokens, validation_y_true = train_test_split(tokens, y_true, test_size=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        train_y_pred = model(train_tokens)
        loss = loss_function(train_y_pred, train_y_true)
        loss.backward()
        optimizer.step()
        validation_y_pred = model(validation_tokens)
        validation_loss = loss_function(validation_y_pred, validation_y_true)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {validation_loss.item()}")
    return


if __name__ == "__main__":
    d_model = 64 # 43 is the number of unique characters in the SMILES string
    num_heads = 1
    num_layers = 1
    d_ff = 128
    dropout = 0.1
    d_pred = 64
    d_out = 5 # number of output features ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = pd.read_csv("train.csv")
    max_seq_length_index = data['SMILES'].str.len().idxmax()
    max_seq_length = len(data['SMILES'].iloc[max_seq_length_index])
    
    y_true = torch.tensor(data[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()).to(device)
    y_true = utils.normalize(y_true)
    tokens = utils.character_tokenizer(data['SMILES'], d_model).to(device)
    
    pred_model = TransformerPredictor(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, [d_pred, d_out]).to(device)
    
    torchinfo.summary(pred_model, input_data=tokens, device=device)
    
    train_model(pred_model, tokens, y_true, epochs=100, lr=0.00001)
    