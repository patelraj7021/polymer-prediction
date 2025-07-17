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
    def __init__(self, d_model, max_seq_length, d_pred, d_out):
        super(FFPredictor, self).__init__()
        self.fc1 = nn.Linear(d_model * max_seq_length, d_pred)
        self.fc2 = nn.Linear(d_pred, d_out)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    
class TransformerPredictor(nn.Module):
    # d_model is the dimension of the model's embedding
    # num_heads is the number of attention heads
    # num_layers is the number of layers in the encoder
    # d_ff is the dimension of the feedforward network in the encoder
    # max_seq_length is the maximum sequence length
    # dropout is the dropout rate
    # d_pred is the dimension of the feedforward network in the predictor
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, d_pred, d_out):
        super(TransformerPredictor, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        self.ff_predictor = FFPredictor(d_model, max_seq_length, d_pred, d_out)
        
    def forward(self, src):
        enc_output = self.encoder(src)  
        pred_output = self.ff_predictor(enc_output)
        return pred_output
    

def loss_function(pred, target):
    diff = torch.abs(pred - target)
    loss = torch.nanmean(diff)
    return loss
    
def train_model(model, tokens, y_true, epochs, lr):
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(tokens)
        loss = loss_function(y_pred, y_true)
        loss.backward()
        optimizer.step()   
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    return

if __name__ == "__main__":
    d_model = 64 # 43 is the number of unique characters in the SMILES string
    num_heads = 1
    num_layers = 1
    d_ff = 64
    dropout = 0.1
    d_pred = 16
    d_out = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = pd.read_csv("train.csv")
    max_seq_length_index = data['SMILES'].str.len().idxmax()
    max_seq_length = len(data['SMILES'].iloc[max_seq_length_index])
    
    y_true = torch.tensor(data[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()).to(device)
    tokens = utils.character_tokenizer(data['SMILES'], d_model).to(device)
    
    pred_model = TransformerPredictor(d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, d_pred, d_out).to(device)
    
    torchinfo.summary(pred_model, input_data=tokens, device=device)
    
    train_model(pred_model, tokens, y_true, epochs=100, lr=0.001)
    