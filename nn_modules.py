import torch
import torch.nn as nn
import math
from torch.utils.data import TensorDataset, DataLoader



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_query, d_key, d_value, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_query // num_heads # Dimension of each head's key, query, and value
        self.d_value = d_value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_query) # Query transformation
        self.W_k = nn.Linear(d_model, d_key) # Key transformation
        self.W_v = nn.Linear(d_model, d_value) # Value transformation
        self.W_o = nn.Linear(d_value, d_model) # Output transformation
        
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
        batch_size, seq_length, d = x.size()
        return x.view(batch_size, seq_length, self.num_heads, d // self.num_heads).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, d*self.num_heads)
    
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
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
    # need to allow for different sizes of q, k, v
    def __init__(self, d_model, d_query, d_key, d_value, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_query, d_key, d_value, num_heads)
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
    def __init__(self, d_model, d_query, d_key, d_value, d_ff, 
                 num_heads, num_layers, max_seq_length, dropout):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_query, d_key, d_value, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        enc_output = self.dropout(self.positional_encoding(src))
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        return enc_output


class FFPredictor(nn.Module):
    # might want to use CNN instead of FF
    # need to try with more layers
    def __init__(self, input_size, layer_sizes, dropout):
        super(FFPredictor, self).__init__()
        
        # Create the layer dimensions list
        all_layer_sizes = [input_size] + layer_sizes
        
        # Create the linear layers dynamically
        self.layers = nn.ModuleList([
            nn.Linear(all_layer_sizes[i], all_layer_sizes[i + 1])
            for i in range(len(all_layer_sizes) - 1)
        ])
        # self.norms = nn.ModuleList([
        #     nn.LayerNorm(all_layer_sizes[i+1]) 
        #     for i in range(len(all_layer_sizes) - 1)
        # ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Process through all layers except the last one with ReLU activation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU to all layers except the last one
            if i < len(self.layers) - 1:
                x = self.relu(x)
                # x = self.dropout(x)
                # x = self.norms[i](x)
        return x
     
     
class TransformerPredictor(nn.Module):
    # d_model is the dimension of the model's embedding
    # d_ff is the dimension of the feedforward network in the encoder
    # num_heads is the number of attention heads
    # num_layers is the number of layers in the encoder
    # max_seq_length is the maximum sequence length
    # dropout is the dropout rate
    # layer_sizes is a list of hidden layer sizes and output size for the predictor
    def __init__(self, d_model, d_query, d_key, d_value, d_ff, num_heads, num_layers, 
                 max_seq_length, dropout, pred_layer_sizes):
        super(TransformerPredictor, self).__init__()
        self.encoder = TransformerEncoder(d_model, d_query, d_key, d_value, d_ff, 
                                          num_heads, num_layers, max_seq_length, 
                                          dropout)
        
        # Calculate input size for the FFPredictor
        input_size = d_model * max_seq_length
        self.ff_predictor = FFPredictor(input_size, pred_layer_sizes, dropout)
        
    def forward(self, src):
        enc_output = self.encoder(src)
        # reshape to flatten for ff predictor
        flattened_output = torch.reshape(
            enc_output, 
            (enc_output.shape[0], enc_output.shape[1] * enc_output.shape[2])
        )
        pred_output = self.ff_predictor(flattened_output)
        return pred_output