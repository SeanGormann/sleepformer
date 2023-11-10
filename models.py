import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fastai.optimizer import Adam
import math


CFG = {
    #'block_size': 17280,
    #'block_stride': 17280 // 16,
    #'patch_size': 12,
    'block_size': 18000,
    'block_stride': 18000 // 8,
    'patch_size': 24,

    'sleepformer_dim': 192,
    'sleepformer_num_heads': 6,
    'sleepformer_num_encoder_layers': 5,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.0,
    'sleepformer_ffn_multiplier': 1,

}


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x, training):
        """if training:
            random_shifts = torch.randint(-x.size(1), x.size(1), (x.size(0),), device=x.device)
            x_shifted = torch.stack([x[i].roll(shifts=random_shifts[i].item(), dims=0) for i in range(x.size(0))])
            return x_shifted + self.pos_encoding
        else:
            return x + self.pos_encoding"""
        return x + self.pos_encoding





class TemporalEncoding(nn.Module):
    def __init__(self, dim, max_len=1440):  # There are 1440 minutes in a day
        super(TemporalEncoding, self).__init__()
        self.dim = dim
        
        # Create a positional encoding matrix that represents the minute of the day
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, dim)
        self.register_buffer('pe', pe)


    def forward(self, x, time_indices):
        # Convert hour and minute to a single value (minute of the day)
        minutes_of_day = time_indices[:, :, 0] * 60 + time_indices[:, :, 1]
        minutes_of_day = minutes_of_day.view(x.shape[0], x.shape[1])  # Reshape to match x's batch and sequence length
        
        # Normalize to [0, 1] range
        normalized_mod = minutes_of_day / 1440.0
        
        # Map to the temporal encoding
        # Use broadcasting to expand to the necessary dimensions without explicit loops
        pe = self.pe[:, :x.shape[1], :]  # Take only the required sequence length from positional encoding
        pe = pe * normalized_mod.unsqueeze(-1)  # Broadcast normalized_mod across the dim dimension
        
        # Add the temporal encodings to the input
        x = x + pe
        
        return x







class EncoderLayer(nn.Module):
    def __init__(self, CFG):
        super(EncoderLayer, self).__init__()
        self.CFG = CFG

        self.mha = nn.MultiheadAttention(self.CFG['sleepformer_dim'], self.CFG['sleepformer_num_heads'])

        self.layer_norm1 = nn.LayerNorm(self.CFG['sleepformer_dim'])
        self.layer_norm2 = nn.LayerNorm(self.CFG['sleepformer_dim'])

        self.sequential = nn.Sequential(
            nn.Linear(self.CFG['sleepformer_dim'], self.CFG['sleepformer_dim']*self.CFG['sleepformer_ffn_multiplier']),
            nn.GELU(),
            nn.Linear(self.CFG['sleepformer_dim']*self.CFG['sleepformer_ffn_multiplier'], self.CFG['sleepformer_dim']),
            nn.Dropout(self.CFG['sleepformer_first_dropout'])
        )

        self.mha_dropout = nn.Dropout(self.CFG['sleepformer_mha_dropout'])
        self.attention_weights = None

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_output, att_weights = self.mha(x_norm, x_norm, x_norm)
        self.attention_weights = att_weights
        x = x + self.mha_dropout(attn_output)

        x = x + self.sequential(self.layer_norm2(x))

        return x


class SleepformerEncoder(nn.Module):
    def __init__(self, CFG):
        super(SleepformerEncoder, self).__init__()
        self.CFG = CFG

        self.pos_enc = PositionalEncoding(self.CFG['sleepformer_dim'], self.CFG['block_size'] // self.CFG['patch_size'])
        self.first_dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.enc_layers = nn.ModuleList([EncoderLayer(self.CFG) for _ in range(self.CFG['sleepformer_num_encoder_layers'])])
        self.first_linear = nn.Linear(self.CFG['patch_size']*2, self.CFG['sleepformer_dim'])

        self.lstm_layers = nn.ModuleList()
        for i in range(self.CFG['sleepformer_num_lstm_layers']):
            input_dim = self.CFG['sleepformer_dim'] * (2 if i > 0 else 1)
            self.lstm_layers.append(nn.LSTM(input_dim, self.CFG['sleepformer_dim'],
                                            bidirectional=True, batch_first=True, dropout=self.CFG['sleepformer_lstm_dropout'] if i < self.CFG['sleepformer_num_lstm_layers'] - 1 else 0))

    def forward(self, x, training=True):
        # Try normalise x maybe? Data should already be normalized at this point though
        x = self.first_linear(x)

        x = self.pos_enc(x, training=training)


        x = self.first_dropout(x)
        x = x.transpose(0, 1)

        #clipped_mask = ~clipped_mask
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        x = x.transpose(0, 1)

        if self.lstm_layers is not None:
            #lengths = mask.sum(dim=1)
            #x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            for lstm in self.lstm_layers:
                #x_packed, _ = lstm(x_packed)
                x, _ = lstm(x)
            #x, _ = pad_packed_sequence(x_packed, batch_first=True)

        return x


class SleepFormer(nn.Module):
    def __init__(self, cfg):
        super(SleepFormer, self).__init__()
 
        self.CFG = cfg
        self.encoder = SleepformerEncoder(self.CFG)
        last_dim = self.CFG['sleepformer_dim'] * (2 if self.CFG['sleepformer_num_lstm_layers'] > 0 else 1)
        self.last_linear = nn.Linear(last_dim, 1)
        self.dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x["X"]
        encoded_seq = self.encoder(x)
        encoded_seq = self.dropout(encoded_seq)

        output = self.last_linear(encoded_seq)
        #output = self.sigmoid(output)

        return output








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class RSleepformerEncoder(nn.Module):
    def __init__(self, CFG):
        super(RSleepformerEncoder, self).__init__()
        self.CFG = CFG
        self.pos_enc = PositionalEncoding(self.CFG['sleepformer_dim'], self.CFG['block_size'] // self.CFG['patch_size'])        
        self.first_dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.enc_layers = nn.ModuleList([EncoderLayer(CFG) for _ in range(self.CFG['sleepformer_num_encoder_layers'])])
        self.first_linear = nn.Linear(self.CFG['patch_size']*2, self.CFG['sleepformer_dim'])

        # Recycling components
        self.recycle_iterations = self.CFG.get('recycle_iterations', 3)
        self.recycle_emb = nn.Linear(1, self.CFG['sleepformer_dim'])  # Embedding for recycled output
        self.recycle_norm = nn.LayerNorm(self.CFG['sleepformer_dim'])  # Layer normalization for recycling embedding

        self.last_dim = self.CFG['sleepformer_dim'] * (2 if self.CFG['sleepformer_num_lstm_layers'] > 0 else 1)
        self.last_linear = nn.Linear(self.last_dim, 1)
        
        self.lstm_layers = nn.ModuleList()
        lstm_input_dim = self.CFG['sleepformer_dim']
        for i in range(self.CFG['sleepformer_num_lstm_layers']):
            self.lstm_layers.append(nn.LSTM(
                lstm_input_dim, 
                self.CFG['sleepformer_dim'],
                bidirectional=True, 
                batch_first=True, 
                dropout=self.CFG['sleepformer_lstm_dropout'] if i < self.CFG['sleepformer_num_lstm_layers'] - 1 else 0
            ))
            lstm_input_dim = self.CFG['sleepformer_dim'] * 2  # Double the input dimension for the next LSTM layer if bidirectional


    def forward(self, x, training=True):

        # Try normalise x maybe? Data should already be normalized at this point though
        x = self.first_linear(x)

        x = self.pos_enc(x, training=training)
        
        if self.training:
            # Random number of recycling iterations during training
            N = torch.randint(1, self.recycle_iterations + 1, (1,)).item()
        else:
            # Fixed number of recycling iterations during evaluation
            N = self.CFG['recycle_iterations']

        x = self.first_dropout(x)


        # Initialize recycled output with zeros
        #recycled_output = torch.zeros(x.size(), device=x.device)
        recycled_output = torch.zeros((x.size(0), x.size(1), 1), device=device)
        
        for _ in range(N):
            # Add recycled output to input
            recycled_output = self.recycle_emb(recycled_output)
            recycled_output = self.recycle_norm(recycled_output)
            x_emb = x + recycled_output

            x_emb = x_emb.transpose(0, 1)
            for enc_layer in self.enc_layers:
                x_emb = enc_layer(x_emb)
            x_emb = x_emb.transpose(0, 1)

            # Check here if the LSTM layers list is not None and not empty
            if self.lstm_layers:
                # Loop through the LSTM layers with an index
                for i, lstm in enumerate(self.lstm_layers):
                    recycled_output, _ = lstm(x_emb)
                    x_emb = recycled_output
            
            recycled_output = self.last_linear(recycled_output)

        return recycled_output


class RSleepFormer(nn.Module):
    def __init__(self, CFG):
        super(RSleepFormer, self).__init__()
        self.CFG = CFG
        self.encoder = RSleepformerEncoder(CFG)
        #last_dim = CFG['sleepformer_dim'] * (2 if CFG['sleepformer_num_lstm_layers'] > 0 else 1)
        #self.last_linear = nn.Linear(last_dim, 1)
        self.dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x["X"]
        encoded_seq = self.encoder(x)
        encoded_seq = self.dropout(encoded_seq)

        #output = self.last_linear(encoded_seq)

        return encoded_seq





##~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

class EncoderLayer(nn.Module):
    def __init__(self, CFG):
        super(EncoderLayer, self).__init__()
        self.CFG = CFG

        self.mha = nn.MultiheadAttention(self.CFG['sleepformer_dim'], self.CFG['sleepformer_num_heads'])

        self.layer_norm1 = nn.LayerNorm(self.CFG['sleepformer_dim'])
        self.layer_norm2 = nn.LayerNorm(self.CFG['sleepformer_dim'])

        self.sequential = nn.Sequential(
            nn.Linear(self.CFG['sleepformer_dim'], self.CFG['sleepformer_dim']*self.CFG['sleepformer_ffn_multiplier']),
            nn.GELU(),
            nn.Linear(self.CFG['sleepformer_dim']*self.CFG['sleepformer_ffn_multiplier'], self.CFG['sleepformer_dim']),
            nn.Dropout(self.CFG['sleepformer_first_dropout'])
        )

        self.mha_dropout = nn.Dropout(self.CFG['sleepformer_mha_dropout'])
        self.attention_weights = None

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_output, att_weights = self.mha(x_norm, x_norm, x_norm)
        self.attention_weights = att_weights
        x = x + self.mha_dropout(attn_output)

        x = x + self.sequential(self.layer_norm2(x))

        return x


class SleepformerEncoder(nn.Module):
    def __init__(self, CFG):
        super(SleepformerEncoder, self).__init__()
        self.CFG = CFG

        self.pos_enc = PositionalEncoding(self.CFG['sleepformer_dim'], self.CFG['block_size'] // self.CFG['patch_size'])
        self.first_dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.enc_layers = nn.ModuleList([EncoderLayer(self.CFG) for _ in range(self.CFG['sleepformer_num_encoder_layers'])])
        self.first_linear = nn.Linear(self.CFG['patch_size']*2, self.CFG['sleepformer_dim'])

        self.lstm_layers = nn.ModuleList()
        for i in range(self.CFG['sleepformer_num_lstm_layers']):
            input_dim = self.CFG['sleepformer_dim'] * (2 if i > 0 else 1)
            self.lstm_layers.append(nn.GRU(input_dim, self.CFG['sleepformer_dim'],
                                            bidirectional=True, batch_first=True, dropout=self.CFG['sleepformer_lstm_dropout'] if i < self.CFG['sleepformer_num_lstm_layers'] - 1 else 0))

    def forward(self, x, training=True):
        # Try normalise x maybe? Data should already be normalized at this point though
        x = self.first_linear(x)

        x = self.pos_enc(x, training=training)
        


        x = self.first_dropout(x)
        x = x.transpose(0, 1)

        #clipped_mask = ~clipped_mask
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        x = x.transpose(0, 1)

        if self.lstm_layers is not None:
            #lengths = mask.sum(dim=1)
            #x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            for lstm in self.lstm_layers:
                #x_packed, _ = lstm(x_packed)
                x, _ = lstm(x)
            #x, _ = pad_packed_sequence(x_packed, batch_first=True)

        return x


class SleepFormer(nn.Module):
    def __init__(self, cfg):
        super(SleepFormer, self).__init__()
 
        self.CFG = cfg
        self.encoder = SleepformerEncoder(self.CFG)
        last_dim = self.CFG['sleepformer_dim'] * (2 if self.CFG['sleepformer_num_lstm_layers'] > 0 else 1)
        self.last_linear = nn.Linear(last_dim, 2)
        self.dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x["X"]
        encoded_seq = self.encoder(x)
        encoded_seq = self.dropout(encoded_seq)

        output = self.last_linear(encoded_seq)
        #output = self.sigmoid(output)

        return output

