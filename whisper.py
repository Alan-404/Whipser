import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable

class Whisper(nn.Module):
    def __init__(self, token_size: int, n_mel_channels: int, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.encoder = Encoder(n_mel_channels ,n, d_model, heads, d_ff, activation, dropout_rate, eps)
        self.decoder = Decoder(token_size ,n, d_model, heads, d_ff, activation, dropout_rate, eps)

    def forward(self, spectrogram: torch.Tensor, text: torch.Tensor):
        padding_mask = None
        look_ahead_mask = None
        if self.training:
            padding_mask = generate_padding_mask(spectrogram)
            look_ahead_mask = generate_look_ahead_mask(text)
        encoder_output = self.encoder(spectrogram, padding_mask)
        output = self.decoder(text, encoder_output, look_ahead_mask, padding_mask)
        return output

class Decoder(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.learned_positional_encoder = LearnedPositionalEncoding(token_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, activation, dropout_rate, eps) for _ in range(n)])
        self.classifier = nn.Linear(in_features=d_model, out_features=token_size)
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: Union[torch.Tensor, None], padding_mask: Union[torch.Tensor, None]) -> torch.Tensor:
        x = self.learned_positional_encoder(x)
        for layer in self.layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)
        x = F.gelu(x)
        x = self.classifier(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_mel_channels: int, n: int, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.mel_extractor = MelExtractor(n_mel_channels, d_model)
        self.sinusoidal_position_encoder = PositionalEncoding()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, activation, dropout_rate, eps) for _ in range(n)])
    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        x = self.mel_extractor(x)
        x = self.sinusoidal_position_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class MelExtractor(nn.Module):
    def __init__(self, n_mel_channels: int, d_model: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=n_mel_channels, out_channels=d_model, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1)

        self.dropout_rate = dropout_rate
    def forward(self, x: torch.Tensor):
        x = self.conv_1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.gelu(x)
        x = self.conv_2(x)
        x = F.gelu(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        self.dropout_rate = dropout_rate

        assert self.d_model % self.heads == 0

        self.head_samples = self.d_model//self.heads

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        dk = torch.tensor(k.size(-1))
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/torch.sqrt(dk)

        if mask:
            attention_scores += mask*(-1e15)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_weights, v)

        return attention_context
    
    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ctx, embedding_dim = x.size()

        assert embedding_dim == self.d_model

        x = x.reshape((batch_size, n_ctx, self.heads, self.head_samples))
        x = x.permute((0, 2, 1, 3))

        return x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        batch_size, n_ctx, _ = q.size()

        qw = F.dropout(self.linear_q(q), p=self.dropout_rate, training=self.training)
        kw = F.dropout(self.linear_k(k), p=self.dropout_rate, training=self.training)
        vw = F.dropout(self.linear_v(v), p=self.dropout_rate, training=self.training)

        q_heads = self.split_head(qw)
        k_heads = self.split_head(kw)
        v_heads = self.split_head(vw)

        attention_context = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_context = attention_context.permute((0, 2, 1, 3))
        attention_context = attention_context.reshape((batch_size, n_ctx, self.d_model))

        attention_context = self.linear_output(attention_context)
        return attention_context

class PositonWiseFeedForwardNetworks(nn.Module):
    def __init__(self, d_ff: int, d_model: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float = 0.1) -> None:
        super().__init__()      
        self.hidden_layer = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = activation
        self.output_layer = nn.Linear(in_features=d_ff, out_features=d_model)

        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.norm_layer = nn.LayerNorm(normalized_shape=d_model, eps=eps)

    def forward(self, x: torch.Tensor, pre_x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_layer(x)
        x += pre_x
        x = self.norm_layer(x)
        return x
        

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.attention_layer = MultiHeadAttention(heads, d_model, dropout_rate)
        self.ffn = PositonWiseFeedForwardNetworks(d_ff, d_model, activation, dropout_rate)

        self.residual_1 = ResidualConnection(d_model, dropout_rate, eps)
        self.residual_2 = ResidualConnection(d_model, dropout_rate, eps)

    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        # sublayer 1
        attention_output = self.attention_layer(x, x, x, mask)
        attention_output = self.residual_1(attention_output, x)

        # sublayer 2
        ffn_output = self.ffn(attention_output)
        ffn_output = self.residual_2(ffn_output, attention_output)

        return ffn_output

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, activation: Callable[[torch.Tensor], torch.Tensor], dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.local_attention = MultiHeadAttention(heads, d_model, dropout_rate)
        self.global_attention = MultiHeadAttention(heads, d_model, dropout_rate)
        self.ffn = PositonWiseFeedForwardNetworks(d_ff, d_model, activation, dropout_rate)

        self.residual_1 = ResidualConnection(d_model, dropout_rate, eps)
        self.residual_2 = ResidualConnection(d_model, dropout_rate, eps)
        self.residual_3 = ResidualConnection(d_model, dropout_rate, eps)


    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: Union[torch.Tensor, None], padding_mask: Union[torch.Tensor, None]) -> torch.Tensor:
        # sublayer 1
        local_attention_output = self.local_attention(x, x, x, look_ahead_mask)
        local_attention_output = self.residual_1(local_attention_output, x)

        # sublayer 2
        global_attention_output = self.global_attention(local_attention_output, encoder_output, encoder_output, padding_mask)
        global_attention_output = self.residual_2(global_attention_output, local_attention_output)

        # sublayer 3
        ffn_output = self.ffn(global_attention_output)
        ffn_output = self.residual_3(ffn_output, global_attention_output)
        
        return ffn_output

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, token_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.positional_encoder = PositionalEncoding()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        x = self.positional_encoder(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def __encode_ctx(self, n_ctx: int) -> torch.Tensor:
        pos = torch.arange(n_ctx)
        pos = pos.unsqueeze(-1)
        return pos.type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int) -> torch.Tensor:
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1/(torch.pow(10000, angles/embedding_dim))
        angles = angles.unsqueeze(0)
        return angles
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.__encode_ctx(x.size(1))
        angles = self.__encode_embedding(x.size(2))
        
        pos_angles = torch.matmul(pos, angles)
        pos_angles[0::2] = torch.sin(pos_angles[0::2])
        pos_angles[1::2] = torch.cos(pos_angles[1::2])

        pos_angles = pos_angles.unsqueeze(0)
        x += pos_angles.to(x.device)
        return x
    

def generate_padding_mask(tensor: torch.Tensor)-> torch.Tensor:
    return torch.Tensor(tensor == 0).type(torch.int64)[:, None, None, :]

def __generate_look_ahead_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_look_ahead_mask(tensor: torch.Tensor) -> torch.Tensor:
    padding_mask = generate_padding_mask(tensor)

    look_ahead_mask = __generate_look_ahead_mask(tensor.size(1)).to(tensor.device)


    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask)

    return look_ahead_mask