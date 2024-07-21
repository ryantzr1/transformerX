import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder, DecoderBlock
from input_embedding import InputEmbedding
from projection_layer import ProjectionLayer
from encoder_block import EncoderBlock
from feed_forward import FeedForwardLayer
from multi_head_attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src_embedded = self.src_pos(src)
        encoded = self.encoder(src_embedded, src_mask)
        return encoded

    def decode(self, tgt, encoded, src_mask, tgt_mask):
        tgt_embedded = self.tgt_embedding(tgt) 
        tgt_embedded = self.target_pos(tgt_embedded)
        decoded = self.decoder(tgt_embedded, encoded, src_mask, tgt_mask)
        return decoded
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, layers=6, src_seq_length=350, tgt_seq_length=350):
    # Create input embeddings
    src_embedding = InputEmbedding(src_vocab_size, d_model)
    tgt_embedding = InputEmbedding(tgt_vocab_size, d_model)
    
    # Create positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_length)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_length)
    
    # Create feedforward layer
    feed_forward = FeedForwardLayer(d_model, 2048)

    # Encoder self-attention
    encoder_self_attention = MultiHeadAttention(d_model, nhead)

    # Create encoder blocks
    encoder_blocks = nn.ModuleList([EncoderBlock(d_model, encoder_self_attention, feed_forward) for _ in range(layers)])
    
    # Create encoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Decoder self-attention
    decoder_self_attention = MultiHeadAttention(d_model, nhead)

    # Decoder cross-attention
    decoder_cross_attention = MultiHeadAttention(d_model, nhead)

    # Create decoder blocks
    decoder_blocks = nn.ModuleList([DecoderBlock(d_model, decoder_self_attention, decoder_cross_attention, feed_forward) for _ in range(layers)])

    # Create decoder
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Build transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    return transformer
