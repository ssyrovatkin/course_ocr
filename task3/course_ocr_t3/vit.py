##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom
from course_ocr_t3.transformer import MultiHeadAttention


# Utils
class Transpose(nn.Module):
    def __init__(self, d0, d1): 
        super(Transpose, self).__init__()
        self.d0, self.d1 = d0, d1

    def forward(self, x):
        return x.transpose(self.d0, self.d1)


##################################################
# ViT Transformer Encoder Layer
##################################################

class ViTransformerEncoderLayer(nn.Module):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale, Dosovitskiy et al, 2020.
    https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, h_dim, num_heads, d_ff=2048, dropout=0.0):
        super(ViTransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim, num_heads)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, h_dim)
        )

    def forward(self, x, mask=None):
        x_ = self.norm1(x)
        x = self.mha(x_, x_, x_, mask=mask) + x
        x_ = self.norm2(x)
        x = self.ffn(x_) + x
        return x


##################################################
# Vit Transformer Encoder
##################################################

class ViTransformerEncoder(nn.Module):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale, Dosovitskiy et al, 2020.
    https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, num_layers, h_dim, num_heads, d_ff=2048, 
                 max_time_steps=None, use_clf_token=False, dropout=0.0, dropout_emb=0.0):
        super(ViTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ViTransformerEncoderLayer(h_dim, num_heads, d_ff=d_ff, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.pos_emb = nn.Embedding(max_time_steps, h_dim)
        self.use_clf_token = use_clf_token
        if self.use_clf_token:
            self.clf_token = nn.Parameter(torch.randn(1, h_dim))
        self.dropout_emb = nn.Dropout(dropout_emb)

    def forward(self, x, mask=None):
        if self.use_clf_token:
            clf_token = self.clf_token.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([clf_token, x], 1)
            if mask is not None:
                raise Exception('Error. clf_token with mask is not supported.')
        embs = self.pos_emb.weight[:x.shape[1]]
        x += embs
        x = self.dropout_emb(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


##################################################
# Visual Transformer (ViT)
##################################################

class ViT(nn.Module):
    """
    An Image is Worth 16x16 Words: Transformers for Image Recognition At Scale, Dosovitskiy et al, 2020.
    https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, img_size, patch_size, num_layers, h_dim, num_heads, num_classes, num_tokens, 
                 d_ff=2048, max_time_steps=None, use_clf_token=True, dropout=0.0, dropout_emb=0.0):
        super(ViT, self).__init__()
        self.proc = nn.Sequential(
            nn.Unfold((patch_size, patch_size), 
                      stride=(patch_size, patch_size)),
            Transpose(1, 2),
            nn.Linear(3 * patch_size * patch_size, h_dim),
        )
        self.enc = ViTransformerEncoder(num_layers, h_dim, num_heads, 
                                         d_ff=d_ff, 
                                         max_time_steps=max_time_steps, 
                                         use_clf_token=use_clf_token, dropout=dropout, dropout_emb=dropout_emb)
        self.mlp = nn.Linear(h_dim, num_tokens*num_classes)
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        

    def forward(self, x):
        x = self.proc(x)
        x = self.enc(x)
        x = x.mean(1)
        b, _ = x.shape
        x = self.mlp(x)
        return x.reshape((b, self.num_classes, self.num_tokens))