from torch import nn


#from models.transformer.encoder_layer import EncoderLayer
from models.transformer.layer_norm import LayerNorm
from models.transformer.multi_head_attention import MultiHeadAttention
from models.transformer.position_wise_feed_forward import PositionwiseFeedForward


class SAFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=cfg.CONST.D_MODEL,
                                                  drop_prob=cfg.CONST.DROP_PROB,
                                                  n_head=cfg.CONST.N_HEAD)
                                     for _ in range(cfg.CONST.N_LAYERS)])

    def forward(self, x, s_mask):
        for layer in self.layers:
            x = layer(x, s_mask)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        '''self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)'''

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        '''# 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)'''
        return x