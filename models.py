import torch
import torch.nn as nn
import torch.nn.functional as F
import math



    ## AttentionTorch implements the multi-head attention mechanism
class AttentionTorch(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

        self.project_out = not (heads == 1 and dim_head == dim)
        if self.project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.to_out = nn.Identity()


    def forward(self, x, mask=None):
#
    #def forward(self, x):
        # x: (b, n, d)
        b, n, d = x.shape
        x = self.norm(x)

        # (b, n, heads*dim_head)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to (b, heads, n, dim_head)
        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)  # (b, heads, n, dim_head)
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)

        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale

        if mask is not None:
            scores = scores + mask  # broadcast

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        # (b, heads, n, dim_head) -> (b, n, heads*dim_head)
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.wo(out)
        if self.project_out:
            out = self.to_out(out)
        return out


class FeedForwardTorch(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, mlp_dim, bias=False)

    def forward(self, x):
        x_norm = self.norm(x)
        x1 = self.w1(x_norm)
        x_silu = F.silu(x1)
        x2 = x_silu * self.w3(x_norm)
        x2 = self.drop(x2)
        return self.w2(x2)

    ## BlockTorch is a single transformer block that includes attention and feed-forward layers
class BlockTorch(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, seq_len, dropout):
        super().__init__()
        self.attn = AttentionTorch(dim, heads, dim_head, dropout)
        self.ff = FeedForwardTorch(dim, mlp_dim, dropout)
        # Build a causal mask if needed
#
        #self.register_buffer("_mask", self._causal_mask(seq_len), persistent=False)

##
    ## Create a causal mask for the attention mechanism
    # def _causal_mask(self, n):
    #     # shape: (1, 1, n, n) for broadcasting in multi-head attention
    #     # or simply (1, n, n). We'll do (1, n, n):
    #     mask = torch.triu(torch.full((n, n), float('-inf')), diagonal=1)
    #     return mask

    def forward(self, x):
        # x: (b, n, d)
        b, n, d = x.shape
        # Expand mask to (b, 1, n, n) if needed:

#
        #mask = self._mask.unsqueeze(0)  # (1, n, n)
        # attn
        x = x + self.attn(x)
#
        #x = x + self.attn(x, mask=mask)
        x = x + self.ff(x)
        return x

    ## Put all the layers together
class TransformerTorch(nn.Module):
    def __init__(self, depth, dim, heads, n_tokens, seq_len, dropout ,pool='cls'):
        super().__init__()
        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.embedding = nn.Embedding(n_tokens, dim)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, dim) * 0.02)
        self.layers = nn.ModuleList([
            BlockTorch(dim, heads, dim // heads, dim * 4, seq_len, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_tokens, bias=False)

    def forward(self, x):
        # x shape: (b, n)

        x = self.embedding(x)  # (b, n, dim)
        x_emb = x + self.pos_embedding.unsqueeze(0)
        x = x_emb

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            # last token
            x = x[:, -1]
        logits = self.out(x)
        return logits