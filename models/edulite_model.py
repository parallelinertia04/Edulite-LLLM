import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.embedding(x)

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()

        self.heads = heads
        self.head_dim = emb_dim // heads

        self.query = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)

        return self.fc(out)

class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(self, emb_dim, heads):
        super().__init__()

        self.attention = SelfAttention(emb_dim, heads)
        self.ff = FeedForward(emb_dim)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):

        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))

        return x

class EduLiteLLM(nn.Module):

    def __init__(self, vocab_size, emb_dim=256, layers=6, heads=4):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.blocks = nn.Sequential(
            *[TransformerBlock(emb_dim, heads) for _ in range(layers)]
        )

        self.norm = nn.LayerNorm(emb_dim)

        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        x = self.blocks(x)

        x = self.norm(x)

        logits = self.fc(x)

        return logits
                    