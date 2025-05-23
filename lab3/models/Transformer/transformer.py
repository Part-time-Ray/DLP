import torch.nn as nn
import torch
from .modules import Encoder, TokenPredictor


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    
class BidirectionalTransformer(nn.Module):
    def __init__(self, configs):
        super(BidirectionalTransformer, self).__init__()
        self.num_image_tokens = configs['num_image_tokens']
        
        self.tok_emb = nn.Embedding(configs['num_codebook_vectors'] + 1, configs['dim'])
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(configs['num_image_tokens'], configs['dim'])), 0., 0.02)
        
        self.blocks = nn.Sequential(*[Encoder(configs['dim'], configs['hidden_dim']) for _ in range(configs['n_layers'])])
        self.Token_Prediction = TokenPredictor(configs['dim'])
        self.LN = nn.LayerNorm(configs['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        
        self.bias = nn.Parameter(torch.zeros(self.num_image_tokens, configs['num_codebook_vectors'] + 1)) # +1 mask token
        self.apply(weights_init)

    def forward(self, x):
        # Token domain -> Latent domain
        # print("x shape: ", x.shape)
        token_embeddings = self.tok_emb(x)

        # print("token_embeddings shape: ", token_embeddings.shape)
        # print("pos_emb shape: ", self.pos_emb.shape)
        embed = self.drop(self.LN(token_embeddings + self.pos_emb))
        # print("embed shape: ", embed.shape)
        embed = self.blocks(embed)
        embed = self.Token_Prediction(embed)
        # print("embed shape: ", embed.shape)

        logits = torch.matmul(embed, self.tok_emb.weight.T) + self.bias
        # print("tok_emb weight shape: ", self.tok_emb.weight.shape)

        return logits