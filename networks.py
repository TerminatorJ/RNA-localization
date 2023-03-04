# %%
import sys

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# %%
from layers import base
from layers import embedding
# Conv1DFirstLayer, Conv1DResBlock, IndexEmbeddingOutputHead, LinearProjection

# %%
@gin.configurable()
class MultiRBPNet(nn.Module):
    def __init__(self, n_tasks, n_layers=9, n_body_filters=256):
        super(MultiRBPNet, self).__init__()

        self.n_tasks = n_tasks
        self.n_body_filters = n_body_filters

        self.body = nn.Sequential(*[base.Conv1DFirstLayer(4, n_body_filters, 6)]+[(base.Conv1DResBlock(n_body_filters, n_body_filters, dilation=(2**i))) for i in range(n_layers)])
        self.linear_projection = base.LinearProjection(in_features=n_body_filters)
        self.head = embedding.IndexEmbeddingOutputHead(self.n_tasks, dims=n_body_filters)
    def _make_res_tower(self, n_layers, filters, dilation_factor):
        tower = nn.Sequential([
            (base.ResConv1DBlock(filters, filters, dilation=(dilation_factor**i))) for i in range(n_layers)
        ])
        return tower

    
    def forward(self, inputs, **kwargs):
        x = inputs
        for layer in self.body:
            x = layer(x)
        # transpose: # (batch_size, dim, N) --> (batch_size, N, dim)
        x = torch.transpose(x, dim0=-2, dim1=-1)
        x = self.linear_projection(x)

        return self.head(x)


@gin.configurable()
class ProteinEmbeddingMultiRBPNet(nn.Module):
    def __init__(self, n_layers=9, n_body_filters=256):
        super(ProteinEmbeddingMultiRBPNet, self).__init__()

        # layers RNA
        self.body = nn.Sequential(*[base.Conv1DFirstLayer(4, n_body_filters, 6)]+[(base.Conv1DResBlock(n_body_filters, n_body_filters, dilation=(2**i))) for i in range(n_layers)])
        self.rna_projection = nn.Linear(in_features=n_body_filters, out_features=256, bias=False)

        # layers protein
        self.protein_projection = nn.Linear(in_features=1280, out_features=256, bias=False)

    def forward(self, inputs, **kwargs):
        # forward RNA
        x_r = inputs.view(-1,4,8000)
        for layer in self.body:
            x_r = layer(x_r)
        # transpose: # (batch_size, dim, N) --> (batch_size, N, dim)
        x_r = torch.transpose(x_r, dim0=-2, dim1=-1)
        # project: (batch_size, N, dim) --> (batch_size, N, new_dim)
        x_r = self.rna_projection(x_r)
        
        # forward protein
        x_p = inputs['embedding']
        x_p = self.protein_projection(x_p)
        # x_r: (#proteins, dim)

        # transpose representations for matmul
        x_p = torch.transpose(x_p, dim0=1, dim1=0) # (dim, #proteins)
        
        try:
            x = torch.matmul(x_r, x_p) # (batch_size, N, #proteins)
        except:
            print('x_r.shape', x_r.shape, file=sys.stderr)
            print('x_p.shape', x_p.shape, file=sys.stderr)
            raise

        return  x
    

# if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = MultiRBPNet(n_tasks = 4, n_layers=9, n_body_filters=256)
    model = model.to(device = device)
    input = torch.ones(1, 4, 8000, device = "cuda")
    model(input)
    for i in list(model.state_dict().keys())[:]:
        print(model.state_dict()[i].shape)
    # print(model)
    summary(model, input_size = (4, 8000), batch_size = 0, device = device)

    #loading the model
    # ck_point = torch.load("/binf-isilon/winthergrp/jwang/panRBPnet/checkpoint/epoch=21-step=61160.ckpt")
    # print(ck_point)