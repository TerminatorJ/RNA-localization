# %%
import torch
import torch.nn as nn

# %%
class IndexEmbeddingOutputHead(nn.Module):
    def __init__(self, n_tasks, dims, task_names_txt=None):
        super(IndexEmbeddingOutputHead, self).__init__()

        # protein/experiment embedding of shape (p, d)
        self.embedding = torch.nn.Embedding(n_tasks, dims)

        if task_names_txt is not None:
            self.task_names = self._parse_names_txt(task_names_txt)
    
    def forward(self, bottleneck, **kwargs):
        # bottleneck of shape (batch_size, N, dim)
        # bottleneck = torch.transpose(bottleneck, -1, -2) # This was moved to the network module
        
        # embedding of (batch, p, d) --> (batch, d, p)
        embedding = torch.transpose(self.embedding.weight, 0, 1)
        
        # print(self.embedding.weight.shape)
        
        # print(bottleneck.shape)
        # print(embedding.shape)
        logits = torch.matmul(bottleneck, embedding) # torch.transpose(self.embedding.weight, 0, 1)  
        # logits = torch.matmul(self.embedding.weight, bottleneck)
        # print("logits", logits.shape)
        # logits = torch.matmul(bottleneck, embedding) # torch.transpose(self.embedding.weight, 0, 1) 
        # 
         
        return logits
    
    def _parse_task_names_txt(self, filepath):
        names = []
        with open(filepath) as f:
            for line in f:
                names.append(line.strip())
        return names