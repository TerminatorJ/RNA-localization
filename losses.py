# %%
import torch
import torch.nn as nn

# %%
class MultinomialNLLLossFromLogits(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(MultinomialNLLLossFromLogits, self).__init__()
        self.reduction = reduction
    
    def __name__(self):
        return 'MultinomialNLLLossFromLogits'
    
    def __call__(self, y, y_pred, dim=-1):
        neg_log_probs = self.log_likelihood_from_logits(y, y_pred, dim) * -1
        if self.reduction is not None:
            return self.reduction(neg_log_probs)
        return neg_log_probs

    def log_likelihood_from_logits(self, y, y_pred, dim):
        return torch.sum(torch.mul(torch.log_softmax(y_pred, dim=dim), y), dim=dim) + self.log_combinations(y, dim)

    def log_combinations(self, input, dim):
        total_permutations = torch.lgamma(torch.sum(input, dim=dim) + 1)
        counts_factorial = torch.lgamma(input + 1)
        redundant_permutations = torch.sum(counts_factorial, dim=dim)
        return total_permutations - redundant_permutations
