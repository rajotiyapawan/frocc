import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np

# Example ROC AUC scores for the two models
roc_auc_pos = 0.8544776119402990
roc_auc_neg = 0.5614783226723530

# Convert to PyTorch tensors
roc_auc_pos_tensor = torch.tensor(roc_auc_pos, dtype=torch.float32)
roc_auc_neg_tensor = torch.tensor(roc_auc_neg, dtype=torch.float32)

# Initialize weights
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)

# Define the combined function
def combined_score(w1, w2, auc_pos, auc_neg):
    numerator = w1 * auc_pos - w2 * auc_neg
    denominator = w1 + w2
    return numerator / denominator

# Define the loss function (negative combined score, as we want to maximize it)
def loss_fn(w1, w2, auc_pos, auc_neg):
    return -combined_score(w1, w2, auc_pos, auc_neg)

# Set up the optimizer
optimizer = optim.Adam([w1, w2], lr=0.01)

# Optimization loop
num_iterations = 1000
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = loss_fn(w1, w2, roc_auc_pos_tensor, roc_auc_neg_tensor)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 100 == 0:
        print(f'Iteration {i + 1}, Loss: {loss.item()}, w1: {w1.item()}, w2: {w2.item()}')

# Print the final weights
print(f'Final weights - w1: {w1.item()}, w2: {w2.item()}')

# Calculate the final combined score
final_score = combined_score(w1, w2, roc_auc_pos_tensor, roc_auc_neg_tensor)
print(f'Final Combined Score: {final_score.item()}')
