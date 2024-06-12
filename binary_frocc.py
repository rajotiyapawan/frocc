import argparse
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from dataset_openml import Dataset as db
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import torch
import torch.nn.functional as F

import data_gen
import frocc
import dfrocc
import sparse_dfrocc
import pardfrocc
import kernels as k

# import utils

parser = argparse.ArgumentParser()

parser.add_argument("--dimension", default=1000, type=int)
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--repetitions", default=1, type=int)
parser.add_argument("--outfile", default="results/out_50_7270_combined.csv")
parser.add_argument("--n_samples", default=1000, type=int)
parser.add_argument("--n_dims", default=1000, type=int)
parser.add_argument("--fineTune", default=False, type=bool)

args = parser.parse_args()

dataset = db.from_openml(50)


dataset_len = len(dataset.X)
seed = 7270

class0 = dataset.y.unique()[0]
class1 = dataset.y.unique()[1]

X_positive = dataset.X[dataset.y == class1]
# y_positive = np.ones(len(X_positive))

X_negative = dataset.X[dataset.y == class0]
# y_negative = np.zeros(len(X_negative))


X_train_positive, X_test_positive = train_test_split(X_positive, test_size=0.2, random_state=seed)
X_train_negative, X_test_negative = train_test_split(X_negative, test_size=0.2, random_state=seed)

X_test = np.concatenate([X_test_positive, X_test_negative])

encoder = OneHotEncoder(handle_unknown='ignore')
X_train_positive = encoder.fit_transform(X_train_positive)
X_test_positive = encoder.transform(X_test_positive)
X_train_negative = encoder.transform(X_train_negative)
X_test_negative = encoder.transform(X_test_negative)

X_test = encoder.transform(X_test)

y_train_positive = np.ones(X_train_positive.shape[0])
y_train_negative = -np.ones(X_train_negative.shape[0])
y_test_positive = np.ones(X_test_positive.shape[0])
y_test_negative = -np.ones(X_test_negative.shape[0])
y_test = np.concatenate([y_test_positive, y_test_negative])



kernels = dict(
    zip(
        ["rbf", "linear", "poly", "sigmoid"],
        [k.rbf(), k.linear(), k.poly(), k.sigmoid()],
    )
)
try:
    kernel = kernels.get(args.kernel)
except KeyError as e:
    kernel = "rbf"

df = pd.DataFrame()

for run in range(args.repetitions):
    # print(
    #     f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run {run + 1} of "
    #     + f"{args.repetitions}"
    # )
    clf_positive = frocc.FROCC(num_clf_dim=1610, epsilon=0.1, kernel=kernel)
    clf_negative = frocc.FROCC(num_clf_dim=10, epsilon=0.01, kernel=kernel)

    # x = x.toarray()
    # xtest = xtest.toarray()

    tic = time()
    clf_positive.fit(X_train_positive.toarray(), y_train_positive)
    clf_negative.fit(X_train_negative.toarray(), y_train_negative)

    train_time = (time() - tic) * 1000 / X_train_positive.shape[0]
    tic = time()

    scores_positive = clf_positive.decision_function(X_test)
    scores_negative = clf_negative.decision_function(X_test)
    
    test_time = (time() - tic) * 1000 / X_test.shape[0]

    # Define combined function score
    # def combined_score(scores_positive, scores_negative, w1, w2):
    #     return (w1 * scores_positive - w2 * scores_negative)/(w1+w2)

    # # Define loss function
    # def loss_function(weights, scores_positive, scores_negative, y_test):
    #     w1, w2 = weights
    #     combined_scores = combined_score(scores_positive, scores_negative, w1, w2)
    #     return -roc_auc_score(y_test, combined_scores)

    # # Optimize weights
    # initial_weights = np.array([0.9, 0.1])
    # result = minimize(loss_function, initial_weights, args=(scores_positive, scores_negative, y_test),
    #                   bounds=[(0, 1), (0, 1)], method='SLSQP')
    # w1_opt, w2_opt = result.x
    # print(f"Optimal weights: w1 = {w1_opt}, w2 = {w2_opt}")

    # # Combined scores using optimal weights
    # combined_scores_opt = combined_score(scores_positive, scores_negative, w1_opt, w2_opt)

    def combined_function(weights, s1, s2):
        w1, w2 = weights
        return (w1 * s1 - w2 * s2) / (w1 + w2)

    # Normalize the scores to be in the range [0, 1]
    def normalize_scores(scores):
        return (scores - scores.min()) / (scores.max() - scores.min())

    s1_tensor = torch.tensor(normalize_scores(scores_positive), dtype=torch.float32)
    s2_tensor = torch.tensor(normalize_scores(scores_negative), dtype=torch.float32)
    y_true_binary = np.where(y_test == -1, 0, 1)
    y_true_tensor = torch.tensor(y_true_binary, dtype=torch.float32)

    # Initialize weights
    weights = torch.tensor([1.0, 1.0], requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([weights], lr=0.01)

    # Regularization term
    lambda_reg = 0.001

    for epoch in range(10000):
        optimizer.zero_grad()
        combined_scores = combined_function(weights, s1_tensor, s2_tensor)
    
    # Ensure combined_scores are logits by converting them if necessary
        combined_scores = torch.sigmoid(combined_scores)
    
       # Reshape y_true_tensor to match the shape of combined_scores
        y_true_tensor = y_true_tensor.view_as(combined_scores)
    
        # Calculate binary cross-entropy loss
        loss = F.binary_cross_entropy(combined_scores, y_true_tensor)
        loss += lambda_reg * torch.sum(weights**2)  # L2 regularization
        # Calculate mean squared error loss (optional)
        # loss = F.mse_loss(combined_scores, y_true_tensor)
        loss.backward()
        optimizer.step()
        
        # Optionally, print loss for monitoring
        if epoch % 1000 == 0:
            print(f"Iteration {epoch}: Loss = {loss.item()}")


    # Optimal weights
    optimal_weights = weights.detach().numpy()
    w1_opt, w2_opt = optimal_weights
    print(f"Optimal weights: w1 = {w1_opt}, w2 = {w2_opt}")

    # Calculate combined scores with optimal weights
    combined_scores = combined_function(optimal_weights, scores_positive, scores_negative)
    combined_scores = torch.sigmoid(torch.tensor(combined_scores, dtype=torch.float32)).numpy()
    roc_auc_combined = roc_auc_score(y_true_binary, combined_scores)

    # Calculate ROC AUC scores
    roc_positive = roc_auc_score(y_test, scores_positive)
    roc_negative = roc_auc_score(y_test, scores_negative)
    roc_auc_combined = roc_auc_score(y_test, combined_scores)

    print(f"Model 1 ROC AUC: {roc_positive:.2f}")
    print(f"Model 2 ROC AUC: {roc_negative:.2f}")
    print(f"Combined Model ROC AUC: {roc_auc_combined:.2f}")

    # Plot ROC curves
    fpr1, tpr1, _ = roc_curve(y_test, scores_positive)
    fpr2, tpr2, _ = roc_curve(y_test, scores_negative)
    fpr_combined, tpr_combined, _ = roc_curve(y_true_binary, combined_scores)

    plt.figure()
    plt.plot(fpr1, tpr1, label=f"Model 1 (AUC = {roc_positive:.2f})")
    plt.plot(fpr2, tpr2, label=f"Model 2 (AUC = {roc_negative:.2f})")
    plt.plot(fpr_combined, tpr_combined, label=f"Combined Model (AUC = {roc_auc_combined:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

    df = df._append(
        {
            "Run ID": run,
            "Kernel": args.kernel,
            "Dimension": args.dimension,
            "Epsilon": args.epsilon,
            "AUC of ROC Positive": roc_positive,
            "AUC of ROC Negative": roc_negative,
            "AUC of ROC Combined": roc_auc_combined,
            "Train Time": train_time,
            "Test Time": test_time,
        },
        ignore_index=True,
    )
df.to_csv(args.outfile)
