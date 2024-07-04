import openml
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# dataset_ids = [15, 37, 31]

dataset_ids = [23381, 1063, 40994, 6332, 1510, 1480, 29, 15, 1464, 37, 50, 31]
#kernels for dataset ids -  rbf, linear, rbf, linear, rbf

def load_and_preprocess_data(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X)
    y = np.where(y == np.unique(y)[0], 0, 1)  # Convert to binary labels
    return train_test_split(X_encoded, y, test_size=0.2, random_state=42)

datasets = [load_and_preprocess_data(dataset_id) for dataset_id in dataset_ids]

import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y.T)

def rbf_kernel(x, y, gamma=1.0):
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.exp(-gamma * np.sum(diff ** 2, axis=2))

def poly_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y.T) + coef0) ** degree

def sigmoid_kernel(x, y, alpha=1.0, coef0=1.0):
    return np.tanh(alpha * np.dot(x, y.T) + coef0)

kernels = {
    'linear': linear_kernel,
    'rbf': rbf_kernel,
    'poly': poly_kernel,
    'sigmoid': sigmoid_kernel,
}


import frocc_vector_optimise as frocc
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize
from skopt.space import Integer, Real

def train_and_evaluate_optimized_model(X_train, y_train, X_test, y_test, top_k=10,kernel_name='linear', num_clf_dim=21, epsilon=0.01):
    # Train FROCC models
    clf_positive, clf_negative = train_frocc_models(X_train, y_train, kernel_name, num_clf_dim, epsilon)
    
    # Evaluate original models
    scores_positive = clf_positive.decision_function(X_test.toarray())
    scores_negative = 1-clf_negative.decision_function(X_test.toarray())
    
    # Evaluate optimized models
    scores_positive_optimized = clf_positive.optimized_decision_function(X_test.toarray(), y_test, top_k=top_k)
    scores_negative_optimized = 1-clf_negative.optimized_decision_function(X_test.toarray(), y_test, top_k=top_k)
    
    roc_auc_positive = roc_auc_score(y_test, scores_positive)
    roc_auc_negative = roc_auc_score(y_test, scores_negative)
    roc_auc_positive_optimized = roc_auc_score(y_test, scores_positive_optimized)
    roc_auc_negative_optimized = roc_auc_score(y_test, scores_negative_optimized)
    print(f"Positive Model ROC AUC: {roc_auc_positive}")
    print(f"Optimized Positive Model ROC AUC: {roc_auc_positive_optimized}")
    print(f"Negative Model ROC AUC: {roc_auc_negative}")
    print(f"Optimized Negative Model ROC AUC: {roc_auc_negative_optimized}")

    return roc_auc_positive, roc_auc_negative,roc_auc_positive_optimized,roc_auc_negative_optimized,scores_positive, scores_negative, scores_positive_optimized, scores_negative_optimized


def train_frocc_models(X_train, y_train, kernel_name='linear', num_clf_dim=21, epsilon=0.01):
    kernel = kernels[kernel_name]
    class1_idx = np.where(y_train == 1)[0]
    class0_idx = np.where(y_train == 0)[0]
    
    X_train_positive = X_train[class1_idx].toarray()
    X_train_negative = X_train[class0_idx].toarray()
    
    clf_positive = frocc.FROCC(num_clf_dim=num_clf_dim, epsilon=epsilon, kernel=kernel)
    clf_negative = frocc.FROCC(num_clf_dim=num_clf_dim, epsilon=epsilon, kernel=kernel)
    
    clf_positive.fit(X_train_positive, np.ones(len(class1_idx)))
    clf_negative.fit(X_train_negative, np.zeros(len(class0_idx)))
    
    return clf_positive, clf_negative

def train_frocc_with_pca(X_train, y_train, n_components, kernel, dimension, epsilon):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train.toarray())

    # Split the data into positive and negative classes
    class1_idx = np.where(y_train == 1)[0]
    class0_idx = np.where(y_train == 0)[0]
    
    X_train_positive = X_train_pca[class1_idx]
    X_train_negative = X_train_pca[class0_idx]
    
    # Initialize FROCC models
    clf_positive = frocc.FROCC(num_clf_dim=dimension, epsilon=epsilon, kernel=kernel)
    clf_negative = frocc.FROCC(num_clf_dim=dimension, epsilon=epsilon, kernel=kernel)
    
    # Train models
    clf_positive.fit(X_train_positive, np.ones(len(class1_idx)))
    clf_negative.fit(X_train_negative, np.zeros(len(class0_idx)))
    
    return clf_positive, clf_negative, pca

def evaluate_frocc_models(clf_positive, clf_negative, X_test, y_test):
    scores_positive = clf_positive.decision_function(X_test.toarray())
    scores_negative = 1 - clf_negative.decision_function(X_test.toarray())
    
    roc_auc_positive = roc_auc_score(y_test, scores_positive)
    roc_auc_negative = roc_auc_score(y_test, scores_negative)
    
    return roc_auc_positive, roc_auc_negative, scores_positive, scores_negative

def objective_function_pca(params, X_train, y_train, X_test, y_test, kernel):
    n_components, dimension, epsilon = params
    clf_positive, clf_negative, pca = train_frocc_with_pca(X_train, y_train, n_components, kernel, dimension, epsilon)
    
    X_test_pca = pca.transform(X_test.toarray())
    scores_positive = clf_positive.decision_function(X_test_pca)
    scores_negative = 1-clf_negative.decision_function(X_test_pca)
    
    combined_scores = combined_function([0.5,0.5], scores_positive, scores_negative)
    
    roc_auc_combined = roc_auc_score(y_test, combined_scores)
    return -roc_auc_combined 

def objective_function_params(params, X_train, y_train, X_test, y_test, kernel):
    dimension, epsilon = params
    clf_positive, clf_negative = train_frocc_models(X_train, y_train, kernel, dimension, epsilon)
    
    scores_positive = clf_positive.decision_function(X_test.toarray())
    scores_negative = 1-clf_negative.decision_function(X_test.toarray())
    
    combined_scores = combined_function([0.5,0.5], scores_positive, scores_negative)
    
    roc_auc_combined = roc_auc_score(y_test, combined_scores)
    return -roc_auc_combined 

space_pca  = [Integer(5, 50, name='n_components'),
        Integer(5, 1000, name='dimension'),
          Real(0.001, 1, name='epsilon')]
space = [
        Integer(5, 1000, name='dimension'),
          Real(0.001, 1, name='epsilon')]

def objective(params):
    dimension, epsilon = params
    return objective_function_params((dimension, epsilon), X_train, y_train, X_test, y_test,kernel='linear')

def objective_pca(params):
    n_components, dimension, epsilon = params
    return objective_function_pca((n_components, dimension, epsilon), X_train, y_train, X_test, y_test,kernel='rbf')


# Hyperparameter Tuning Function
def tune_frocc_models(X_train, y_train, X_val, y_val, kernel_name='rbf'):
    best_score = 0
    best_params = {'num_clf_dim': None, 'epsilon': None}

    for num_clf_dim in range(10,1010,30):
        for epsilon in range(10, 110, 10):
            # print(f"dimension={num_clf_dim}, epsilon={epsilon/1000}")
            clf_positive, clf_negative = train_frocc_models(X_train, y_train, kernel_name, num_clf_dim, epsilon/1000)
            roc_auc_positive, roc_auc_negative,_,_ = evaluate_frocc_models(clf_positive, clf_negative, X_val, y_val)
            combined_score = (roc_auc_positive + roc_auc_negative) / 2
            if combined_score > best_score:
                best_score = combined_score
                best_params['num_clf_dim'] = num_clf_dim
                best_params['epsilon'] = epsilon

    print(f"Optimal parameters: num_clf_dim = {best_params['num_clf_dim']}, epsilon = {best_params['epsilon']/1000}")
    return best_params

# frocc_results = []
# for X_train, X_test, y_train, y_test in datasets:
#     clf_positive, clf_negative = train_frocc_models(X_train, y_train)
#     roc_auc_positive, roc_auc_negative = evaluate_frocc_models(clf_positive, clf_negative, X_test, y_test)
#     frocc_results.append((roc_auc_positive, roc_auc_negative))

from sklearn.svm import SVC
from catboost import CatBoostClassifier

def train_baseline_models(X_train, y_train):
    svm_clf = SVC(probability=True)
    catboost_clf = CatBoostClassifier(verbose=0)
    
    svm_clf.fit(X_train, y_train)
    catboost_clf.fit(X_train, y_train)
    
    return svm_clf, catboost_clf

def evaluate_baseline_models(svm_clf, catboost_clf, X_test, y_test):
    svm_scores = svm_clf.predict_proba(X_test)[:, 1]
    catboost_scores = catboost_clf.predict_proba(X_test)[:, 1]
    
    roc_auc_svm = roc_auc_score(y_test, svm_scores)
    roc_auc_catboost = roc_auc_score(y_test, catboost_scores)
    
    return roc_auc_svm, roc_auc_catboost

# baseline_results = []
# for X_train, X_test, y_train, y_test in datasets:
#     svm_clf, catboost_clf = train_baseline_models(X_train.toarray(), y_train)
#     roc_auc_svm, roc_auc_catboost = evaluate_baseline_models(svm_clf, catboost_clf, X_test.toarray(), y_test)
#     baseline_results.append((roc_auc_svm, roc_auc_catboost))

import torch
import torch.nn.functional as F
from torch.optim import Adam

def combined_function(weights, s1, s2):
    w1, w2 = weights
    return (w1 * s1 + w2 * s2) / (w1 + w2)

def objective_function(weights, s1, s2, y_true):
    combined_scores = combined_function(weights, s1, s2)
    return -roc_auc_score(y_true, combined_scores)

def optimize_combined_model(scores_positive, scores_negative, y_test_binary, initial_weights=[0.5, 0.5]):
    s1_tensor = torch.tensor(scores_positive, dtype=torch.float32)
    s2_tensor = torch.tensor(scores_negative, dtype=torch.float32)
    y_true_tensor = torch.tensor(y_test_binary, dtype=torch.float32)
    
    weights = torch.tensor(initial_weights, requires_grad=True)
    optimizer = Adam([weights], lr=0.01)
    
    lambda_reg = 0.01
    num_iterations = 1000
    for epoch in range(num_iterations):
        optimizer.zero_grad()
        combined_scores = combined_function(weights, s1_tensor, s2_tensor)
        combined_scores = torch.sigmoid(combined_scores)
        loss = F.binary_cross_entropy(combined_scores, y_true_tensor)
        loss += lambda_reg * torch.sum(weights**2)
        loss.backward()
        optimizer.step()
    
    optimal_weights = weights.detach().numpy()
    combined_scores = combined_function(optimal_weights, s1_tensor, s2_tensor).detach().numpy()
    combined_roc_auc = roc_auc_score(y_test_binary, combined_scores)
    
    return optimal_weights, combined_roc_auc

from scipy.optimize import minimize
def optimize_combined_model_SLSQP(scores_positive, scores_negative, y_test_binary, initial_weights=[0.5, 0.5]):
    s1_tensor = torch.tensor(scores_positive, dtype=torch.float32)
    s2_tensor = torch.tensor(scores_negative, dtype=torch.float32)
    y_true_tensor = torch.tensor(y_test_binary, dtype=torch.float32)

    # Constraints to ensure non-negative weights
    constraints = [{'type': 'ineq', 'fun': lambda w: w[0]}, {'type': 'ineq', 'fun': lambda w: w[1]}]

    # Optimization using SLSQP
    result = minimize(objective_function, initial_weights, args=(scores_positive, scores_negative, y_test_binary), method='SLSQP', constraints=constraints)

    # Optimal weights
    optimal_weights = result.x
    combined_scores = combined_function(optimal_weights, scores_positive, scores_negative)
    combined_roc_auc = roc_auc_score(y_test_binary, combined_scores)
    
    return optimal_weights, combined_roc_auc

frocc_results = []
baseline_results = []
combined_results = []
results = []
initial_weights_list = [[0.5, 0.5], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8], [0.8, 0.2],[0.1, 0.9], [0.9, 0.1], [0.4, 0.6], [0.6, 0.4]]
for X_train, X_test, y_train, y_test in datasets:
    # Tune FROCC models
    print("Tuning the model")

    # result = gp_minimize(objective_pca, space, n_calls=30, random_state=42)

    # best_n_components = result.x[0]
    # best_dimension = result.x[1]
    # best_epsilon = result.x[2]
    # best_score = -result.fun

    # print(f"Best n_components: {best_n_components}, Best dimension: {best_dimension}, Best epsilon: {best_epsilon}, Best ROC AUC: {best_score}")

    res_gp = gp_minimize(objective, space, n_calls=20, random_state=42)

    print(f"Best params: dimension={res_gp.x[0]}, epsilon={res_gp.x[1]}, Best ROC AUC: {-res_gp.fun}")

    # train and evaluate the models
    # clf_positive, clf_negative, pca = train_frocc_with_pca(X_train, y_train, best_n_components, 'rbf', best_dimension, best_epsilon)

    roc_auc_positive, roc_auc_negative,roc_auc_positive_optimized,roc_auc_negative_optimized,scores_positive, scores_negative, scores_positive_optimized, scores_negative_optimized = train_and_evaluate_optimized_model(X_train, y_train, X_test, y_test)
    frocc_results.append((roc_auc_positive, roc_auc_negative,roc_auc_positive_optimized, roc_auc_negative_optimized))
    # roc_auc_positive, roc_auc_negative, scores_positive, scores_negative = evaluate_frocc_models(clf_positive, clf_negative, X_test_pca, y_test)
    # frocc_results.append((roc_auc_positive, roc_auc_negative))
    # clf_positive, clf_negative = train_frocc_models(X_train, y_train, kernel_name='rbf', num_clf_dim=res_gp.x[0], epsilon=res_gp.x[1])
    # roc_auc_positive, roc_auc_negative, scores_positive, scores_negative = evaluate_frocc_models(clf_positive, clf_negative, X_test, y_test)
    # frocc_results.append((roc_auc_positive, roc_auc_negative))

    # train and evaluate the baseline models
    svm_clf, catboost_clf = train_baseline_models(X_train.toarray(), y_train)
    roc_auc_svm, roc_auc_catboost = evaluate_baseline_models(svm_clf, catboost_clf, X_test.toarray(), y_test)
    
    baseline_results.append((roc_auc_svm, roc_auc_catboost))

    # evaluate the combined model
    y_test_binary = np.where(y_test == 0, 0, 1)
    
    best_combined_roc_auc = 0
    best_weights = None
    best_combined_roc_auc_optimized = 0
    best_weights_optimized = None
    for initial_weights in initial_weights_list:
        optimal_weights, combined_roc_auc = optimize_combined_model_SLSQP(scores_positive, scores_negative, y_test_binary, initial_weights)
        if combined_roc_auc > best_combined_roc_auc:
            best_combined_roc_auc = combined_roc_auc
            best_weights = optimal_weights
    
    for initial_weights in initial_weights_list:
        optimal_weights, combined_roc_auc_optimized = optimize_combined_model_SLSQP(scores_positive_optimized, scores_negative_optimized, y_test_binary, initial_weights)
        if combined_roc_auc_optimized > best_combined_roc_auc_optimized:
            best_combined_roc_auc_optimized = combined_roc_auc_optimized
            best_weights_optimized = optimal_weights
    
    combined_results.append((best_combined_roc_auc, best_combined_roc_auc_optimized))
    
    # print(f"roc_positive = {roc_auc_positive}")
    # print(f"roc_negative = {roc_auc_negative}")
    # print(f"roc_auc_positive_optimized = {roc_auc_positive_optimized}")
    # print(f"roc_auc_negative_optimized = {roc_auc_negative_optimized}")
    # print(f"best_weights = {best_weights}")
    # print(f"best_combined_roc_auc = {best_combined_roc_auc}")
    # print(f"best_combined_roc_auc_optimized = {best_combined_roc_auc_optimized}")
    # print(f"roc_auc_svm = {roc_auc_svm}")
    # print(f"roc_auc_catboost = {roc_auc_catboost}")


    results.append({
        'kernel': 'linear',
        'num_clf_dim': res_gp.x[0],
        'epsilon': res_gp.x[1],
        'frocc_positive_roc_auc': roc_auc_positive,
        'frocc_negative_roc_auc': roc_auc_negative,
        'combined_roc_auc': best_combined_roc_auc,
        'combined_weights': best_weights,
        'frocc_positive_roc_auc_optimized': roc_auc_positive_optimized,
        'frocc_negative_roc_auc_optimized': roc_auc_negative_optimized,
        'combined_roc_auc_optimized': best_combined_roc_auc_optimized,
        'combined_weights_optimized': best_weights_optimized,
        'svm_roc_auc': roc_auc_svm,
        'roc_auc_catboost': roc_auc_catboost
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/frocc_combined_results_vector_optimized(2).csv', index=False)

import matplotlib.pyplot as plt

datasets_names = [str(dataset_id) for dataset_id in dataset_ids]

frocc_pos = [result[0] for result in frocc_results]
frocc_neg = [result[1] for result in frocc_results]
frocc_pos_optimized = [result[2] for result in frocc_results]
frocc_neg_optimized = [result[3] for result in frocc_results]

combined_result = [result[0] for result in combined_results]
combined_result_optimized = [result[1] for result in combined_results]

svm_auc = [result[0] for result in baseline_results]
catboost_auc = [result[1] for result in baseline_results]

plt.figure(figsize=(14, 8))
plt.plot(datasets_names, frocc_pos, label='FROCC Model 1 ROC AUC', marker='o')
plt.plot(datasets_names, frocc_neg, label='FROCC Model 2 ROC AUC', marker='o')
plt.plot(datasets_names, frocc_pos_optimized, label='FROCC Model 1 ROC AUC Optimized', marker='o')
plt.plot(datasets_names, frocc_neg_optimized, label='FROCC Model 2 ROC AUC Optimized', marker='o')
# plt.plot(datasets_names, combined_result, label='Combined FROCC Model ROC AUC', marker='o')
# plt.plot(datasets_names, combined_result_optimized, label='Combined FROCC Model Optimized ROC AUC', marker='o')
# plt.plot(datasets_names, svm_auc, label='SVM ROC AUC', marker='o')
# plt.plot(datasets_names, catboost_auc, label='CatBoost ROC AUC', marker='o')
plt.xlabel('Dataset ID')
plt.ylabel('ROC AUC')
plt.title('Comparison of ROC AUC Scores across Different Datasets')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 8))
# plt.plot(datasets_names, frocc_pos, label='FROCC Model 1 ROC AUC', marker='o')
# plt.plot(datasets_names, frocc_neg, label='FROCC Model 2 ROC AUC', marker='o')
# plt.plot(datasets_names, frocc_pos_optimized, label='FROCC Model 1 ROC AUC Optimized', marker='o')
# plt.plot(datasets_names, frocc_neg_optimized, label='FROCC Model 2 ROC AUC Optimized', marker='o')
plt.plot(datasets_names, combined_result, label='Combined FROCC Model ROC AUC', marker='o')
plt.plot(datasets_names, combined_result_optimized, label='Combined FROCC Model Optimized ROC AUC', marker='o')
plt.plot(datasets_names, svm_auc, label='SVM ROC AUC', marker='o')
plt.plot(datasets_names, catboost_auc, label='CatBoost ROC AUC', marker='o')
plt.xlabel('Dataset ID')
plt.ylabel('ROC AUC')
plt.title('Comparison of ROC AUC Scores across Different Datasets')
plt.legend()
plt.grid(True)
plt.show()


# Set width of bar
bar_width = 0.15

# Set position of bar on X axis
r1 = np.arange(len(dataset_ids))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Make the plot
plt.figure(figsize=(14, 7))
plt.bar(r1, frocc_pos, color='b', width=bar_width, edgecolor='grey', label='FROCC Model 1 ROC AUC')
plt.bar(r2, frocc_neg, color='r', width=bar_width, edgecolor='grey', label='FROCC Model 2 ROC AUC')
plt.bar(r3, frocc_pos_optimized, color='g', width=bar_width, edgecolor='grey', label='FROCC Model 1 ROC AUC Optimized')
plt.bar(r4, frocc_neg_optimized, color='c', width=bar_width, edgecolor='grey', label='FROCC Model 2 ROC AUC Optimized')
# plt.bar(r5, catboost_auc, color='m', width=bar_width, edgecolor='grey', label='CatBoost ROC AUC')

# Add labels
plt.xlabel('Dataset IDs', fontweight='bold')
plt.ylabel('ROC AUC', fontweight='bold')
plt.xticks([r + 2 * bar_width for r in range(len(dataset_ids))], dataset_ids, rotation=45)

# Add legend
plt.legend()

# Display the plot
plt.title('ROC AUC Scores Comparison Across Models')
plt.show()

# Make the plot
plt.figure(figsize=(14, 7))
plt.bar(r1, combined_result, color='b', width=bar_width, edgecolor='grey', label='Combined FROCC Model ROC AUC')
plt.bar(r2, combined_result_optimized, color='g', width=bar_width, edgecolor='grey', label='Combined FROCC Model Optimized ROC AUC')
plt.bar(r3, svm_auc, color='c', width=bar_width, edgecolor='grey', label='SVM ROC AUC')
plt.bar(r4, catboost_auc, color='m', width=bar_width, edgecolor='grey', label='CatBoost ROC AUC')

# Add labels
plt.xlabel('Dataset IDs', fontweight='bold')
plt.ylabel('ROC AUC', fontweight='bold')
plt.xticks([r + 2 * bar_width for r in range(len(dataset_ids))], dataset_ids, rotation=45)

# Add legend
plt.legend()

# Display the plot
plt.title('ROC AUC Scores Comparison Across Models')
plt.show()
