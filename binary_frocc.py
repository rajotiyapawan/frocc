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

import data_gen
import frocc
import dfrocc
import sparse_dfrocc
import pardfrocc
import kernels as k

# import utils

parser = argparse.ArgumentParser()

parser.add_argument("--dimension", default=21, type=int)
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--repetitions", default=5, type=int)
parser.add_argument("--outfile", default="results/out.csv")
parser.add_argument("--n_samples", default=1000, type=int)
parser.add_argument("--n_dims", default=1000, type=int)

args = parser.parse_args()

dataset = db.from_openml(31)


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
y_train_negative = np.zeros(X_train_negative.shape[0])
y_test_positive = np.ones(X_test_positive.shape[0])
y_test_negative = np.zeros(X_test_negative.shape[0])
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
    kernel = "linear"

df = pd.DataFrame()

for run in range(args.repetitions):
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run {run + 1} of "
        + f"{args.repetitions}"
    )
    clf_positive = frocc.FROCC(num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel)
    clf_negative = frocc.FROCC(num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel)

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
    
    roc_positive = roc_auc_score(y_test, scores_positive)
    roc_negative = roc_auc_score(y_test, scores_negative)
    

    df = df._append(
        {
            "Run ID": run,
            "Dimension": args.dimension,
            "Epsilon": args.epsilon,
            "AUC of ROC Positive": roc_positive,
            "AUC of ROC Negative": roc_negative,
            "Train Time": train_time,
            "Test Time": test_time,
        },
        ignore_index=True,
    )
df.to_csv(args.outfile)
