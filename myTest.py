import argparse
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
parser.add_argument("--outfile", default="results/out.csv")
parser.add_argument("--method", default="pardfrocc")
parser.add_argument("--n_samples", default=1000, type=int)
parser.add_argument("--n_dims", default=1000, type=int)

args = parser.parse_args()

data = pd.read_csv('diabetes/Diabetes-Data/data-01', sep='\t', header=None, names=['Date', 'Time', 'code', 'value'])
# Load your dataset into a DataFrame (replace this with your actual dataset loading code)
# Assuming 'Feature1' is the column with seven unique entries

# Assuming 'Date' and 'Time' are in the format 'MM-DD-YYYY' and 'HH:MM'
# Split 'Date' into 'Month', 'Day', and 'Year'
data[['Month', 'Day', 'Year']] = data['Date'].str.split('-', expand=True).astype(int)

# Split 'Time' into 'Hour' and 'Minute'
data[['Hour', 'Minute']] = data['Time'].str.split(':', expand=True).astype(int)

# Drop the original 'Date' and 'Time' columns
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Assuming the third column ('Feature1') is considered as labels
# X = data[['Month', 'Day', 'Year', 'Hour', 'Minute', 'value']]  # Features
# y = data['code']  # Labels

# # Separate Date and Time into individual components
# data['Year'] = pd.to_datetime(data['Date']).dt.year
# data['Month'] = pd.to_datetime(data['Date']).dt.month
# data['Day'] = pd.to_datetime(data['Date']).dt.day
# data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
# data['Minute'] = pd.to_datetime(data['Time'], format='%H:%M').dt.minute

# # Drop original 'Date' and 'Time' columns
# data.drop(['Date', 'Time'], axis=1, inplace=True)

unique_entries = data['code'].unique()

# Select positive and negative classes based on the dataset
positive_class = unique_entries[0]  # Choose one of the classes randomly
negative_classes = unique_entries[1:]  # All other classes are negative

# Separate positive and negative instances
positive_instances = data[data['code'] == positive_class]
negative_instances = data[data['code'].isin(negative_classes)]

# Split positive instances into train and test sets
X_pos_train, X_pos_test = train_test_split(positive_instances[['Month', 'Day', 'Year', 'Hour', 'Minute', 'value']], test_size=0.5, random_state=42)

# Create negative instances by duplicating positive instances
X_neg_train = X_neg_test = negative_instances[['Month', 'Day', 'Year', 'Hour', 'Minute', 'value']].sample(len(X_pos_test), replace=True, random_state=42)

# Combine positive and negative instances for train and test sets
x = X_pos_train
xtest = pd.concat([X_pos_test, X_neg_test])

# Create labels for OCC
y = [np.ones(len(X_pos_train))]
ytest = np.concatenate([np.ones(len(X_pos_test)), np.zeros(len(X_neg_test))])


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

print(
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Running "
    + f"diabetes dataset with {args.dimension} dimensions and "
    + f"epsilon={args.epsilon} with {args.kernel} kernel for "
    + f"{args.repetitions} repetitions."
)

for run in range(args.repetitions):
    print(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run {run + 1} of "
        + f"{args.repetitions}"
    )
    if args.method == "frocc":
        clf = frocc.FROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )
        # x = x.toarray()
        # xtest = xtest.toarray()
    elif args.method == "dfrocc":
        clf = dfrocc.DFROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )
    elif args.method == "sparse_dfrocc":
        clf = sparse_dfrocc.SDFROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )
    elif args.method == "pardfrocc":
        clf = pardfrocc.ParDFROCC(
            num_clf_dim=args.dimension, epsilon=args.epsilon, kernel=kernel
        )

    tic = time()
    clf.fit(x)
    train_time = (time() - tic) * 1000 / x.shape[0]
    tic = time()
    scores = clf.decision_function(xtest)
    test_time = (time() - tic) * 1000 / xtest.shape[0]
    roc = roc_auc_score(ytest, scores)
    df = df._append(
        {
            "Run ID": run,
            "Dimension": args.dimension,
            "Epsilon": args.epsilon,
            "AUC of ROC": roc,
            "Train Time": train_time,
            "Test Time": test_time,
        },
        ignore_index=True,
    )
df.to_csv(args.outfile)
