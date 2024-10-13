
"""
Generates a continuous attribute binary classification dataset.

This code is originally written by Brophy and Lowd (2021):
[1] J. Brophy and D. Lowd, "Machine Unlearning for Random Forests,"
    in Proceedings of the 38th International Conference on Machine Learning, PMLR,
    Jul. 2021, pp. 1092-1104.
    Available: https://proceedings.mlr.press/v139/brophy21a.html
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def dataset_specific(random_state, test_size):

	# retrieve dataset
	df = pd.read_csv(os.path.join('HIGGS.csv'), header=None)
	df.columns = ['label'] + ['f{}'.format(i) for i in range(len(df.columns) - 1)]
	df['label'] = df['label'].astype(int)

	# remove select columns
	remove_cols = []
	if len(remove_cols) > 0:
		df = df.drop(columns=remove_cols)

	# remove nan rows
	nan_rows = df[df.isnull().any(axis=1)]
	print('nan rows: {}'.format(len(nan_rows)))
	df = df.dropna()

	# split into train and test
	indices = np.arange(len(df))
	n_train_samples = int(len(indices) * (1 - test_size))

	np.random.seed(random_state)
	train_indices = np.random.choice(indices, size=n_train_samples, replace=False)
	test_indices = np.setdiff1d(indices, train_indices)

	train_df = df.iloc[train_indices]
	test_df = df.iloc[test_indices]

	# categorize attributes
	columns = list(df.columns)
	label = ['label']
	numeric = ['f{}'.format(i) for i in range(len(columns) - 1)]
	categorical = list(set(columns) - set(numeric) - set(label))
	print('label', label)
	print('numeric', numeric)
	print('categorical', categorical)

	return train_df, test_df, label, numeric, categorical


def main(random_state=1, test_size=0.2, out_dir='continuous'):

	train_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state,
																	  test_size=test_size)

	# encode categorical inputs
	ct = ColumnTransformer([('kbd', 'passthrough', numeric),
							('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical)])
	X_train = ct.fit_transform(train_df)
	X_test = ct.transform(test_df)

	# binarize outputs
	le = LabelEncoder()
	y_train = le.fit_transform(train_df[label].to_numpy().ravel()).reshape(-1, 1)
	y_test = le.transform(test_df[label].to_numpy().ravel()).reshape(-1, 1)

	# add labels
	train = np.hstack([X_train, y_train]).astype(np.float32)
	test = np.hstack([X_test, y_test]).astype(np.float32)

	print('\ntrain:\n{}, dtype: {}'.format(train, train.dtype))
	print('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))

	print('\ntest:\n{}, dtype: {}'.format(test, test.dtype))
	print('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

	m = X_train.shape[1]
	n_train = X_train.shape[0]
	n_test = X_test.shape[0]

	s = f"{n_train} {m+1}\n"
	wtr = open("train.txt", "w")
	for i in range(n_train):
		for j in range(m):
			s += f"{X_train[i][j]:.5f} "
		s += str(y_train[i]) + '\n'
	wtr.write(s)
	wtr.close()

	s = f"{n_test} {m+1}\n"
	wtr = open("test.txt", "w")
	for i in range(n_test):
		for j in range(m):
			s += f"{X_test[i][j]:.5f} "
		s += str(y_test[i]) + '\n'
	wtr.write(s)
	wtr.close()

if __name__ == '__main__':
	main()
