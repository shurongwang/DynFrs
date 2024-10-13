
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
from sklearn.datasets import make_classification


def main(random_state=1,
		 test_size=0.2,
		 n_samples=1000000,
		 n_features=40,
		 n_informative=5,
		 n_redundant=5,
		 n_repeated=0,
		 n_clusters_per_class=2,
		 flip_y=0.05,
		 out_dir='continuous'):

	# retrieve dataset
	data = make_classification(n_samples=n_samples,
							   n_features=n_features,
							   n_informative=n_informative,
							   n_redundant=n_redundant,
							   n_repeated=n_repeated,
							   n_clusters_per_class=n_clusters_per_class,
							   flip_y=flip_y,
							   random_state=random_state)
	X, y = data
	indices = np.arange(len(X))

	np.random.seed(random_state)
	train_indices = np.random.choice(indices, size=int(len(X) * (1 - test_size)), replace=False)
	test_indices = np.setdiff1d(indices, train_indices)

	X_train, y_train = X[train_indices], y[train_indices]
	X_test, y_test = X[test_indices], y[test_indices]

	# cleanup
	train = np.hstack([X_train, y_train.reshape(-1, 1)]).astype(np.float32)
	test = np.hstack([X_test, y_test.reshape(-1, 1)]).astype(np.float32)

	print(train.shape, train[:, -1].sum())
	print(test.shape, test[:, -1].sum())

	if train[:, -1].sum() == 0 or train[:, -1].sum() == len(train):
		print('train only contains 1 class!')
	if test[:, -1].sum() == 0 or test[:, -1].sum() == len(test):
		print('test only contains 1 class!')
	
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

