#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

/// @tparam T1 Type of array elements, should be a numerical type
/// @tparam T2 Type of array elements, should be a numerical type
/// @param label Array of ground truth labels, 0 is negative, 1 is positive
/// @param score Array of predicted scores, can be any real finite number
/// @param n Number of elements in the array, I assume it's correct
/// @return AUROC/ROC-AUC score, range [0.0, 1.0]
template<class T1, class T2>
double roc_auc(const vector<T1> &label, const vector<T2> &score) {
	int n = label.size();
	for (int i = 0; i < n; i++)
		if (!std::isfinite(score[i]) || label[i] != 0 && label[i] != 1)
			return std::numeric_limits<double>::signaling_NaN();

	const auto order = new int[n];
	std::iota(order, order + n, 0);
	std::sort(order, order + n, [&](int a, int b) { return score[a] > score[b]; });
	const auto y = new double[n];
	const auto z = new double[n];
	for (int i = 0; i < n; i++) {
		y[i] = label[order[i]];
		z[i] = score[order[i]];
	}

	const auto tp = y; // Reuse
	std::partial_sum(y, y + n, tp);

	int top = 0; // # diff
	for (int i = 0; i < n - 1; i++)
		if (z[i] != z[i + 1])
			order[top++] = i;
	order[top++] = n - 1;
	n = top; // Size of y/z -> sizeof tps/fps

	const auto fp = z; // Reuse
	for (int i = 0; i < n; i++) {
		tp[i] = tp[order[i]]; // order is mono. inc.
		fp[i] = 1 + order[i] - tp[i]; // Type conversion prevents vectorization
	}
	delete[] order;

	const auto tpn = tp[n - 1], fpn = fp[n - 1];
	for (int i = 0; i < n; i++) { // Vectorization
		tp[i] /= tpn;
		fp[i] /= fpn;
	}

	auto area = tp[0] * fp[0] / 2; // The first triangle from origin;
	double partial = 0; // For Kahan summation
	for (int i = 1; i < n; i++) {
		const auto x = (fp[i] - fp[i - 1]) * (tp[i] + tp[i - 1]) / 2 - partial;
		const auto sum = area + x;
		partial = (sum - area) - x;
		area = sum;
	}

	delete[] tp;
	delete[] fp;

	return area;
}
