
// #include <bits/stdc++.h>
#include <vector>
#include "forest-5.h"
#include "roc_auc.h"

using namespace std;
using namespace std::chrono;
vector<vector<Tx>> X_train, X_test;
vector<Ty> Y_train, Y_test;

void load_file(const string &filename, vector<vector<Tx>> &X, vector<Ty> &Y) {
	FILE *buf_;
	int buf__;
	int n, m;
	
	buf_ = freopen(filename.c_str(), "r", stdin);
	cin >> n >> m;
	
	X.resize(n, vector<Tx>(m - 1));
	Y.resize(n);
	for (int r = 0; r < n; ++r) {
		for (int c = 0; c < m - 1; ++c) {
			cin >> X[r][c];
		}
		Tx tmp;
		cin >> tmp;
		Y[r] = tmp;
	}
}

bool auc = 0;
double eval(random_forest &RF, vector<vector<Tx>> X_test, vector<Ty> Y_test, bool dbg = 0) {
	vector<double> res, pred(X_test.size());
	int correct = 0;
	for (int i = 0; i < X_test.size(); ++i) {
		const auto &X = X_test[i];
		const auto &Y = Y_test[i];
		
		RF.qry(X, res);
		pred[i] = res[1];
		correct += (pred[i] >= 0.5) == Y_test[i];
	}
	double result = (double) correct / Y_test.size();
	if (auc) result = roc_auc(Y_test, pred);
	return result;
}

const bool dbg = 1;

void upd(long long &sum, long long &min, long long &max, int t) {
	sum += t;
	if (min == -1 || min > t) min = t;
	if (max == -1 || max < t) max = t;
}

int main(int argc, char *args[]) {
	srand(time(NULL));
	ios::sync_with_stdio(0);
	
	int T = 100, k = 10, max_dep = 20, n_unlearn = 0, mode = 1;
	long long time_us = 0;
	string data_dir = "./";

	unordered_map<string, string> cmd;
	for (int i = 1; i < argc; i += 2) {
		string s1 = "", s2 = "";
		char *c1 = args[i], *c2 = args[i + 1];
		for (; *c1 != '\0'; s1 += *c1, ++c1);
		for (; *c2 != '\0'; s2 += *c2, ++c2);
		cmd[s1] = s2;
	}
	for (auto s : cmd) {
		const string &tp = s.first, &r = s.second;
		if (tp == "-data") {
			data_dir = "Datasets/" + r + '/';
		} else if (tp == "-T") {
			T = atoi(r.c_str());
		} else if (tp == "-k") {
			k = atoi(r.c_str());
		} else if (tp == "-d" || tp == "-max_dep") {
			max_dep = atoi(r.c_str());
		} else if (tp == "-s") {
			p_tries = atoi(r.c_str());
		} else if (tp == "-auc") {
			auc = atoi(r.c_str());
		} else if (tp == "-t" || tp == "-time") {
			time_us = atoi(r.c_str()) * 1000;
		} else if (tp == "-n_unl" || tp == "-n_unlearn") {
			n_unlearn = atoi(r.c_str());
		} else if (tp == "-dly" || tp == "-delay") {
			dly = atoi(r.c_str());
		} else if (tp == "-mode") {
			mode = atoi(r.c_str());
		}
	}
	cout << data_dir << ' ' << T << ' ' << k << ' ' << max_dep << ' ' << p_tries << ' ' << dly << ' ' << mode << endl;

	load_file(data_dir + "train.txt", X_train, Y_train);
	load_file(data_dir + "test.txt", X_test, Y_test);
	
	p_count = int(sqrt(X_train[0].size())) + 1;

	if (dbg) {
		cerr << "train size: " << X_train.size() << ' ' << X_train[0].size() << endl;
		cerr << "test size:  " << X_test.size() << ' ' << X_test[0].size() << endl;
	}
	
	auto time_s = high_resolution_clock::now(), time_e = time_s;

	time_s = high_resolution_clock::now();
	
	random_forest RF(X_train, Y_train, T, k, max_dep);
	
	time_e = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(time_e - time_s);
	cout << "bui: " << duration.count() << "ms" << endl;	

//	cout << eval(RF, X_test, Y_test) << endl;

// test online data stream
/*
	int n_test = Y_test.size();
	int n_correct = 0;
	vector<double> res;
	int cur, size = Y_train.size();
	long long n_add = 0, add_sum = 0, add_min = -1, add_max = -1;
	long long n_del = 0, del_sum = 0, del_min = -1, del_max = -1;
	long long n_qry = 0, qry_sum = 0, qry_min = -1, qry_max = -1;
	unordered_map<int, bool> del;
	auto start = high_resolution_clock::now(), end = start;
	
	// cerr << eval(RF, X_test, Y_test) << endl;
	
	// for (int i = 1001; i < X_train.size(); ++i) {
	// 	RF.add(X_train[i], Y_train[i]);
	// }
	
	// cerr << eval(RF, X_test, Y_test) << endl;
	// return 0;

	vector<vector<Tx>> X_test_2;
	vector<Ty> Y_test_2;
	bool print = 0;
	for (int i = 0; i < 10000; ++i) {
		const auto &X = X_test[i];
		const auto &Y = Y_test[i];
		const int B = 2;
		if (i % B == 0) {
			if ((i / B) >> 1 & 1) {
				start = high_resolution_clock::now(); {
					RF.add(X, Y);
				} end = high_resolution_clock::now();
				cur = duration_cast<microseconds>(end - start).count();
				upd(add_sum, add_min, add_max, cur);
				n_add += 1;
				size += 1;
				if (print) cerr << i << " add: " << cur << endl;
			} else {
				int id = randint(0, size - 1);
				for (; del.count(id); id = randint(0, size - 1));
				start = high_resolution_clock::now(); {
					RF.del(id);
				} end = high_resolution_clock::now();
				cur = duration_cast<microseconds>(end - start).count();
				upd(del_sum, del_min, del_max, cur);
				n_del += 1;
				del[id] = 1;
				if (print) cerr << i << " del: " << cur << endl;
			}
		} else {
			start = high_resolution_clock::now(); {
				RF.qry(X, res);
			} end = high_resolution_clock::now();
			cur = duration_cast<microseconds>(end - start).count();
			upd(qry_sum, qry_min, qry_max, cur);
			
			n_correct += (res[1] >= 0.5) == Y;
			
			n_qry += 1;
			if (print) cerr << i << " qry: " << cur << endl;
			
			// X_test_2.push_back(X);
			// Y_test_2.push_back(Y);
		}
		
		RF.clean_up();
		
		// cerr << '\t' << i << endl;
	}
	
	cout << add_sum << ' ' << del_sum << ' ' << qry_sum << endl;
	cout << "add: " << (float) add_sum / n_add << ' ' << add_min << ' ' << add_max << endl;
	cout << "del: " << (float) del_sum / n_del << ' ' << del_min << ' ' << del_max << endl;
	cout << "qry: " << (float) qry_sum / n_qry << ' ' << qry_min << ' ' << qry_max << endl;
	cout << (float) n_correct / n_qry << endl;
	
	start = high_resolution_clock::now(); {
		// RF.clean_up(1);
	} end = high_resolution_clock::now();
	cur = duration_cast<microseconds>(end - start).count();
	cerr << "clean up: " << cur << endl;
*/	
	
	// vector<vector<Tx>> X_train_2;
	// vector<Ty> Y_train_2;
	// for (int i = 0; i < RF.X.size(); ++i) {
	// 	const auto &X = RF.X[i];
	// 	const auto &Y = RF.Y[i];
	// 	if (X.size() > 1) {
	// 		X_train_2.push_back(X);
	// 		Y_train_2.push_back(Y);
	// 	}
	// }
	
	// random_forest RF2(X_train_2, Y_train_2, T, k, max_dep);
	// cout << eval(RF2, X_test_2, Y_test_2) << " but " << eval(RF, X_test_2, Y_test_2) << endl;
	
// test 1, 10, 100, .1%, 1.%
/*
	long long elapsed = 0, unlearn_count = 0;
	vector<int> id(X_train.size());
	for (int i = 0; i < id.size(); ++i) id[i] = i;
	shuffle(id.begin(), id.end(), mt);

	int n = X_train.size();

//	vector<int> cnt = {1, 3, 10, 32, 100, 316, 1000, 3162, 10000, 31623, 100000, 316227, 1000000};
	vector<int> cnt = {1, 10, 100, (int) (n * 0.001), (int) (n * 0.01)};
	vector<double> time_used = vector<double>(cnt.size(), 0.0);
	int cnt_max = *max_element(cnt.begin(), cnt.end());

	for (int c = 1; c <= cnt_max; ++c) {
		int idx = id[c - 1];
		auto start = high_resolution_clock::now();
		
		RF.del(idx);

		auto end = high_resolution_clock::now();
		elapsed += duration_cast<microseconds>(end - start).count();

		for (int i = 0; i < cnt.size(); ++i) {
			if (c == cnt[i]) time_used[i] = elapsed / 1000.0;
		}
	}

	cout << "unl: ";
	for (auto t : time_used) cout << t << ' ';
	cout << endl;
*/

// other form of unlearning

	long long elapsed = 0, unlearn_count = 0;
	vector<int> id(X_train.size());
	for (int i = 0; i < id.size(); ++i) id[i] = i;
	shuffle(id.begin(), id.end(), mt);
	
	if (time_us) {
		if (time_us < 0) {
			time_s = high_resolution_clock::now();
			
			random_forest RF(X_train, Y_train, T, T, max_dep);
			cerr << "build done" << endl;
			
			time_e = high_resolution_clock::now();

			time_us = duration_cast<microseconds>(time_e - time_s).count();
		}

		counter = 0;
		
		for (int i = 0; elapsed < time_us && i < X_train.size(); ++i, unlearn_count += 1) {
			auto start = high_resolution_clock::now();
		
			RF.del(id[i]);
			
			auto end = high_resolution_clock::now();
			elapsed += duration_cast<microseconds>(end - start).count();
		}
		if (dly) RF.develop();
		
		cout << "unlearn count: " << unlearn_count << endl;
	} else if (n_unlearn) {
		if (mode == 1) {
			cerr << "batch unlearning enabled" << endl;
			vector<int> del(n_unlearn);
			for (int i = 0; i < n_unlearn && i < X_train.size(); ++ i) {
				del[i] = id[i];
			}
			auto start = high_resolution_clock::now();
			
			RF.del(del, dly);
			
			auto end = high_resolution_clock::now();
			elapsed += duration_cast<microseconds>(end - start).count();
		} else {
			for (int i = 0; i < n_unlearn; ++i) {
				auto start = high_resolution_clock::now();

				RF.del(id[i]);

				auto end = high_resolution_clock::now();
				elapsed += duration_cast<microseconds>(end - start).count();
			}
		}

		if (dly) {
			auto start = high_resolution_clock::now();
			RF.develop();
			auto end = high_resolution_clock::now();
			elapsed += duration_cast<microseconds>(end - start).count();
		}
		
		cout << "unl " << n_unlearn << ": " << (double) elapsed / 1000 << " ms" << endl;
	}

//	cout << "unl_acc: " << eval(RF, X_test, Y_test, 0) << endl;

	if (!time_us && !n_unlearn) return 0;
	return 0;

	vector<vector<Tx>> X_2;
	vector<int> Y_2;
	for (int i = n_unlearn; i < X_train.size(); ++i) {
		int idx = id[i];
		X_2.push_back(X_train[idx]);
		Y_2.push_back(Y_train[idx]);
	}

	random_forest RF2(X_2, Y_2, T, k, max_dep);
	cout << "nai_acc: " << eval(RF2, X_test, Y_test) << endl;

}
