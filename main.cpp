
// #include <bits/stdc++.h>
#include <vector>
#include "DynFrs.h"
#include "Disc/roc_auc.h"

using namespace std;
using namespace std::chrono;
vector<vector<Tx>> X_train, X_test;
vector<Ty> Y_train, Y_test;
string dataset;
auto t_start = high_resolution_clock::now(), t_end = t_start;

unordered_map<string, int> T_default = {
	{"Purchase",	250},
	{"Vaccine",		250},
	{"Adult",		100},
	{"Bank",		250},
	{"Heart",		150},
	{"Diabetes",	250},
	{"NoShow",		250},
	{"Synthetic",	150},
	{"Higgs",		100},
};

unordered_map<string, int> k_default = {
	{"Purchase",	10},
	{"Vaccine",		10},
	{"Adult",		10},
	{"Bank",		10},
	{"Heart",		10},
	{"Diabetes",	10},
	{"NoShow",		10},
	{"Synthetic",	10},
	{"Higgs",		10},
};

unordered_map<string, int> d_default = {
	{"Purchase",	10},
	{"Vaccine",		20},
	{"Adult",		20},
	{"Bank",		20},
	{"Heart",		15},
	{"Diabetes",	30},
	{"NoShow",		20},
	{"Synthetic",	40},
	{"Higgs",		30},
};

unordered_map<string, int> s_default = {
	{"Purchase",	30},
	{"Vaccine",		5},
	{"Adult",		30},
	{"Bank",		40},
	{"Heart",		5},
	{"Diabetes",	5},
	{"NoShow",		5},
	{"Synthetic",	30},
	{"Higgs",		20},
};

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

double eval(random_forest &RF, const vector<vector<Tx>> &X_test, const vector<Ty> &Y_test, bool auc = 0) {
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

int unlearn_time_fixed(random_forest &RF, long long time_us) {
	vector<int> id(X_train.size());
	for (int i = 0; i < id.size(); ++i) id[i] = i;
	shuffle(id.begin(), id.end(), mt);
	
	int cnt = 0;
	long long elapsed = 0;
	for (; elapsed < time_us && cnt < X_train.size(); ++cnt) {
		t_start = high_resolution_clock::now();
		RF.del(id[cnt]);
		t_end = high_resolution_clock::now();
		elapsed += duration_cast<microseconds>(t_end - t_start).count();
	}
	
	return cnt;
}

int unlearn_count_fixed(random_forest &RF, int count) {
	vector<int> id(X_train.size());
	for (int i = 0; i < id.size(); ++i) id[i] = i;
	shuffle(id.begin(), id.end(), mt);
	
	int cnt = 0;
	long long elapsed = 0;
	
	if (!dly) {
		for (; cnt < count && cnt < X_train.size(); ++cnt) {
			t_start = high_resolution_clock::now();
			RF.del(id[cnt]);
			t_end = high_resolution_clock::now();
			elapsed += duration_cast<microseconds>(t_end - t_start).count();
		}
	} else {
		vector<int> to_del;
		for (; cnt < count && cnt < X_train.size(); ++cnt) {
			to_del.push_back(id[cnt]);
		}
		t_start = high_resolution_clock::now();
		RF.del(to_del);
		t_end = high_resolution_clock::now();
		elapsed = duration_cast<microseconds>(t_end - t_start).count();
	}
	
	return elapsed;
}

void upd(long long &sum, long long &min, long long &max, int t) {
	sum += t;
	if (min == -1 || min > t) min = t;
	if (max == -1 || max < t) max = t;
}

void online_mixed_data_stream(random_forest &RF, int n_add, int n_del, int n_qry, bool print = 0) {
	vector<double> res;
	int size = Y_train.size();
	long long cur;
	long long add_sum = 0, add_min = -1, add_max = -1;
	long long del_sum = 0, del_min = -1, del_max = -1;
	long long qry_sum = 0, qry_min = -1, qry_max = -1;
	unordered_map<int, bool> del;
	
	int n_req = n_add + n_del + n_qry;
	vector<pair<int, int>> ops;
	ops.reserve(n_req);
	for (int i = 0; i < n_add; ++i) ops.push_back({0, n_qry + i});
	for (int i = 0; i < n_del; ++i) ops.push_back({1, i});
	for (int i = 0; i < n_qry; ++i) ops.push_back({2, i});
	shuffle(ops.begin(), ops.end(), mt);
	
	int n_tot = n_add + n_del + n_qry, n_correct = 0, n_test = Y_test.size();
	for (int i = 0; i < n_req; ++i) {
		int op = ops[i].first, id = ops[i].second;
		const auto &X = X_test[i];
		const auto &Y = Y_test[i];
		if (op == 0) {
			t_start = high_resolution_clock::now();
			RF.add(X, Y);
			t_end = high_resolution_clock::now();
			
			cur = duration_cast<microseconds>(t_end - t_start).count();
			upd(add_sum, add_min, add_max, cur);
			
			size += 1;
			if (print) cerr << i << " add: " << cur << endl;
		} else if (op == 1) {
			int id = randint(0, size - 1);
			for (; del.count(id); id = randint(0, size - 1));
			
			t_start = high_resolution_clock::now();
			RF.del(id);
			t_end = high_resolution_clock::now();
			
			cur = duration_cast<microseconds>(t_end - t_start).count();
			upd(del_sum, del_min, del_max, cur);
			del[id] = 1;
			if (print) cerr << i << " del: " << cur << endl;
		} else if (op == 2) {
			t_start = high_resolution_clock::now();
			RF.qry(X, res);
			t_end = high_resolution_clock::now();
			
			cur = duration_cast<microseconds>(t_end - t_start).count();
			upd(qry_sum, qry_min, qry_max, cur);
			
			n_correct += (res[1] >= 0.5) == Y;
			if (print) cerr << i << " qry: " << cur << endl;
			
		}
		RF.clean_up();
	}
	
	cout << "add request:\t" << "mean " << (float) add_sum / n_add << "\tmin " << add_min << "\tmax " << add_max << endl;
	cout << "del request:\t" << "mean " << (float) del_sum / n_del << "\tmin " << del_min << "\tmax " << del_max << endl;
	cout << "qry request:\t" << "mean " << (float) qry_sum / n_qry << "\tmin " << qry_min << "\tmax " << qry_max << endl;
	cout << (float) n_correct / n_qry << endl;
}

const bool dbg = 1;

int T = 100, k = 10, max_dep = 20, n_unlearn = 0, mode = 1;
long long time_us = 0;
string data_dir = "./";
vector<string> tasks;

void parse(int argc, char *args[]) {
	vector<string> flags;
	for (int i = 1; i < argc; ++i) {
		char *c = args[i];
		string s = "";
		for (; *c != '\0'; s += *c, ++c);
		flags.push_back(s);
	}
	
	for (int i = 0; i < flags.size();) {
		const string &tp = flags[i++];
		
		if (tp == "-data") {
			dataset = flags[i++];
			data_dir = "Datasets/" + dataset + '/';
		}
		
		else if (tp == "-auto") {
			T = T_default[dataset];
			k = k_default[dataset];
			max_dep = d_default[dataset];
			p_tries = s_default[dataset];
		} else if (tp == "-T") {
			T = atoi(flags[i++].c_str());
		} else if (tp == "-k") {
			k = atoi(flags[i++].c_str());
		} else if (tp == "-d" || tp == "-max_dep") {
			max_dep = atoi(flags[i++].c_str());
		} else if (tp == "-s") {
			p_tries = atoi(flags[i++].c_str());
		}
		
		else if (tp == "-dly" || tp == "-delay") {
			dly = atoi(flags[i++].c_str());
		} else if (tp == "-mode") {
			mode = atoi(flags[i++].c_str());
		}
		
		else if (tp == "-acc") {
			tasks.push_back(tp);
		} else if (tp == "-auc") {
			tasks.push_back(tp);
		} else if (tp == "-unl_time") {
			tasks.push_back(tp);
			tasks.push_back(flags[i++]);
		} else if (tp == "-unl_cnt") {
			tasks.push_back(tp);
			tasks.push_back(flags[i++]);
		} else if (tp == "-stream") {
			tasks.push_back(tp);
			tasks.push_back(flags[i++]);
			tasks.push_back(flags[i++]);
			tasks.push_back(flags[i++]);
		}
	}
	cerr << data_dir << ' ' << T << ' ' << k << ' ' << max_dep << ' ' << p_tries << ' ' << dly << ' ' << mode << endl;
}

int main(int argc, char *args[]) {
	srand(time(NULL));
	ios::sync_with_stdio(0);
	
	parse(argc, args);
	
	load_file(data_dir + "train.txt", X_train, Y_train);
	load_file(data_dir + "test.txt", X_test, Y_test);
	
	p_count = int(sqrt(X_train[0].size())) + 1;

	if (dbg) {
		cerr << "train size: " << X_train.size() << ' ' << X_train[0].size() << endl;
		cerr << "test size:  " << X_test.size() << ' ' << X_test[0].size() << endl;
	}
	
	t_start = high_resolution_clock::now();
	random_forest RF(X_train, Y_train, T, k, max_dep);
	t_end = high_resolution_clock::now();
	
	auto duration = duration_cast<milliseconds>(t_end - t_start);
	cout << "Build time:\t" << duration.count() << "ms" << endl;	
	
	for (int i = 0; i < tasks.size();) {
		const string &tp = tasks[i++];
		
		if (tp == "-acc") {
			double acc = eval(RF, X_test, Y_test, 0);
			cout << "Accuracy:\t" << acc << endl;
		} else if (tp == "-auc") {
			double auc = eval(RF, X_test, Y_test, 1);
			cout << "AUROC:\t\t" << auc << endl;
		} else if (tp == "-unl_time") {
			long long time_us = 1000LL * atoi(tasks[i++].c_str());
			int cnt = unlearn_time_fixed(RF, time_us);
			cout << "Unlearned:\t" << cnt << " samples within " << time_us / 1000 << " ms" << endl;
		} else if (tp == "-unl_cnt") {
			int cnt = atoi(tasks[i++].c_str());
			long long time_us = unlearn_count_fixed(RF, cnt);
			cout << "Unlearned:\t" << time_us / 1000 << " ms for " << cnt << " samples" << endl;
		} else if (tp == "-stream") {
			int n_add = atoi(tasks[i++].c_str()), n_del = atoi(tasks[i++].c_str()), n_qry = atoi(tasks[i++].c_str());
			online_mixed_data_stream(RF, n_add, n_del, n_qry);
		}
	}
	
	return 0;
}
