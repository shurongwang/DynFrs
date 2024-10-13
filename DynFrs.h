
// #include <bits/stdc++.h>
// #include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

using namespace std;
using namespace std::chrono;

// input data dtype
typedef double Tx;

// input label dtype
typedef int Ty;

// split score dtype
typedef double Ts;

const Tx eps = 1e-8;

random_device rd;
mt19937 mt(rd());

int randint(int l, int r) {
	return rand() % (r - l + 1) + l;
}

vector<int> id_T;
vector<int> id_d;

bool dly = 1;
int p_tries = 20;
int p_count = 20;

int counter = 0;

bool erase(vector<int> &vec, int a) {
	for (int i = vec.size() - 1; i >= 0; --i) if (vec[i] == a) {
		swap(vec[i], vec.back());
		vec.pop_back();
		return 1;
	}
	return 0;
}

Ts calc_score(int ls_s, int ls_1, int rs_s, int rs_1) {
	return ((Ts) (ls_s - ls_1) * ls_1 / ls_s + (Ts) (rs_s - rs_1) * rs_1 / rs_s) / (ls_s + rs_s);
}

// specialized for binary classification
class random_forest {
public:
	
	int d;
	int n;
	int C;
	vector<vector<Tx>> X;
	vector<Ty> Y;
	vector<bool> X_binary;
	vector<vector<int>> at;
	
	int T;
	int k;
	
	class decision_tree {
		public:
		
		const random_forest &RF;

		int max_dep;
		int min_split_size;

		vector<pair<Tx, Ty>> a;
		
		mt19937 mt;
		
		class attribute {
		public:
		
			decision_tree &DT;

			int d, n, n_1, sz, used_count;
			vector<int> idx;
			vector<bool> used, cons;
			vector<pair<Tx, int>> mn, mx;
			vector<int> fm;
			vector<pair<Ts, Tx>> spl;
			vector<vector<Tx>> thr;
			vector<vector<pair<int, int>>> cnt;

			attribute(decision_tree &DT, int d):
				DT(DT), d(d), sz(0), n_1(0), used_count(0),
				used(vector<bool>(d, 0)), cons(vector<bool>(d, 0)),
				mn(vector<pair<Tx, int>>(p_count)),
				mx(vector<pair<Tx, int>>(p_count)),
				fm(vector<int>(p_count, -1)), spl(vector<pair<Ts, Tx>>(p_count)),
				thr(vector<vector<Tx>>(p_count)), cnt(vector<vector<pair<int, int>>>(p_count)) {
			}
			attribute(decision_tree &DT, int d, const vector<bool> &cons):
				DT(DT), d(d), sz(0), n_1(0), used_count(0),
				used(vector<bool>(d, 0)), cons(cons),
				mn(vector<pair<Tx, int>>(p_count)),
				mx(vector<pair<Tx, int>>(p_count)),
				fm(vector<int>(p_count, -1)), spl(vector<pair<Ts, Tx>>(p_count)),
				thr(vector<vector<Tx>>(p_count)), cnt(vector<vector<pair<int, int>>>(p_count)) {
			}
			
			int new_idx() {
				int at = -1;
				if (idx.size()) {
					at = idx.back();
					idx.pop_back();
					return at;
				}
				at = sz++;
				return at;
			}
			
			int get_next() {
				uniform_int_distribution rnd(0, d - 1);
				int id = -1;
				
				for (int trial = 0; trial < p_count; ++trial) {
					id = rnd(DT.mt);
					
					// cerr << id << ": " << used[id] << ' ' << cons[id] << endl;
					
					if (used[id] | cons[id]) {
						id = -1;
						continue;
					}
					break;
				}
				// cerr << "return: " << id << endl;
				return id;
			}

			void set_as(int id, int sta) {
				// cerr << "set " << id << " as " << sta << endl;
				if (sta == 1) used[id] = 1;
				else if (sta == 2) cons[id] = 1;
			}

			pair<Ts, Tx> gen_bin(int id, int rs_s, int rs_1) {
				int at = new_idx();
				fm[at] = id;
				used_count += 1;
				used[id] = 1;
				
				auto &thr = this->thr[at];
				auto &cnt = this->cnt[at];
				auto &spl = this->spl[at];
				
				const int ls_s = n - rs_s, ls_1 = n_1 - rs_1;
				mn[at] = {0, ls_s}, mx[at] = {1, rs_s};
				
				thr.resize(1), cnt.resize(1);
				thr[0] = eps;
				cnt[0] = {ls_s, ls_1};
				spl = {calc_score(ls_s, ls_1, rs_s, rs_1), eps};
				return spl;
			}
			
			pair<Ts, Tx> gen(int id, int spl_cnt, int a_size, const vector<pair<Tx, Ty>> &a) {
				int at = new_idx();
				fm[at] = id;
				used_count += 1;
				used[id] = 1;
				
				Tx &mn = this->mn[at].first, &mx = this->mx[at].first;
				int &mn_cnt = this->mn[at].second, &mx_cnt = this->mx[at].second;
				mn = mx = a[0].first, mn_cnt = mx_cnt = 1;
				for (int i = 1; i < a_size; ++i) {
					const auto &X = a[i].first;
					if (X < mn) mn = X, mn_cnt = 1;
					else if (X > mx) mx = X, mx_cnt = 1;
					else if (X == mn) ++mn_cnt;
					else if (X == mx) ++mx_cnt;
				}
				
				auto &thr = this->thr[at];
				auto &cnt = this->cnt[at];
				auto &spl = this->spl[at];
				
				thr.resize(spl_cnt), cnt.resize(spl_cnt);
				
				uniform_real_distribution<> rnd(mn, mx);
				for (int i = 0; i < spl_cnt; ++i) thr[i] = rnd(DT.mt);
				sort(thr.begin(), thr.end());
				
				vector<int> pre_s(spl_cnt + 1, 0), pre_1(spl_cnt + 1, 0);
				for (int i = 0; i < a_size; ++i) {
					const auto &X = a[i].first;
					int pos = upper_bound(thr.begin(), thr.end(), X) - thr.begin();
					++pre_s[pos];
					if (a[i].second) ++pre_1[pos];
				}
				for (int i = 1; i < thr.size(); ++i) pre_s[i] += pre_s[i - 1], pre_1[i] += pre_1[i - 1];
				
				for (int i = 0; i < thr.size(); ++i) {
					const auto &thres = thr[i];
					int ls_s = pre_s[i], ls_1 = pre_1[i];
					Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
					cnt[i] = {ls_s, ls_1};
					
					if (!i || score < spl.first) spl = {score, thres};
				}
				return spl;
			}

			void destroy(int at) {
				int id = fm[at];
				auto &thr = this->thr[at];
				auto &cnt = this->cnt[at];
				auto &spl = this->spl[at];
				
				idx.push_back(at);
				
				// erase(used, id);
				// unused.push_back(id), shuffle_back();
				mn[at] = mx[at] = {0, -1};
				thr.clear();
				cnt.clear();
				
				fm[at] = -1;
				
				used_count -= 1;
				used[id] = 0;
			}
			
			// void ins_once(const Ty &Y) {  }
			bool add(const vector<Tx> &Xs, const Ty &Y) {
				n += 1, n_1 += Y;
				
				int destroy_cnt = 0;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					int id = fm[at];
					bool f = 0;
					const auto &X = Xs[id];
					Tx &mn = this->mn[at].first, &mx = this->mx[at].first;
					int &mn_cnt = this->mn[at].second, &mx_cnt = this->mx[at].second;
					if (X < mn) mn = X, mn_cnt = f = 1;
					else if (X > mx) mx = X, mx_cnt = f = 1;
					else if (X == mn) ++mn_cnt;
					else if (X == mx) ++mx_cnt;

					if (!f) {
						auto &thr = this->thr[at];
						auto &cnt = this->cnt[at];
						auto &spl = this->spl[at];
						for (int i = 0; i < thr.size(); ++i) {
							const auto &thres = thr[i];
							int &ls_s = cnt[i].first, &ls_1 = cnt[i].second;
							if (X <= thres) ++ls_s, ls_1 += Y;
							Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
							if (score < spl.first) spl = {score, thres};
						}
					} else {
						destroy(at);
						destroy_cnt += 1;
					}
				}
				return destroy_cnt;
			}

			int del(const vector<Tx> &X, const Ty &Y) {
				n -= 1, n_1 -= Y;
				
				int destroy_cnt = 0;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					int id = fm[at];
					bool f = 0;
					const auto &X_ = X[id];
					pair<Tx, int> &mn = this->mn[at], &mx = this->mx[at];
					if (X_ == mn.first) --mn.second, f = !mn.second;
					else if (X_ == mx.first) --mx.second, f = !mx.second;
					
					if (!f) {
						auto &thr = this->thr[at];
						auto &cnt = this->cnt[at];
						auto &spl = this->spl[at];
						for (int i = 0; i < thr.size(); ++i) {
							const auto &thres = thr[i];
							int &ls_s = cnt[i].first, &ls_1 = cnt[i].second;
							if (X_ <= thres) --ls_s, ls_1 -= Y;
							Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
							if (score < spl.first) spl = {score, thres};
						}
					} else {
						destroy(at);
						destroy_cnt += 1;
					}
				}
				
				return destroy_cnt;
			}
			
			int del(const vector<const vector<Tx>*> &Xs, const vector<Ty> &Ys) {
				n -= Ys.size();
				for (const auto &Y : Ys) n_1 -= Y;
				
				int destroy_cnt = 0;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					int id = fm[at];
					bool f = 0;
					pair<Tx, int> &mn = this->mn[at], &mx = this->mx[at];
					for (const auto &X : Xs) {
						const auto &X_ = (*X)[id];
						if (X_ == mn.first) --mn.second, f = !mn.second;
						else if (X_ == mx.first) --mx.second, f = !mx.second;
						if (f) break;
					}
					
					if (!f) {
						auto &thr = this->thr[at];
						auto &cnt = this->cnt[at];
						auto &spl = this->spl[at];
						
						vector<int> pre_s(thr.size() + 1, 0), pre_1(thr.size() + 1, 0);
						for (int i = 0; i < Ys.size(); ++i) {
							const auto &X_ = (*Xs[i])[id];
							int pos = upper_bound(thr.begin(), thr.end(), X_) - thr.begin();
							++pre_s[pos];
							if (Ys[i]) ++pre_1[pos];
						}
						for (int i = 1; i < thr.size(); ++i) pre_s[i] += pre_s[i - 1], pre_1[i] += pre_1[i - 1];
						
						for (int i = 0; i < thr.size(); ++i) {
							const auto &thres = thr[i];
							int &ls_s = cnt[i].first, &ls_1 = cnt[i].second;
							ls_s -= pre_s[i], ls_1 -= pre_1[i];
							Ts score = calc_score(ls_s, ls_1, n - ls_s, n_1 - ls_1);
							
							if (score < spl.first) spl = {score, thres};
						}
					} else {
						destroy(at);
						destroy_cnt += 1;
					}
				}
				
				return destroy_cnt;
			}
			
			pair<Tx, int> find_best_spl() {
				int attr = -1;
				pair<Ts, Tx> best;
				for (int at = 0; at < spl.size(); ++at) if (fm[at] != -1) {
					const auto &split = spl[at];
					if (attr == -1 || split.first < best.first) best = split, attr = fm[at];
				}
				return {best.second, attr};
			}

			void reset() {
				fill(used.begin(), used.end(), 0);
				// fill(cons.begin(), cons.end(), 0);
				fill(fm.begin(), fm.end(), -1);
				sz = used_count = 0;
				idx.clear();
			}
		};

		class node {
			public:
			
			const random_forest &RF;
			decision_tree &DT;

			vector<int> Xid;
			attribute A;

			pair<Tx, int> spl = {0, -1};
			Tx thres;
			int attr;
			int delay;
			
			int dep;
			node *ls, *rs;
			
			bool old = 0;
			
			node(const random_forest &RF, decision_tree &DT, int dep):
				RF(RF), DT(DT), dep(dep), delay(1), A(DT, RF.d), ls(nullptr), rs(nullptr) {
			}
			node(const random_forest &RF, decision_tree &DT, int dep, const vector<bool> &cons):
				RF(RF), DT(DT), dep(dep), delay(1), A(DT, RF.d, cons), ls(nullptr), rs(nullptr) {
			}
			
			void gen_spl(int trials) {
				bool f = !Xid.size();
				if (f) collect();
				
				pair<Ts, Tx> best;
				for (; A.used_count < trials;) {
					int p = A.get_next();
					
					// cerr << A.used_count << ' ' << p << endl;
					
					if (p == -1) break;
					
					if (RF.X_binary[p]) {
						bool constant = 1;
						int cnt_1 = 0, cnt_1_1 = 0;
						const int X_0 = (bool) RF.X[Xid[0]][p];
						for (int id : Xid) {
							const bool X = RF.X[id][p];
							const auto &Y = RF.Y[id];
							cnt_1 += X;
							if (X) cnt_1_1 += Y;
						}
						
						constant = cnt_1 == 0 || cnt_1 == Xid.size();
						if (constant) {
							// cerr << "cons" << endl;
							A.set_as(p, 2);
							continue;
						}
						
						A.set_as(p, 1);
						A.gen_bin(p, cnt_1, cnt_1_1);
					} else {
						bool constant = 1;
						const Tx &X_0 = RF.X[Xid[0]][p];
						int a_size = 0;
						DT.a.reserve(Xid.size());
						for (int id : Xid) {
							const auto &X = RF.X[id][p];
							const auto &Y = RF.Y[id];
							DT.a[a_size++] = make_pair(X, Y);
							constant &= X == X_0;
						}
						
						if (constant) {
							// cerr << "cons" << endl;
							A.set_as(p, 2);
							continue;
						}
						
						A.set_as(p, 1);
						A.gen(p, p_tries, a_size, DT.a);
					}
				}
				
				
				if (f) Xid.erase(Xid.begin(), Xid.end());
			}

			bool split(bool dbg = 0) {
				delay = 0;
				if (leaf()) return 0;
				
				gen_spl(p_count);
				spl = A.find_best_spl();
				thres = spl.first, attr = spl.second;
				
				return attr != -1;
			}
			
			node* new_child() {
				node *u;
				if (DT.trash.empty()) {
					u = new node(RF, DT, dep + 1, A.cons);
				} else {
					u = DT.trash.back();
					DT.trash.pop_back();
					if (!u->Xid.empty()) u->Xid.clear();
					u->Xid = {};
					u->dep = dep + 1;
					u->A.reset();
					u->A.cons = A.cons;
					u->spl = {0, -1};
					u->delay = 1;
					u->ls = u->rs = nullptr;
					
					u->old = 1;
				}
				return u;
			}

			void separate() {
				ls = new_child();
				rs = new_child();
				
				thres = spl.first, attr = spl.second;
				
				ls->Xid.reserve(Xid.size());
				rs->Xid.reserve(Xid.size());
				
				for (int id : Xid) {
					const auto &X = RF.X[id][attr];
					if (X <= thres) {
						ls->Xid.push_back(id);
					} else {
						rs->Xid.push_back(id);
					}
				}
				
				int &cnt = ls->A.n_1 = 0;
				for (int id : ls->Xid) if (RF.Y[id]) ++cnt;
				ls->A.n = ls->Xid.size();
				rs->A.n = rs->Xid.size();
				rs->A.n_1 = A.n_1 - cnt;
				
				Xid.clear();
				Xid.shrink_to_fit();
			}
			
			void build() {
				if (!split()) return;
				separate();
				ls->build();
				rs->build();
			}
			
			void destroy() {
				if (ls != nullptr) ls->destroy();
				if (rs != nullptr) rs->destroy();
				Xid.clear();
				free(ls);
				free(rs);
			}
			
			void concentrate() {
				if (ls == nullptr) return;
				Xid.reserve(A.n);
				concentrate(Xid);
				ls = rs = nullptr;
			}
			void concentrate(vector<int> &Xids) {
				if (ls == nullptr) {
					if (Xids.empty()) Xids = std::move(Xid);
					else Xids.insert(Xids.end(), make_move_iterator(Xid.begin()), make_move_iterator(Xid.end()));
				} else {
					ls->concentrate(Xids);
					rs->concentrate(Xids);
					DT.trash.push_back(ls);
					DT.trash.push_back(rs);
				}
			}
			
			void collect() {
				if (ls == nullptr) return;
				Xid.reserve(A.n);
				collect(Xid);
			}
			void collect(vector<int> &Xids) {
				if (ls == nullptr) {
					Xids.insert(Xids.end(), Xid.begin(), Xid.end());
				} else {
					ls->collect(Xids);
					rs->collect(Xids);
				}
			}
			
			double qry(const vector<Tx> &X) {
				if (delay) {
					if (split()) separate();
				}
				if (ls == nullptr) return (double) A.n_1 / A.n;
				return (X[attr] <= thres) ? ls->qry(X) : rs->qry(X);
			}

			void add_leaf(int id) {
				if (ls == nullptr) Xid.push_back(id);
				else (RF.X[id][attr] <= thres) ? ls->add_leaf(id) : rs->add_leaf(id);
			}
			void add(int id) {
				const auto &X = RF.X[id];
				const auto &Y = RF.Y[id];
				if (A.add(X, Y)) gen_spl(p_count);
				if (delay) return;
				else if (ls == nullptr && !leaf()) {
					if (split()) separate();
				} else if (best_split_changed()) {
					concentrate();
					if (dly) delay = 2;
					else build();
				} else if (ls != nullptr) {
					(RF.X[id][attr] <= thres) ? ls->add(id) : rs->add(id);
				}
			}

			void del_leaf(int id) {
				if (ls == nullptr) erase(Xid, id);
				else (RF.X[id][attr] <= thres) ? ls->del_leaf(id) : rs->del_leaf(id);
			}
			void del(int id) {
				const auto &X = RF.X[id];
				const auto &Y = RF.Y[id];
				if (A.del(X, Y)) gen_spl(p_count);
				if (delay) return;
				else if (ls != nullptr && leaf()) {
					concentrate();
				} else if (best_split_changed()) {
					concentrate();
					if (dly) delay = 2;
					else build();
				} else if (ls != nullptr) {
					(RF.X[id][attr] <= thres) ? ls->del(id) : rs->del(id);
				}
			}

			void del_leaf(const vector<int> &ids) {
				if (ids.size() <= 5) {
					for (int id : ids) del_leaf(id);
					return;
				}
				
				if (ls == nullptr) {
					int del_cnt = 0, n = Xid.size();
					unordered_map<int, bool> to_del;
					for (int id : ids) to_del[id] = 1;
					for (int i = 0; i + del_cnt < n; ++i) {
						int id = Xid[i];
						if (to_del.count(id)) {
							del_cnt += 1;
							swap(Xid[i], Xid[n - del_cnt]);
							--i;
						}
						if (del_cnt == ids.size()) break;
					}
					for (int i = 0; i < del_cnt; ++i) Xid.pop_back();
				} else {
					vector<int> ids_l, ids_r;
					ids_l.reserve(ids.size());
					ids_r.reserve(ids.size());
					for (int id : ids) {
						const auto &X = RF.X[id][attr];
						if (X <= thres) {
							ids_l.push_back(id);
						} else {
							ids_r.push_back(id);
						}
					}
					if (!ids_l.empty()) ls->del_leaf(ids_l);
					if (!ids_r.empty()) rs->del_leaf(ids_r);
				}
			}
			void del(const vector<int> &ids) {
				int n = ids.size();
				if (n <= 5) {
					for (int id : ids) del(id);
					return;
				}
				vector<const vector<Tx>*> Xs;
				vector<Ty> Ys;
				Xs.reserve(n), Ys.reserve(n);
				for (int id : ids) {
					Xs.emplace_back(&RF.X[id]);
					Ys.emplace_back(RF.Y[id]);
				}
				if (n * 3 > A.n) {
					concentrate();
					if (dly) delay = 1;
					else build();
					return;
				}
				if (A.del(Xs, Ys)) gen_spl(p_count);
				
				if (delay) return;
				else if (ls != nullptr && leaf()) {
					concentrate();
				} else if (best_split_changed()) {
					concentrate();
					if (dly) delay = 1;
					else build();
				} else if (ls != nullptr) {
					vector<int> ids_l, ids_r;
					ids_l.reserve(n);
					ids_r.reserve(n);
					for (int id : ids) {
						const auto &X = RF.X[id][attr];
						if (X <= thres) {
							ids_l.push_back(id);
						} else {
							ids_r.push_back(id);
						}
					}
					if (!ids_l.empty()) ls->del(ids_l);
					if (!ids_r.empty()) rs->del(ids_r);
				}
			}

			
			bool best_split_changed() {
				// cerr << "see if best changed" << endl;
				pair<Tx, int> best = A.find_best_spl();
				// cerr << dep << ": " << best.first << ' ' << best.second << " <- " << this->spl.first << ' ' << this->spl.second << endl;
				// cerr << calc(best) << ' ' << calc(this->spl) << endl;
//				cerr << "best is found!" << ' ' << (this->spl != best) << endl;
				bool f = this->spl != best;
				// cerr << f << endl;
				// if (f) cerr << dep << ": " << best.first << ' ' << best.second << " <- " << this->spl.first << ' ' << this->spl.second << endl;
				this->spl = best;
				return f;
			}
			
			bool leaf() {
				return dep >= DT.max_dep || A.n < DT.min_split_size || A.n_1 == 0 || A.n_1 == A.n;
			}
		};
		node *root;
		
//		decision_tree(): RF(unusable_forest) {}
		decision_tree(const random_forest &RF, vector<int> &Xid, int max_dep, int min_split_size):
			RF(RF), max_dep(max_dep), min_split_size(min_split_size), a(Xid.size()) {
			root = new node(RF, *this, 0);
			root->Xid = std::move(Xid);
			
			trash.reserve(100000);
			
			#pragma omp critical
			{
				this->mt = mt19937(rd());
			}

			int &cnt = root->A.n_1;
			for (int id : root->Xid) if (RF.Y[id]) ++cnt;
			root->A.n = root->Xid.size();
			root->build();
		}
		
		double qry(const vector<Tx> &X) {
			return root->qry(X);
		}
		
		void add(int id) {
			root->add_leaf(id);
			root->add(id);
		}
		
		void del(int id) {
			root->del_leaf(id);
			root->del(id);
		}
		
		void del(const vector<int> &ids) {
			root->del_leaf(ids);
			root->del(ids);
		}
		// void del(node *u, const vector<int> &ids) {
		// 	if (ids.size() <= 3) {
		// 		for (int id : ids) u->del(id);
		// 		return;
		// 	}
		// 	u->del(ids);
		// 	if (u->delay) return;
		// 	else if (u->leaf()) {
		// 		// u->destroy_children();
		// 	}
		// 	else if (u->Xid.size() > 1000 && ids.size() * 3 >= u->Xid.size()) {
		// 		// u->destroy_children();
		// 		if (dly) u->delay = 1;
		// 		else u->separate(), u->ls->build(), u->rs->build();
		// 	}
		// 	else if (u->best_split_changed()) {
		// 	//	cerr << "tagged " << u->Xid.size() << endl;
		// 		// u->destroy_children();
		// 		if (dly) u->delay = 2;
		// 		else u->separate(), u->ls->build(), u->rs->build();
		// 	}
		// 	else if (u->ls != nullptr) {
		// 		vector<int> ids_l, ids_r;
		// 		for (int id : ids) RF.X[id][u->attr] <= u->thres ? ids_l.push_back(id) : ids_r.push_back(id);
		// 		// cerr << ids.size() << " -> " << ids_l.size() << ' ' << ids_r.size() << endl;
		// 		if (ids_l.size()) del(u->ls, ids_l);
		// 		if (ids_r.size()) del(u->rs, ids_r);
		// 	}
		// }

		
		void develop() { develop(root); };
		void develop(node *u) {
			if (u->delay) {
				// counter += u->Xid.size();
				if (u->delay == 1) {
					u->build();
				} else {
					u->separate();
					u->ls->build();
					u->rs->build();
				}
				return;
			}
			if (u->ls != nullptr) develop(u->ls);
			if (u->rs != nullptr) develop(u->rs);
		}

		void clean_up() {
			for (auto u : trash) delete u;
			trash.clear();
		}

		vector<node*> trash;
	};
	decision_tree **tr;
	//vector<decision_tree> tr;
	
	random_forest() {}
	random_forest(vector<vector<Tx>> X, vector<Ty> Y, int T = 100, int k = 10, int max_dep = 15, int min_split_size = 10):
		X(X), Y(Y), T(T), k(k) {
		n = X.size(), d = X[0].size();
		C = *max_element(Y.begin(), Y.end()) + 1;
		X_binary = vector<bool>(d, 0);
	
		id_d.resize(d);
		
		unordered_map<Tx, bool> to;
		for (int p = 0; p < d; ++p) {
			to.clear();
			for (int i = 0; i < n; ++i) {
				Tx cur = X[i][p];
				if (!to.count(cur)) to[cur] = to.size();
				if (to.size() > 2) break;
			}
			if (to.size() == 2) X_binary[p] = 1;
		}
		
		vector<vector<int>> Xids;
		data_distribution(Xids);
		for (int i = 0; i < d; ++i) id_d[i] = i;
		at.resize(n);
		tr = new decision_tree*[T];

		for (int t = 0; t < T; ++t) {
			for (int i : Xids[t]) at[i].push_back(t);
		}
		
		#pragma omp parallel for schedule(static)
		for (int t = 0; t < T; ++t) {
			tr[t] = new decision_tree(*this, Xids[t], max_dep, min_split_size);
		}
	}
	
	void data_distribution(vector<vector<int>> &Xids) {
		id_T.resize(T);
		for (int i = 0; i < T; ++i) id_T[i] = i;
		
		Xids.resize(T);
		for (int i = 0; i < n; ++i) {
			shuffle(id_T.begin(), id_T.end(), mt);
			for (int j = 0; j < k; ++j) {
				int t = id_T[j];
				Xids[t].push_back(i);
			}
		}
	}

	void qry(const vector<Tx> &X, vector<double> &res) {
		// cerr << "qry start" << endl;
		res.resize(C);
		for (int c = 0; c < C; ++c) res[c] = 0;
		double &r_0 = res[0], &r_1 = res[1];

		#pragma omp parallel for schedule(dynamic)
		for (int t = 0; t < T; ++t) {
			double p = tr[t]->qry(X);
			#pragma omp atomic
			r_0 += 1 - p;
			#pragma omp atomic
			r_1 += p;
		}
		
		for (int c = 0; c < C; ++c) res[c] /= T;
		
		// cerr << "qry end" << endl;
	}

	void add(const vector<Tx> &X, const Ty &Y) {
		int id = this->X.size();
		this->X.push_back(X);
		this->Y.push_back(Y);
		
		shuffle(id_T.begin(), id_T.end(), mt);
		at.push_back({});

		for (int j = 0; j < k; ++j) {
			int t = id_T[j];
			at[id].push_back(t);
		}
		
		#pragma omp parallel for schedule(static)
		for (int j = 0; j < k; ++j) {
			int t = id_T[j];
			tr[t]->add(id);
		}
	}
	
	void del(const vector<Tx> &X, const Ty &Y) {
		for (int i = 0; i < X.size(); ++i) {
			if (this->X[i] == X && this->Y[i] == Y) del(i);
		}
	}
	void del(int id) {
		const vector<int> &tid = at[id];
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < tid.size(); ++i) {
			tr[tid[i]]->del(id);
		}
		X[id] = vector<Tx>(0), Y[id] = 0;
	}
	
	void del(const vector<int> &ids, bool clean_tags = 0) {
		vector<vector<int>> del_ids(T);
		for (int id : ids) {
			for (int tid : at[id]) {
				del_ids[tid].push_back(id);
			}
			at[id].clear();
		}
		for (int t = 0; t < T; ++t) {
			if (del_ids[t].size()) tr[t]->del(del_ids[t]);
		}
		
		if (clean_tags) develop();
	}

	void develop() {
		for (int t = 0; t < T; ++t) tr[t]->develop();
	}
	
	void clean_up(bool force = 0) {
		long long sum = 0;
		#pragma omp parallel for schedule(dynamic)
		for (int t = 0; t < T; ++t) {
			#pragma omp atomic
			sum += tr[t]->trash.size();
		}
		if (!force && sum < 1000000) return;
		#pragma omp parallel for schedule(dynamic) 
		for (int t = 0; t < T; ++t) {
			tr[t]->clean_up();
		}
	}
};
