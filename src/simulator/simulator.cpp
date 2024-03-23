#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <time.h>
#define rep(p, q) for (int p=0; p<q; p++)
#define PI 0
#define AND 1
#define NOT 2
#define STATE_WIDTH 16
using namespace std;

int countOnesInBinary(uint64_t num, int width) {
    int count = 0;
    rep (_, width) {
        if (num & 1) {
            count++;
        }
        num >>= 1;
    }
    return count;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "Failed" << endl;
        return 1;
    }
    string in_filename = argv[1];
    string out_filename = argv[2];
    
    cout << "Read File: " << in_filename << endl;
    freopen(in_filename.c_str(), "r", stdin);
    int n, m;  // number of gates
    int no_patterns; 
    scanf("%d %d %d", &n, &m, &no_patterns);
    cout << "Number of gates: " << n << endl;

    // Graph
    vector<int> gate_list(n);
    vector<vector<int> > fanin_list(n);
    vector<vector<int> > fanout_list(n);
    vector<int> gate_levels(n);
    vector<int> pi_list;
    int max_level = 0;

    for (int k=0; k<n; k++) {
        int type, level;
        scanf("%d %d", &type, &level);
        gate_list[k] = type;
        gate_levels[k] = level;
        if (level > max_level) {
            max_level = level;
        }
        if (type == PI) {
            pi_list.push_back(k);
        }
    }
    vector<vector<int> > level_list(max_level+1);
    for (int k=0; k<n; k++) {
        level_list[gate_levels[k]].push_back(k);
    }
    for (int k=0; k<m; k++) {
        int fanin, fanout;
        scanf("%d %d", &fanin, &fanout);
        fanin_list[fanout].push_back(fanin);
        fanout_list[fanin].push_back(fanout);
    }

    int no_pi = pi_list.size();
    cout << "Number of PI: " << no_pi << endl;

    cout<<"Start Simulation"<<endl;
    // Simulation
    vector<vector<uint64_t> > full_states(n); 
    int tot_clk = 0;
    int clk_cnt = 0; 

    while (no_patterns > 0) {
        no_patterns -= STATE_WIDTH; 
        vector<uint64_t> states(n);
        // generate pi patterns 
        rep(k, no_pi) {
            int pi = pi_list[k];
            states[pi] = rand() % int(std::pow(2, STATE_WIDTH)); 
            // cout << "PI: " << pi << " " << states[pi] << endl;
        }
        // Combination
        for (int l = 1; l < max_level+1; l++) {
            for (int gate: level_list[l]) {
                if (gate_list[gate] == AND) {
                    uint64_t res = (states[fanin_list[gate][0]] & states[fanin_list[gate][1]]); 
                    states[gate] = res;
                    // cout << gate << ": " << (res & 1) << " " << (states[fanin_list[gate][0]] & 1) << " " << (states[fanin_list[gate][1]] & 1) << endl;
                }
                else if (gate_list[gate] == NOT) {
                    uint64_t res = ~states[fanin_list[gate][0]]; 
                    states[gate] = res;
                }
            }
        }
        // Record
        rep (k, n) {
            full_states[k].push_back(states[k]);
        }

    }

    // Probability 
    freopen(out_filename.c_str(), "w", stdout);
    vector<float> prob_list(n);
    rep(k, n) {
        int cnt = 0;
        int tot_cnt = 0;
        int all_bits = 0;
        rep(p, full_states[k].size()) {
            cnt = countOnesInBinary(full_states[k][p], STATE_WIDTH);
            tot_cnt += cnt;
            all_bits += STATE_WIDTH;
        }
        prob_list[k] = (float)tot_cnt / all_bits;
    }
    rep(k, n) {
        printf("%d %f\n", k, prob_list[k]);
    }

    // Sample node pair 
    vector<long> pi_cover_hash_list(n);
    rep(k, n) {
        scanf("%ld", &pi_cover_hash_list[k]);
    }
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            // 1. Must have the same PI cover hash
            if (pi_cover_hash_list[i] != pi_cover_hash_list[j])
                continue;
            // 2. Must have the similar probability
            if (std::abs(prob_list[i] - prob_list[j]) > 0.1)
                continue;
            // 3. They are not connected
            bool connected = false;
            for (int k = 0; k < fanout_list[i].size(); k++) {
                if (fanout_list[i][k] == j) {
                    connected = true;
                    break;
                }
            }
            if (connected)
                continue;
            connected = false;
            for (int k = 0; k < fanout_list[j].size(); k++) {
                if (fanout_list[j][k] == i) {
                    connected = true;
                    break;
                }
            }
            if (connected)
                continue;
            // 4. Extreme Case
            int cnt = 0;
            int all_bits = 0;
            rep(p, full_states[i].size()) {
                cnt += countOnesInBinary(~(full_states[i][p] ^ full_states[j][p]), STATE_WIDTH);
                all_bits += STATE_WIDTH; 
            }
            float tt_sim = 1 - (float)cnt / all_bits;
            if (tt_sim > 0.2 and tt_sim < 0.8)
                continue;
            printf("%d %d %f\n", i, j, tt_sim);
        }
    }

}