from typing import Any



import numpy as np
import matplotlib.pyplot as plt
import tqdm
T = 7
sigma = 1.0
S0 = 100
gamma = 0.01
V = np.zeros((2*T+1, T+1))
def Delta(lam):
    return max(0,lam+1)
def C(lam):
    if lam >= -1:
        return 1/2 * np.exp(-(lam+1))
    else:
        return -lam/2
def dp_step(q, V_qm1, V_q, V_qp1):
    return V_q - gamma/2 * q**2 + C(V_q - V_qm1) + C(V_q - V_qp1)
for n in reversed(range(T)):
    for q in range(T-n, T+n+1):
        if q <0: 
            print(q)
        V[q,n] = dp_step(q,V[q-1,n+1],V[q,n+1], V[q+1,n+1])
print(np.round(V,1))

def run_market_simulation(T=7, sigma=1.0, S0=100, gamma=0.01, delta_a=2.0, delta_b=2.0):
    q = np.zeros(T+1).astype(int)
    c = np.zeros(T+1)
    cash = np.zeros(T+1)
    pnl = np.zeros(T+1)
    S = np.zeros(T+1)
    S[0] = S0
    delta_a_arr = np.zeros(T)
    delta_b_arr = np.zeros(T)
    f = lambda x: 1/2*np.exp(-x)
    c_sum = 0 
    for n in range(T):  
        delta_a_opt = Delta(V[q[n]+T,n+1]-V[q[n]+T-1,n+1]) 
        delta_b_opt = Delta(V[q[n]+T,n+1]-V[q[n]+T+1,n+1])
        delta_a_arr[n] = delta_a_opt
        delta_b_arr[n] = delta_b_opt
        p_ask_opt = f(delta_a_opt)
        p_bid_opt = f(delta_b_opt)
        p_none_opt = 1 - p_ask_opt - p_bid_opt
        event_opt = np.random.choice(['ask', 'bid', 'none'], p=[p_ask_opt, p_bid_opt, p_none_opt])
        eta_a_opt = 1 if event_opt == 'ask' else 0
        eta_b_opt = 1 if event_opt == 'bid' else 0
        q[n+1] = q[n] - eta_a_opt + eta_b_opt
        cash[n+1] = cash[n] + eta_a_opt * (S[n] + delta_a_opt) - eta_b_opt * (S[n] - delta_b_opt)
        c[n] = delta_a_opt * eta_a_opt + delta_b_opt * eta_b_opt - gamma/2 * q[n]**2 #idea de Carlos *2
        pnl[n+1] = cash[n+1] + S[n] * q[n+1]
        S[n+1] = S[n]+np.random.normal(0,sigma)
    c_sum = sum(c)
    return pnl, q, cash, S, delta_a_arr, delta_b_arr,c_sum
def simulate_multiple_runs(num_simulations=10000):
    T = 7
    sigma = 1.0
    S0 = 100
    gamma = 0.01
    delta_a = 2.0
    delta_b = 2.0
    all_final_pnls = []
    all_deltas_a = []
    all_deltas_b = []
    sample_q = None
    for i in range(num_simulations):
        pnl, q, cash, S, delta_a_arr, delta_b_arr, c_sum = run_market_simulation(T, sigma, S0, gamma, delta_a, delta_b)
        all_final_pnls.append(c_sum)
        if i == 0:
            # For demonstration, sample the delta array and q for the first run
            sample_deltas_a = delta_a_arr
            sample_deltas_b = delta_b_arr
            sample_q = q
    all_final_pnls = np.array(all_final_pnls)
    percent_positive = np.sum(all_final_pnls > 0) / num_simulations * 100
    print(f"Positive profit occurrences: {np.sum(all_final_pnls > 0)}/{num_simulations} ({percent_positive:.1f}%)")
    print(f"Mean final c_sum: {np.mean(all_final_pnls):.2f}\nStddev: {np.std(all_final_pnls):.2f}")
    plt.figure(figsize=(10, 6))
    plt.hist(all_final_pnls, bins=40, alpha=0.75, color='skyblue', edgecolor='k')
    plt.axvline(0, color='r', linestyle='--', label='Zero Profit')
    plt.title(f"Distribution of Final C_sums over {num_simulations} Simulations")
    plt.xlabel("Final PnL")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    # Plot optimal deltas for the first run as a function of time
    plt.figure(figsize=(12,6))
    plt.plot(np.arange(T), sample_deltas_a, label='Optimal Ask Delta', color='blue')
    plt.plot(np.arange(T), sample_deltas_b, label='Optimal Bid Delta', color='orange')
    plt.ylabel('Optimal Delta')
    plt.xlabel('Time Step')
    plt.legend()
    plt.title('Optimal Deltas vs Time for the First Simulation')
    plt.show()
    # Optional: Plot delta vs q
    plt.figure(figsize=(12,6))
    plt.scatter(sample_q[:-1], sample_deltas_a, label='Ask Delta vs q', alpha=0.6, color='blue', s=20)
    plt.scatter(sample_q[:-1], sample_deltas_b, label='Bid Delta vs q', alpha=0.6, color='orange', s=20)
    plt.xlabel('Inventory q')
    plt.ylabel('Optimal Delta')
    plt.legend()
    plt.title('Optimal Delta vs Inventory (First Simulation)')
    plt.show()

simulate_multiple_runs(num_simulations=10000)
print(V[T,0])
print("a tomar por culo")