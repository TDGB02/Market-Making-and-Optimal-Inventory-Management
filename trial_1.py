import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
# PARAMETERS
T = 100  # Time steps
sigma = 1.0  # Volatility
S0 = 100  # Initial price
q0 = 0      # Initial inventory
X0 = 0      # Initial cash
k = 0.01     # Impact coefficient for prob function

gamma = 0.0001  # Risk aversion penalty

delta_a = 2.0  # Ask spread (can try to optimize this later)
delta_b = 2.0  # Bid spread

# Probability functions
f = lambda delta: max(0, 0.5 - k * delta)  # probability of execution at each side, linear decrease

np.random.seed(42)  # for reproducibility

S = [S0]
q = [q0]
X = [X0]
f_pnl_list = [X0+S0*q0]

for n in range(T):
    # Simulate new stock price
    S_new = S[-1] + np.random.normal(0, sigma)
    S.append(S_new)

    # Calculate probabilities
    p_ask = f(delta_a)
    p_bid = f(delta_b)
    # Adjust to ensure at most one fills per timestep
    p_both = 0  # Impossible, as specified
    p_none = 1 - p_ask - p_bid
    # Randomly choose outcome
    event = np.random.choice(['ask', 'bid', 'none'], p=[p_ask, p_bid, p_none])

    eta_a = 1 if event == 'ask' else 0  # Sold at ask
    eta_b = 1 if event == 'bid' else 0  # Bought at bid
    print(eta_a, eta_b)
    # Update inventory and cash
    q_new = q[-1] - eta_a + eta_b
    X_new = X[-1] + eta_a*(S[-1]+delta_a) - eta_b*(S[-1]-delta_b)
    q.append(q_new)
    X.append(X_new)
    f_pnl = X_new +S_new*q_new
    f_pnl_list.append(f_pnl)
    print(q_new, X_new , f_pnl_list)

# Risk penalty
inventory_penalty = gamma/2 * sum(np.array(q[:-1])**2)  # sum from n=0 to T-1

# Final PnL: cash at end + value of inventory at end - risk penalty
final_pnl = X[-1] + q[-1]*S[-1] - inventory_penalty

print(f"Final PnL: {final_pnl:.2f}")
print(f"Final inventory: {q[-1]}, Final cash: {X[-1]:.2f}, Final price: {S[-1]:.2f}")


# Optionally plot results
try:
    
    plt.figure(figsize=(12,5))
    plt.subplot(4,1,1)
    plt.plot(S, label='Stock Price')
    plt.title('Stock Price Path')
    plt.legend()
    plt.subplot(4,1,2)
    plt.plot(X, label='Cash')
    plt.title('Cash over Time')
    plt.legend()
    plt.subplot(4,1,3)
    plt.plot(q, label='Inventory')
    plt.title('Inventory over Time')
    plt.legend()
    plt.subplot(4,1,4)
    plt.plot(f_pnl_list, label='Inventory')
    plt.title('Fake PnL over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
except ImportError:
    pass


#test the optimal solution: 
def Bellman_constant(lam):
    def objective(delta):
        return -((delta - lam) * f(delta))  # Negate for maximization
    # You need to specify a reasonable upper bound. 50/k is a practical cap for the given f.
    res = minimize_scalar(objective, bounds=(0, 50/k), method='bounded')
    if res.success:
        return -res.fun  # Don't forget to flip the sign!
    # Fallback or error handling if not successful
    raise RuntimeError("Optimization failed")

#backward induction: 

def optimal_value(gamma, p_ask, T):
    return 2*sum(Bellman_constant(gamma*p_ask/2) for lam in range(T))

    