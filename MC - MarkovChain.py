import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(7)
target = beta(4, 7)
def log_target(x):
    if x <= 0 or x >= 1:
        return -np.inf
    return target.logpdf(x)

N = 50000
burn_in = 5000
step = 0.1
chain = np.zeros(N)
chain[0] = 0.5
accepted = 0

for i in range(1, N):
    current  = chain[i-1]
    proposal = current + np.random.normal(0, step)
    log_ratio = log_target(proposal) - log_target(current)
    if np.log(np.random.rand()) < log_ratio:
        chain[i] = proposal
        accepted += 1
    else:
        chain[i] = current

samples = chain[burn_in:]
accept_rate = accepted / (N - 1)

print(f"Acceptance rate: {accept_rate:.1%}")

x_line = np.linspace(0, 1, 300)

fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot(chain[:20000], lw=0.7, color="steelblue", alpha=0.7)
ax1.scatter(0, chain[0], color="orange", alpha = 0.7, s=80, label="Starting point")
ax1.axvline(burn_in, color="red", ls="--", lw=1.5, label="end of burn-in")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Value")
ax1.set_title("First 20000 steps")
ax1.legend()

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.hist(samples, bins=60, density=True,color="steelblue", alpha=0.6, label=f"Accepted samples({len(samples):,})")
ax2.plot(x_line, target.pdf(x_line), "k--",lw=2, label="Target Beta(2,5)")
ax2.set_title(f"MCMC Histogram (acceptance rate={accept_rate:.1%})")
ax2.legend(loc="upper right")

plt.show()