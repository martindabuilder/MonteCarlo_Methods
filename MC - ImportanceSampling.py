import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm

np.random.seed(42)

#target and proposal distribution
#the target is what we want to achieve
#the proposal is what we suggest to use
target   = beta(2, 5)
proposal = norm(0.3, 0.15)

#generate samples from proposal
N = 20000
samples = proposal.rvs(N)
mask = (samples > 0) & (samples < 1) #they have to be between 0 and 1
samples = samples[mask]

# adding weight to the proposed samples
w = target.pdf(samples) / proposal.pdf(samples)
w_norm = w / w.sum() #normalizes the results

#cumulative score with the weights
cumulative_score = np.cumsum(w * samples ** 2) / np.cumsum(w)
exact_score = (2 * 3) / (7 * 8)
x_line = np.linspace(0, 1, 300)

fig1, ax1 = plt.subplots(figsize=(6, 5))

print(f"Cumulative score from Normal= {cumulative_score}")
print(f"Exact score of Beta(2,5) = {exact_score}")

#target vs proposal
ax1.plot(x_line, target.pdf(x_line), color="blue", lw=2, label="Target p(x) = Beta(2,5)")
ax1.plot(x_line, proposal.pdf(x_line), color="red", lw=2, ls="--", label="Proposal q(x) = N(0.3, 0.15)")
ax1.fill_between(x_line, target.pdf(x_line), alpha=0.15, color="blue")
ax1.fill_between(x_line, proposal.pdf(x_line), alpha=0.15, color="red")
overlap = np.minimum(target.pdf(x_line), proposal.pdf(x_line))
ax1.fill_between(x_line, overlap,alpha=0.4, color="purple", label="Overlap")
ax1.set_title("Target vs Proposal")
ax1.set_xlabel("x")
ax1.set_ylabel("плътност")
ax1.legend()

#weight distribution
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(x_line, target.pdf(x_line) / proposal.pdf(x_line), color="purple", lw=2)
ax2.set_title("Importance weights  w(x) = p(x) / q(x)")
ax2.set_xlabel("x")
ax2.set_ylabel("w(x)")


plt.show()