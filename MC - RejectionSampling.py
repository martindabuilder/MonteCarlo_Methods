import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

#chosen function to decide whether a point will be accepted or rejected
target = beta(6, 2)
x_line = np.linspace(0, 1, 300)

#a limit to the highest point of the function
M = target.pdf(x_line).max() * 1.05

#50000 samples
N = 50000
x_proposal = np.random.uniform(0, 1, N)

#for each sample we create a random new value between 0 and 1
u = np.random.uniform(0, M, N)

#checks if the sample is above/below the beta function
mask = u < target.pdf(x_proposal)

accepted = x_proposal[mask] #everything below is accepted
rejected = x_proposal[~mask] #everything above is denied
rate = mask.mean()

print(f"Acceptance rate: {rate:.1%}")

fig1, ax1 = plt.subplots(figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))

#shows all the samples, beta func and the distribution that will be chosen
ax1.scatter(rejected[:3000], u[~mask][:3000],s=2, alpha=0.3, color="red",  label="Rejected")
ax1.scatter(accepted[:3000], u[mask][:3000],s=2, alpha=0.4, color="blue", label="Accepted")
ax1.plot(x_line, target.pdf(x_line), "k--",  lw=2, label="Target: Beta(6,2)")
ax1.axhline(M, color="green", ls="--", lw=1.5, label=f"M = {M:.2f}")
ax1.fill_between(x_line, target.pdf(x_line), alpha=0.3, color="blue")
ax1.set_title(f"Rejection Sampling  (rate={rate:.1%})")
ax1.legend(markerscale=4)

#histogram that displays the count and form of accepted samples
ax2.hist(accepted, bins=60, density=True, color="blue", alpha=0.5, label=f"samples (n={len(accepted):,})")
ax2.plot(x_line, target.pdf(x_line), "k--", lw=2, label="Beta(6,2)")
ax2.set_title("Histogram of accepted samples")
ax2.legend()

plt.show()