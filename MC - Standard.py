import numpy as np
import matplotlib.pyplot as plt

#approximating P
def monte_carlo_pi(num_samples):
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)

    distance = np.sqrt(x ** 2 + y ** 2)
    inside_circle = np.sum(distance <= 1)
    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate

#sample count
num_samples = 10000

pi_estimate = monte_carlo_pi(num_samples)

print(f"Estimated value of π with {num_samples} samples: {pi_estimate}")
x = np.random.uniform(-1, 1, num_samples)
y = np.random.uniform(-1, 1, num_samples)
plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=np.sqrt(x ** 2 + y ** 2) <= 1, cmap='bwr', s=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.suptitle('Monte Carlo Simulation for π')
plt.title(f"sample count: {num_samples}" )
plt.xlabel('x')
plt.ylabel('y')
plt.show()