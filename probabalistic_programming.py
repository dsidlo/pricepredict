import pymc as pm
import numpy as np

# Let's imagine we're trying to model the height of adults in a population

# Data (heights in cm of 100 people)
heights = np.array([170, 165, 180, 175, 172, 168, 177, 183, 162, 179] * 10)  # Simplified example

# Define the model
with pm.Model() as height_model:
    # Prior for the mean height (mu)
    mu = pm.Normal('mu', mu=170, sigma=10)

    # Prior for the standard deviation (sigma)
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Likelihood or observation model - assuming heights follow a normal distribution
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=heights)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000)

# Now, trace contains samples from our posterior distributions
# We can summarize these samples to understand our model's predictions
print(pm.summary(trace))

# If you want to visualize:
import matplotlib.pyplot as plt

pm.plot_posterior(trace, var_names=['mu', 'sigma'])
plt.show()
