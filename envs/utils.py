import numpy as np


## Sampleing Random Variable ##
def get_sample(distribution='uniform'):
    if distribution == 'uniform':
        return sample_uniform_distribution()
    elif distribution == 'exponential':
        return sample_exponential_distribution()
    elif distribution == 'beta':
        return sample_beta_distribution()
    else:
        assert False, f"Invalid distribution: {distribution}"


def sample_uniform_distribution():
    # mean=0.5
    return np.random.uniform()


def sample_exponential_distribution(scale=0.5):
    # mean=0.5 for scale=0.5
    return np.random.exponential(scale)


def sample_beta_distribution(a=2, b=2):
    # mean=0.5 for (a,b) = (2,2)
    return np.random.beta(a, b)


## Getting A Decay
def get_decay(type, t, lam=0.1):
    if type == 'exponential':
        return exponential_decay(t, lam)
    elif type == 'linear':
        return linear_decay(t, lam)
    else:
        assert Fase, f"Invalid decay type: {type}"


def exponential_decay(t, lam=0.1):
    return np.exp(-lam * t)


def linear_decay(t, lam=0.1):
    return (1 - t * lam)