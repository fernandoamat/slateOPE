import numpy as np

"""
File containing implementation of different estimators

policy is a K-tuple indicating the non-contextual policy we want to evaluate
slateSamples is an NxK matrix generated with method generateData
R is an N vector generated with method generateData
mu is the logging policy
"""


def PI(policy, slateSamples, R, mu):
    slateDimensions = np.max(slateSamples, axis=0) + 1
    K = len(slateDimensions)
    assert (len(policy) == K)
    Y = np.zeros(slateSamples.shape, dtype=np.float64)
    for i in range(K):
        Y[:, i] = (policy[i] == slateSamples[:, i]) / mu[i][np.asarray(slateSamples[:, i])]
    G = 1 + np.sum(Y - 1, axis=1)
    return np.mean(G * R)


def wPI(policy, slateSamples, R, mu):
    slateDimensions = np.max(slateSamples, axis=0) + 1
    K = len(slateDimensions)
    assert (len(policy) == K)
    Y = np.zeros(slateSamples.shape, dtype=np.float64)
    for i in range(K):
        Y[:, i] = (policy[i] == slateSamples[:, i]) / mu[i][np.asarray(slateSamples[:, i])]
    G = 1 + np.sum(Y - 1, axis=1)
    return np.mean(G * R) / np.mean(G)


def PIsingleCV(policy, slateSamples, R, mu):
    slateDimensions = np.max(slateSamples, axis=0) + 1
    K = len(slateDimensions)
    assert (len(policy) == K)
    Y = np.zeros(slateSamples.shape, dtype=np.float64)
    for i in range(K):
        Y[:, i] = (policy[i] == slateSamples[:, i]) / mu[i][np.asarray(slateSamples[:, i])]
    G = 1 + np.sum(Y - 1, axis=1)
    beta = np.sum(G * R * (G - 1)) / np.sum(np.sum((Y - 1) ** 2, axis=0))
    # print("beta = %s" % beta)
    cv = beta * np.sum(np.mean(Y - 1, axis=0))
    return np.mean(G * R) - cv


def PImultiCV(policy, slateSamples, R, mu):
    num_samples = slateSamples.shape[0]
    slateDimensions = np.max(slateSamples, axis=0) + 1
    K = len(slateDimensions)
    assert (len(policy) == K)
    Y = np.zeros(slateSamples.shape, dtype=np.float64)
    for i in range(K):
        Y[:, i] = (policy[i] == slateSamples[:, i]) / mu[i][np.asarray(slateSamples[:, i])]
    G = 1 + np.sum(Y - 1, axis=1)
    GR = np.reshape(G * R, (num_samples, 1))
    weights = np.sum(GR * (Y - 1), axis=0) / np.sum((Y - 1) ** 2, axis=0)
    #print("weights = %s" % weights)
    cv = np.mean(Y - 1, axis=0) @ weights.T
    return np.mean(GR) - cv


def estimatorSelector(policy, slateSamples, reward, estimatorName, mu):
    if estimatorName == "PI":
        return PI(policy, slateSamples, reward, mu)
    elif estimatorName == "wPI":
        return wPI(policy, slateSamples, reward, mu)
    elif estimatorName == "PIsingleCV":
        return PIsingleCV(policy, slateSamples, reward, mu)  # NEW
    elif estimatorName == "PImultiCV":
        return PImultiCV(policy, slateSamples, reward, mu)  # NEW
    else:
        raise ValueError('{} not implemented'.format(estimatorName))
