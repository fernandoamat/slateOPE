import numpy as np
import itertools

"""
File containing implementation of different reward functions
"""

def getMwaySlateReward(numCandidatesArray, M):
    K = len(numCandidatesArray)
    indices = [_ for _ in itertools.combinations(range(K), M)]

    # generate rewards for each slot tuple
    pairReward = []
    for idx in indices:
        rew = np.random.normal(loc=0.5 / K, scale=0.1 / K, size=[numCandidatesArray[_] for _ in idx])  # Gauss draws
        #         # modified reward model
        #         if M==1 and idx[0]==0:
        #             rew *= np.asarray([0.5**k for k in range(len(rew))])
        #         else:
        #             rew *= 0.01
        pairReward.append(rew)

    reward = np.zeros(numCandidatesArray, dtype=np.float64)
    for index, _ in np.ndenumerate(reward):
        ss = np.sum([pairReward[_][tuple(index[x] for x in idx)] for _, idx in enumerate(indices)])
        assert (ss <= 1)
        reward[index] = ss
    return reward


def getSlateReward(numCandidatesArray, rewardType, Mway):
    """
    This is the method you will call for simulations to get the K-Tensor with the reward for each possible slate
    It is just a selector based on methods implemented above
    """
    if rewardType == "Mway":
        return getMwaySlateReward(numCandidatesArray, Mway)
    else:
        raise ValueError('{} not implemented'.format(rewardType))


def getPolicyReward(policy, rewardTensor):
    """
    Given a policy and a reward tensor this method returns the reward of that policy
    @param policy: tuple defining a non-contextual policy: it indicates which action is chosen for each slot in the slate
    """
    assert (len(rewardTensor.shape) == len(policy))
    return rewardTensor[policy]


def generateData(numSamples, rewardTensor, mu):
    """
    Generate random synthetic data
    @param numSamples: number of samples N to generate
    @param rewardTensor: K-Tensor containing the reward for each possible slate realization
    @return slateSamples: NxK integer matrix: contains the slate selected on each sample ~ mu[K,[...]]
    @return rewardSample: N binary vector with reward for each sample. 0 -> fail; 1-> success
    """
    slateDimensions = rewardTensor.shape  # number of candidates per position in the slate
    slateSize = len(slateDimensions)
    slateSamples = np.zeros((numSamples, slateSize), dtype=np.int32)
    for i in range(slateSize):
        slateSamples[:, i] = np.random.choice(len(mu[i]), numSamples, p=mu[i])

    rewardSample = np.zeros(numSamples)
    tf = [rewardTensor[tuple(slateSamples[_])] for _ in range(numSamples)]

    rewardSample[tf > np.random.rand(numSamples)] = 1.0
    return slateSamples, rewardSample
