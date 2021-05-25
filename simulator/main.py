from argparse import ArgumentParser
import numpy as np
from estimators import estimatorSelector
from joblib import Parallel, delayed
from rewards import generateData, getSlateReward, getPolicyReward


def runEstimatorFixedTensor(rTensor, estimatorId, mu):
    """
    Helper function
    :param rTensor:
    :param estimatorId:
    :param mu:
    :return:
    """
    slates, rewards = generateData(N, rTensor, mu)
    rHat = estimatorSelector(targetPolicy, slates, rewards, estimatorId, mu)
    return rHat


# input arguments
parser = ArgumentParser(description='Simulation with synthetic data for different slate OPE estimators')
parser.add_argument("-r", "--reward", default="Mway", type=str, help="reward type", required=False)
parser.add_argument("-m", "--mway", default=2, type=int, help="reward type", required=False)
parser.add_argument("-k", "--numslots", default=3, type=int, help="number of slots", required=False)
parser.add_argument("-d", "--numactions", default=10, type=int, help="number of actions per slot", required=False)
parser.add_argument("-n", "--samplesize", default=600, type=str, help="sample size", required=False)
parser.add_argument("-s", "--numsimulations", default=1000, type=int,
                    help="number of runs to estimate MSE for a given (N,K,D) tuple", required=False)

args = parser.parse_args()

rewardKind = args.reward
Mway = args.mway
K = args.numslots
D = args.numactions
N = args.samplesize
numRunsSimulation = args.numsimulations

# setup intermediate variables
numCandidatesArray = [D] * K  # D: number of actions per slot in array format
mu = [np.asarray([1 / 10] * D)] * K
targetPolicy = tuple([0] * len(numCandidatesArray))  # target policy is (wlog) fixed to (0,0,...)

print('Drawing random tensors...')
numSampledTensors = 20
rTensors = []
for _ in range(numSampledTensors):
    rTensors.append(getSlateReward(numCandidatesArray, rewardKind, Mway))

print("N=%s, dims=%s" % (N, numCandidatesArray))

# run estimators
all_estimators = ["PI", "wPI", "PIsingleCV", "PImultiCV"]
for estimatorId in all_estimators:

    all_mse = []
    all_sse = []
    for i in range(numSampledTensors):
        rTensor = rTensors[i]

        resultParallel = Parallel(n_jobs=-1)(
            delayed(runEstimatorFixedTensor)(rTensor, estimatorId, mu) for _ in range(numRunsSimulation))

        expectedPolicyReward = getPolicyReward(targetPolicy, rTensor)

        mse = np.mean((resultParallel - expectedPolicyReward) ** 2)
        sse = np.std((resultParallel - expectedPolicyReward) ** 2)

        all_mse.append(mse)
        all_sse.append(sse)

    mse = np.mean(all_mse)
    sse = np.mean(all_sse)

    logrmse = np.log10(mse) / 2
    sse = sse / (2 * mse * np.sqrt(numRunsSimulation))

    print(estimatorId, round(logrmse, 3), round(sse, 3))
