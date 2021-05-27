import numpy
from argparse import ArgumentParser
import pathlib
import Datasets
import Settings
import Policy
import Metrics

# input arguments
parser = ArgumentParser(description='Simulation with real data MSLR_WEB30K for different slate OPE estimators')
parser.add_argument("-n", "--samplesize", default=5000, type=int, help="sample size", required=False)
parser.add_argument("-m", "--maxactions", default=10, type=int, help="keep a subset of maximum M allowed docs",
                    required=False)
parser.add_argument("-k", "--numslots", default=5, type=int, help="sample size", required=False)
parser.add_argument("-r", "--metric", default="NDCG", type=str,
                    help="Metric. Valid options are: NDCG, ERR, SumRelevance, MaxRelevance, DCG", required=False)
parser.add_argument("-s", "--numsimulations", default=1000, type=int,
                    help="number of runs to estimate MSE for a given (N,K,D) tuple", required=False)

args = parser.parse_args()

SAMPLE_SIZE = args.samplesize
M = args.maxactions
L = args.numslots
METRIC = args.metric
iterations = args.numsimulations

resetSeed = 387

print("sample size = %s" % SAMPLE_SIZE)

# load dataset

trainDataset = Datasets.Datasets()
DATA_DIR = str(pathlib.Path(__file__).parent.absolute())
trainDataset.loadTxt(DATA_DIR + '/Datasets/mslr/mslr.txt', 'MSLR')

anchorURLFeatures, bodyTitleDocFeatures = Settings.get_feature_sets("MSLR")

numpy.random.seed(resetSeed)
detLogger = Policy.DeterministicPolicy(trainDataset, 'tree', False)  # without replacement
detLogger.train(anchorURLFeatures, 'url')
detLogger.filterDataset(M)  # keep a subset of maximum M allowed docs
data = detLogger.dataset
del detLogger
loggingPolicy = Policy.UniformPolicy(data, True)  # with replacement
targetPolicy = Policy.DeterministicPolicy(data, 'tree', False)
targetPolicy.train(bodyTitleDocFeatures, 'body')

if METRIC == 'DCG':
    metric = Metrics.DCG(data, L)
elif METRIC == 'NDCG':
    metric = Metrics.NDCG(data, L, True)  # True for replacements
elif METRIC == 'ERR':
    metric = Metrics.ERR(data, L)
elif METRIC == 'SumRelevance':
    metric = Metrics.SumRelevance(data, L)
elif METRIC == 'MaxRelevance':
    metric = Metrics.MaxRelevance(data, L)

numQueries = len(data.docsPerQuery)

# find all queries that have exactly M valid docs
# and precompute the corresponding target policy rankings
J = []
Rankings = []
for i in range(numQueries):
    if data.docsPerQuery[i] == M:
        J.append(i)
        Rankings.append(targetPolicy.predict(i,L))

trueMetric=[]
for i in J:
    trueMetric.append(metric.computeMetric(i, targetPolicy.predict(i, L)))
    if i%100==0:
        print(".", end="", flush=True)
print("", flush=True)
target=numpy.mean(trueMetric)
print("Parallel:main [LOG] *** TARGET: ", target, flush = True)
del trueMetric

SE = numpy.zeros((iterations, 4))
for iteration in range(iterations):

    numpy.random.seed(resetSeed + 7 * iteration)

    Y = numpy.zeros(L)
    sumY1 = numpy.zeros(L)
    sumY1sq = numpy.zeros(L)
    sumG = 0.0
    sumGR = 0.0
    sumGRY1 = numpy.zeros(L)
    sumGRG1 = 0.0
    
    sampleSize = 0
    while sampleSize < self.SAMPLE_SIZE:

        sampleSize += 1
        random_index = int(numpy.random.randint(0, len(J)))

        currentQuery = J[random_index]
        newRanking   = Rankings[random_index]

        loggedRanking=loggingPolicy.predict(currentQuery, L)
        loggedValue=metric.computeMetric(currentQuery, loggedRanking)

        Y = M * ((newRanking == loggedRanking) * 1.0)
        sumY1 += (Y - 1)
        sumY1sq += (Y - 1) ** 2

        G = (Y - 1).sum() + 1
        sumG += G

        GR = G * loggedValue
        sumGR += GR
        sumGRY1 += GR * (Y - 1)
        sumGRG1 += GR * (G - 1)

    PI = sumGR / sampleSize
    SNPI = sumGR / sumG

    beta = sumGRG1 / sum(sumY1sq)
    PIsingleCV = PI - beta * sum(sumY1) / sampleSize

    weights = sumGRY1 / sumY1sq
    PICV = PI - sum(weights * sumY1) / sampleSize

    SE[iteration] = [(PI - target) ** 2, (SNPI - target) ** 2, (PIsingleCV - target) ** 2, (PICV - target) ** 2]
    print(".", end="", flush=True)

stds = numpy.std(SE, axis=0)

log10rmses = numpy.log10(numpy.sqrt(numpy.mean(SE, axis=0)))
print("\n log10(sqrt(mean(SE))):")
print(log10rmses)

# To get confidence intervals (2 standard errors) for the log(RMSE), for plotting, we use the delta method: 
# CI = 2*stds / (2*numpy.mean(SE)*numpy.sqrt(iterations)) / numpy.log(10)
