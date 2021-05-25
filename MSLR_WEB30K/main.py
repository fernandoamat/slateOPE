import subprocess
import numpy

SAMPLE_SIZE = 5000
M = 10
L = 5
METRIC = 'NDCG'
#METRIC = 'ERR'
#METRIC = 'SumRelevance'
#METRIC = 'MaxRelevance'
#METRIC = 'DCG'

resetSeed=387

import Datasets
import Settings
import Policy
import Metrics
import Estimators

print("sample size = %s" % SAMPLE_SIZE)

trainDataset=Datasets.Datasets()
trainDataset.loadTxt(Settings.DATA_DIR+'mslr/mslr.txt', 'MSLR')

anchorURLFeatures, bodyTitleDocFeatures=Settings.get_feature_sets("MSLR")

numpy.random.seed(resetSeed)
detLogger=Policy.DeterministicPolicy(trainDataset, 'tree', False)   # without replacement
detLogger.train(anchorURLFeatures, 'url')
detLogger.filterDataset(M)   # keep a subset of maximum M allowed docs
data=detLogger.dataset
del detLogger
loggingPolicy=Policy.UniformPolicy(data, True)  # with replacement
targetPolicy=Policy.DeterministicPolicy(data, 'tree', False)
targetPolicy.train(bodyTitleDocFeatures, 'body')

if METRIC == 'DCG':
    metric=Metrics.DCG(data, L)
elif METRIC == 'NDCG':
    metric=Metrics.NDCG(data, L, True)    # True for replacements
elif METRIC == 'ERR':
    metric=Metrics.ERR(data, L)
elif METRIC == 'SumRelevance':
    metric=Metrics.SumRelevance(data, L)
elif METRIC == 'MaxRelevance':
    metric=Metrics.MaxRelevance(data, L)

numQueries=len(data.docsPerQuery)
trueMetric=[]
for i in range(numQueries):
    if data.docsPerQuery[i] == M:
        trueMetric.append(metric.computeMetric(i, targetPolicy.predict(i, L)))
        if i%100==0:
            print(".", end="", flush=True)
print("", flush=True)
target=numpy.mean(trueMetric)
print("Parallel:main [LOG] *** TARGET: ", target, flush = True)
del trueMetric

iterations = 1000
SE = numpy.zeros((iterations, 4))
for iteration in range(iterations):

    numpy.random.seed(resetSeed + 7*iteration)

    Y = numpy.zeros(L)
    sumY1 = numpy.zeros(L)
    sumY1sq = numpy.zeros(L)
    sumG = 0.0
    sumGR = 0.0
    sumGRY1 = numpy.zeros(L)
    sumGRG1 = 0.0

    sampleSize = 0
    for j in range(SAMPLE_SIZE):
        currentQuery=numpy.random.randint(0, numQueries)

        if data.docsPerQuery[currentQuery] == M:
            sampleSize += 1

            loggedRanking=loggingPolicy.predict(currentQuery, L)
            loggedValue=metric.computeMetric(currentQuery, loggedRanking)

            newRanking=targetPolicy.predict(currentQuery,L)

            Y = M * ((newRanking==loggedRanking)*1.0)
            sumY1 += (Y-1)
            sumY1sq += (Y-1)**2

            G = (Y-1).sum() + 1
            sumG += G

            GR = G * loggedValue
            sumGR += GR
            sumGRY1 += GR * (Y-1)
            sumGRG1 += GR * (G-1)

    PI = sumGR / sampleSize
    SNPI = sumGR / sumG

    beta = sumGRG1 / sum(sumY1sq)
    PIsingleCV = PI - beta * sum(sumY1) / sampleSize

    weights = sumGRY1 / sumY1sq
    PICV = PI - sum(weights * sumY1) / sampleSize

    SE[iteration] = [(PI-target)**2, (SNPI-target)**2, (PIsingleCV-target)**2, (PICV-target)**2]

    print(".", end = "", flush = True)

final_sample_size = sampleSize
print("final sample size = %s" % sampleSize)

stds = numpy.std(SE,axis=0)
print("std(SE):")
print(stds)

log10rmses = numpy.log10(numpy.sqrt(numpy.mean(SE,axis=0)))
print("log10(sqrt(mean(SE))):")
print(log10rmses)

results = dict()
for i in inputs:
    if not i.node_failed:
        results[str(i.final_sample_size)] = (i.log10rmses, i.stds)
print(results)

print("M=%s, L=%s, METRIC=%s" % (M,L,METRIC))

