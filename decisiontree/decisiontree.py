"""Decision Tree Classification

Requirements
    Python 3.x

Interface Functions
    loadFeatureMap(filename)
    loadData(filename)
    constructDecisionTree(data, featureMap, continuousFeatures)
    predict(tree, record, featureMap)
    accuracy(tree, data, featureMap)
    prune(tree, data, featureMap)

Interface Classes
    Bagging(sampleSize, classifiersCount)
        > construct(data, featureMap, continuousFeatures)
        > predict(row, featureMap)
        > accuracy(data, featureMap)
    Boosting(iterationsCount, sampleSize)
        > construct(data, featureMap, continuousFeatures)
        > predict(row, featureMap)
        > accuracy(data, featureMap)
"""

from collections import Counter
from math import log
from pprint import pprint

import os
import random


# Comparison functions used to run a data point through decison tree
conditionFunc = {
    "==" : lambda target, test: test == target,
    "<=" : lambda target, test: test <= target,
    ">"  : lambda target, test: test >  target
}


def loadFeatureMap(path, prefix=None):
    """Load dataset.names file.

    Args
        path: string, absolute/relative path to .names file
        prefix: string, prefix to path (eg: common directory)
    Returns
        dict, feature name-index mapping
    """
    if prefix:
        path = os.path.join(prefix, path)
    print("File Loaded: ", path)
    with open(path, "r") as fp:
        lines = fp.readlines()
        names = [f.strip() for f in lines[1].split(",")]
        names.append("outcome")  # last column name is outcomes by default
        
        featureMap = dict([(f, i) for i, f in enumerate(names)])
        featureMap["outcome"] = -1  # this is more flexible approach..
        return featureMap

def loadData(path, prefix=None):
    """Load dataset.data file

    Args
        path: string, absolute/relative path to .names file
        prefix: string, prefix to path (eg: common directory)
    Returns
        2-Dimensional list of data; last value is outcome
    """
    if prefix:
        path = os.path.join(prefix, path)
    print("File Loaded: ", path)
    with open(path, "r") as fp:
        data = []
        for line in fp:
            data.append([float(i) for i in line.split(",")])
        return data


def entropy(array):
    """Calculate entropy of a list of values. Entropy is
    calculated on a subset of outcomes in this module.
    
    Args
        array: 1-Dimensional list of integers/floats/strings
    Returns
        float, entropy of the values
    """
    total = len(array) * 1.0
    value = 0.0
    for v in Counter(array).values():
        prob = v / total
        value += -prob * log(prob, 2)
    return value


def gain(outcome, feature):
    """Calculate information gain between outcome and feature. 
    Information gain or Mutual information is not commutative.
    
    Args
        outcome: 1-Dimensional list of elements of any size K > 0
        feature: 1-Dimensional list of elements of size K
    Returns
        float, information gain IG (Outcome ; Feature)
        i.e., Entropy(Outcome) - Probability(Feature-Val) * Entropy(Feature-Val)
            for-all feature values that correspond to featureIndex in data
    """
    # frequency of feature elements
    frequency = Counter(feature)
    totalSize = len(feature) * 1.0

    # outcome
    informationGained = entropy(outcome)
    
    # entropy per feature-value
    # subset of outcome in which feature = featureValue
    for featureValue, count in frequency.items():
        subset = [o for f, o in zip(feature, outcome) if f == featureValue]
        informationGained -= (len(subset) / totalSize) * entropy(subset)
    
    return informationGained


def satisfiesConditions(row, featureMap, initialized={}):
    """Check if a record satisfies certain initialized conditions.

    Args
        row: 1-D list, one vector of features
        featureMap: dict, feature name-index mapping
        initialized: dict, feature name-value mapping
    Returns
        boolean, true if the initialized values are true in the row
    """
    for name, (operator, targetValue) in initialized.items():
        index = featureMap[name]
        testValue = row[index]
        if not conditionFunc[operator](targetValue, testValue):
            return False
    return True


def nextFeature(data, featureMap, continuous=[], initialized={}):
    """Determine the feature that gives maximum information gain.
    
    Args
        data: 2-Dimensional list
        featureMap: dict, feature and feature-index mapping
        continuous: list, features that are numeric (non-categorical)
        initialized: dict, feature name-value mapping
    Returns
        tuple (string, float, list)
            string, feature name that gives max gain
            float, information gain corresponding to the feature
            list, conditions for sub-trees
    """
    # prepare a subset of data where initialized feature values are true
    subset = [row for row in data 
        if satisfiesConditions(row, featureMap, initialized)]

    # identify the next feature that is not yet initialized
    maxFeature = None  # feature with max gain
    maxGain = None  # max gain
    maxSplitConditions = None  # split conditions of this feature
    
    # features to search for the next best..
    remainingFeatures = featureMap.keys() - initialized.keys() - set(["outcome"])
    for nextFeature in remainingFeatures:
        featureIndex = featureMap[nextFeature]
        outcomeIndex = featureMap["outcome"]

        # For continuous (precisely, ordinal) variables, identify split
        # condition that gives maximum gain
        if nextFeature in continuous:
            # sorted data by next-feature values
            values = []
            outcomes = []
            for row in sorted(subset, key=lambda row: row[featureIndex]):
                values.append(row[featureIndex])
                outcomes.append(row[outcomeIndex])

            # list of indexes where consecutive values are not equal
            splitIndexes = [ i
                for i in range(len(values)-1) if values[i] != values[i+1]]

            # pick the index that gives max gain
            nextGain = 0
            nextIndex = None
            nextSplitConditions = None

            for idx in splitIndexes:
                discretizedValues = [int(i <= idx) for i in range(len(values))]
                idxGain = gain(outcomes, discretizedValues)
                if idxGain > nextGain:
                    nextGain = idxGain
                    nextIndex = idx
                    nextSplitConditions = [("<=", values[idx]), (">", values[idx])]

        # if the next feature is categorical, simply calculate the gain 
        # on the outcomes and feature values in the subset
        else:
            values = []
            outcomes = []
            for row in subset:
                values.append(row[featureIndex])
                outcomes.append(row[outcomeIndex])
            
            nextGain = gain(outcomes, values)
            nextSplitConditions = [("==", v) for v in set(values)]
            
        if maxFeature is None:
            maxFeature = nextFeature
            maxGain = nextGain
            maxSplitConditions = nextSplitConditions

        if maxGain < nextGain:
            maxFeature = nextFeature
            maxGain = nextGain
            maxSplitConditions = nextSplitConditions

    return maxFeature, maxGain, maxSplitConditions


def constructDecisionTree(data, featureMap, continuousFeatures=[]):
    """Construct decision tree using ID3 algorithm.
    
    Args
        data: 2-Dimensional list, data
        featureMap: dict, feature name-index mapping
        continuousFeatures: list, features that are non-categorical
    Returns
        dict, decision tree.
    """
    # tree is a dictionary; each node has the following information
    #
    # Non-Leaf Nodes:
    #   1) feature                       : feature-name corresponding the node
    #   2) children                      : { split-condition: { sub-tree } }
    #
    # Leaf Nodes:
    #   1) outcome                       : [ target-class ]
    #
    # Split Condition is a tuple
    #    (operator,              <=, >, ==
    #     value)                 number
    tree = {}

    # each queue element is a tuple
    #   (sub-tree,   sub-tree object to be constructed
    #    init)       feature-name and split-condition map to initialize subtree
    queue = [ (tree, {}) ]
    
    while len(queue):
        # pick a node in the tree
        node, init = queue.pop(0)

        # pick the next best feature
        feature, gain, splitConditions = nextFeature(data, featureMap,
            continuousFeatures, init)
        
        # is this a leaf node?
        if not gain:
            index = featureMap["outcome"]
            node["outcome"] = set([ row[index] for row in data 
                if satisfiesConditions(row, featureMap, init) ])
            continue
    
        # assign feature to the node
        node["feature"] = feature
        
        # create child nodes (i.e., sub-tree for each split condition)
        node["children"] = {}
        for condition in splitConditions:
            child = {}

            # update initialization for child
            childInit = dict(init)
            childInit[feature] = condition
            
            # add child to the parent. i.e., condition vs. sub-tree mapping
            node["children"][condition] = child

            # put the child in the queue
            queue.append([child, childInit])
    
    # return the root of the tree
    return tree


def predict(tree, row, featureMap):
    """Predict outcome of the row.
    
    Args
        tree: dict, decision tree
        row: 1-Dimensional list, an instance of data containing all features
        featureMap: feature-name and feature-index map
    Returns
        list, predicted outcomes; typically list size is 1
    """
    # i.e., loop until leaf node; leaf does not have children
    while tree.get("children"):
        name = tree["feature"]
        value = row[featureMap[name]]    
        for (operator, targetValue) in tree["children"]:
            if conditionFunc[operator](targetValue, value):
                tree = tree["children"][(operator, targetValue)]
    return list(tree["outcome"])


def accuracy(tree, data, featureMap):
    """Accuracy of the decision tree on the data in range [0,1].
    
    Args
        tree: dict, decision tree
        data: 2-Dimensional list, list of data instances (a subset)
        featureMap: dict, feature-name and feature-index mapping
    Returns
        tuple (total accuracy, outcome accuracy)
        float, [0,1] total accuracy
        float, [0,1] outcome wise accuracy
    """
    outcomeIndex = featureMap["outcome"]
    correct = 0.0
    correctOutcome = dict([(row[outcomeIndex],0) for row in data])
    
    count = len(data)
    countOutcome = Counter([row[outcomeIndex] for row in data])
    
    for row in data:
        predicted = predict(tree, row, featureMap)
        # if outcomes are more than 1, consider the prediction is not correct
        if len(predicted) == 1 and predicted[0] == row[outcomeIndex]:
            correct += 1
            correctOutcome[predicted[0]] += 1
    
    outcomeAccuracy = {}
    for k in countOutcome:
        outcomeAccuracy[k] = correctOutcome[k] / countOutcome[k]

    # total-accuracy, class-accuracy
    return correct / count, outcomeAccuracy


def prune(tree, data, featureMap):
    """Reduced error pruning on a decision tree.

    Args
        tree: dict, decision tree 
        data: 2-D list, a list of feature vectors
        featureMap: dict, feature name-index mapping
    Returns
        updated tree (not-copy) of pruned decision tree
    """
    queue = [(tree, {})]
    sequence = []

    while len(queue):
        # pick the most recent node from the queue
        # node and feature initialization conditions for that sub-tree
        node, init = queue.pop(0)

        # add to the traversal sequence
        sequence.insert(0, (node, init))

        # add children to the queue
        for condition, subtree in node.get("children", {}).items():
            name = node.get("feature")
            if name:
                subtreeInit = dict(init)
                subtreeInit[name] = condition
                queue.insert(0, (subtree, subtreeInit))

    while len(sequence):
        # node and its initialization conditions
        subtree, init = sequence.pop(0)

        # prepare a subset for the subtree
        subset = [row for row in data 
            if satisfiesConditions(row, featureMap, init)]

        if len(subset) == 0:
            continue

        # get subtree accuracy
        subtreeAccuracy, _ = accuracy(subtree, subset, featureMap)

        # accuracy of most-popular class as leaf
        idx = featureMap["outcome"]
        counts = Counter([row[idx] for row in subset])
        popularClass = max(counts, key=counts.get)
        classAccuracy = counts[popularClass] / len(subset) * 1.0

        # prune if accuracy of most-popular class is higher or equal..
        # i.e., replace subtree with most-popular class as leaf
        if classAccuracy >= subtreeAccuracy:
            subtree.pop("children", None)
            subtree.pop("feature", None)
            subtree["outcome"] = set([popularClass])

    return tree


class Bagging(object):
    """Ensemble of decision trees to determine outcome by voting."""

    def __init__(self, sampleCount, classifierCount):
        """Initialize Bagging Decision Tree Constructor.

        Args
            sampleCount: integer, the size of sample to construct classifier
            classifierCount: integer, number of classifiers to use
        """
        self.sampleCount = sampleCount
        self.classifierCount = classifierCount
        self.classifiers = []

    def construct(self, data, featureMap, continuousFeatures=[]):
        """Construct a bagging (boostraped) decision tree classifier.

        Args
            data: 2-Dimensional list, data
            featureMap: dict, feature name-index mapping
            continuousFeatures: list, features that are non-categorical
        Returns
            Bagging object, for convenience
        """
        for _ in range(self.classifierCount):
            sample = random.choices(data, k=self.sampleCount)  # sample with replacement
            tree = constructDecisionTree(sample, featureMap, continuousFeatures)
            self.classifiers.append(tree)
        return self

    def predict(self, row, featureMap):
        """Predict outcome of a row.

        Args
            row: 1-Dimensional list, an instance of the data
            featureMap: feature name-index mapping
        Returns
            Bagging object, for convenience
        """
        # count predicted outcome vs number of times predicted
        counts = {}
        for tree in self.classifiers:
            outcome = predict(tree, row, featureMap)[0]
            counts[outcome] = counts.get(outcome, 0) + 1

        # pick the most predicted outcome
        finalOutcome = None
        finalCount = None
        for outcome, count in counts.items():
            if finalCount is None or finalCount < count:
                finalOutcome = outcome
                finalCount = count
        return finalOutcome

    def accuracy(self, data, featureMap):
        """Predict accuracy of classifier on a subset.

        Args
            data: 2-D list, list of feature vectors
            featureMap: dict, feature name-index mapping
        Returns
            tuple (float, dict)
                total accuracy of classifier,
                class-level accuracy of classifier
        
        Pretty much same as __main__.accuracy, with slight change!
        """
        index = featureMap["outcome"]
        correct = 0.0
        correctOutcome = dict([(row[index],0) for row in data])
        
        count = len(data)
        countOutcome = Counter([row[index] for row in data])
        
        for row in data:
            predicted = self.predict(row, featureMap)
            if predicted == row[index]:
                correct += 1
                correctOutcome[predicted] += 1
        
        outcomeAccuracy = {}
        for k in countOutcome:
            outcomeAccuracy[k] = correctOutcome[k] / countOutcome[k]

        # total-accuracy, class-accuracy
        return correct / count, outcomeAccuracy


class Boosting(object):
    """Adaboost: Ensemble of decision trees to determine outcome by
        adaptive weighted voting."""

    def __init__(self, iterationsCount, sampleCount):
        """Create instance of Adaboost Decision Tree Constructor.

        Args
            interationsCount: integer, number of interations, degree of optimization
            sampleCount: integer, sample size to construct tree
        """
        self.iterationsCount = iterationsCount
        self.sampleCount = sampleCount
        self.trees = []
        self.alphas = []

    def construct(self, data, featureMap, continuousFeatures=[]):
        """Construct a bagging (boostraped) decision tree classifier.

        Args
            data: 2-Dimensional list, data
            featureMap: dict, feature name-index mapping
            continuousFeatures: list, features that are non-categorical
        Returns
            Boosting object, for convenience
        """

        # initialize weights
        weights = [1./len(data) for _ in range(len(data))]

        # add index in to the data.. its required to keep track of 
        # original instance in a sample, to update weights.
        indexedData = []
        for i, row in enumerate(data):
            indexedData.append((i, row))

        # at each iteration, construct a decision tree on a sample
        # that is created based on weights updated in previous iteration..
        for _ in range(self.iterationsCount):
            # construct decision tree on a sample, chosen by weights..
            sample = random.choices(indexedData, weights=weights, k=self.sampleCount)
            tree = constructDecisionTree([s[1] for s in sample], featureMap, continuousFeatures)
            self.trees.append(tree)

            # predict outcomes
            predicted = [predict(tree, row[1], featureMap)[0] for row in sample]
            target = [row[1][featureMap["outcome"]] for row in sample]

            # sum of weights of misclassified rows
            error = 0
            for sampleIndex, (dataIndex, row) in enumerate(sample):
                if predicted[sampleIndex] != target[sampleIndex]:
                    error += weights[dataIndex]

            if error == 0:
                self.alphas.append(0)
            else:
                # update weights of misclassified rows
                alpha = (1.-error) / error       
                self.alphas.append(alpha)
                for sampleIndex, (dataIndex, row) in enumerate(sample):
                    if predicted[sampleIndex] != target[sampleIndex]:
                        weights[dataIndex] *= alpha

                # normalize weights
                tot = sum(weights)
                weights = [w / tot for w in weights]

    def predict(self, row, featureMap):
        """Predict outcome of a row.

        Args
            row: 1-Dimensional list, an instance of the data
            featureMap: feature name-index mapping
        Returns
            Boosting object, for convenience
        """
        # weighted voting..
        total = {}
        for i in range(self.iterationsCount):
            alpha = self.alphas[i]
            predicted = predict(self.trees[i], row, featureMap)[0]
            target = row[featureMap["outcome"]]
            total[target] = total.get(target, 0) + alpha * int(predicted == target)

        # outcome with max voting
        finalOutcome = None
        maxValue = None
        for outcome, value in total.items():
            if maxValue is None or maxValue < value:
                finalOutcome = outcome
                maxValue = value        
        return finalOutcome

    def accuracy(self, data, featureMap):
        """Predict accuracy of classifier on a subset.

        Args
            data: 2-D list, list of feature vectors
            featureMap: dict, feature name-index mapping
        Returns
            tuple (float, dict)
                total accuracy of classifier,
                class-level accuracy of classifier        
        """
        index = featureMap["outcome"]
        correct = 0.0
        correctOutcome = dict([(row[index],0) for row in data])
        
        count = len(data)
        countOutcome = Counter([row[index] for row in data])
        
        for row in data:
            predicted = self.predict(row, featureMap)
            if predicted == row[index]:
                correct += 1
                correctOutcome[predicted] += 1
        
        outcomeAccuracy = {}
        for k in countOutcome:
            outcomeAccuracy[k] = correctOutcome[k] / countOutcome[k]

        # total-accuracy, class-accuracy
        return correct / count, outcomeAccuracy


def runBaggingExample():
    # load files
    featureMap = loadFeatureMap("pen/pen2.names")
    data = loadData("pen/pen3.csv")

    # list of non-categorical feature names (i.e., all except outcome)
    continuousFeatures = list(featureMap.keys() - set(["outcome"]))

    # create classifier
    # 1) sample size is at most 80% of the total data size
    # 2) total number of classifiers is 5
    clf = Bagging(sampleCount=int(.80 * len(data)), classifierCount=5)
    clf.construct(data, featureMap, continuousFeatures)
    
    # accuracy
    resp = clf.accuracy(data, featureMap)
    print("Accuracy Range [0,1]         ", resp[0])
    print("Class Accuracy Range [0,1]   ", resp[1])


def runBoostingExample():
    # load files
    featureMap = loadFeatureMap("pen/pen2.names")
    data = loadData("pen/pen3.csv")

    # list of non-categorical feature names (i.e., all except outcome)
    continuousFeatures = list(featureMap.keys() - set(["outcome"]))

    # create classifier
    # 1) sample size is at most 80% of the total data size
    # 2) number of iterations is 100
    clf = Boosting(iterationsCount=100, sampleCount=int(.80 * len(data)))
    clf.construct(data, featureMap, continuousFeatures)
    
    # accuracy
    resp = clf.accuracy(data, featureMap)
    print("Accuracy Range [0,1]         ", resp[0])
    print("Class Accuracy Range [0,1]   ", resp[1])


def runPruningExample():
    # load files
    featureMap = loadFeatureMap("pen/pen.names")
    trainData = loadData("pen/pen-30.csv")
    testData = loadData("pen/pen-test.csv")
    pruneData = loadData("pen/pen-prune.csv")

    # list of non-categorical feature names (i.e., all except outcome)
    continuousFeatures = list(featureMap.keys() - set(["outcome"]))

    # create decision tree
    tree = constructDecisionTree(trainData, featureMap, continuousFeatures)
    
    # before and after prining
    # comment this out.. for "effects of pruning.."
    prune(tree, pruneData, featureMap)

    # accuracy
    resp = accuracy(tree, testData, featureMap)
    print("Accuracy Range [0,1]         ", resp[0])
    print("Class Accuracy Range [0,1]   ", resp[1])


def runSimpleExample():
    # load names
    featureMap = loadFeatureMap("wdbc/wdbc.names")
    trainData = loadData("wdbc/wdbc-train.data")
    testData = loadData("wdbc/wdbc-test.data")

    # feature-names in `.names` file that represent continuous values
    # empty list implies all features are categorical
    continuousFeatures = list(featureMap.keys() - set(["outcome"]))

    # create decision tree
    tree = constructDecisionTree(trainData, featureMap, continuousFeatures)
    
    # print decision tree
    pprint(tree)
    
    # predict
    resp = accuracy(tree, testData, featureMap)
    print("Accuracy Range [0,1]         ", resp[0])
    print("Class Accuracy Range [0,1]   ", resp[1])

if __name__ == "__main__":
    runSimpleExample()
    #runPruningExample()
    #runBaggingExample()
    #runBoostingExample()
