Software Prerequisites
======================
Python 3.x


How to Execute
==============

Call the following functions

runSimpleExample()
    Construct decision tree, print tree, print accuracy

runPruningExample()
    Construct decision tree, prune, print accuracy 

runBaggingExample()
    Construct bagging ensemble, print accuracy

runBoostingExample()
    Construct boosting ensemble, print accuracy


High Level Interfaces
=====================

The program file provide documentation to all functions/classes in docstrings. 
Following are the high-level interfacing functions/classes.

Functions
---------

loadFeatureMap(filename)
    Args: filename of dataset.names
    Returns a feature-map dictionary. Keys are feature names, Values are feature indexes in data
    
loadData(filename)
    Args: filename of dataset.data
    Returns 2-D list of data points

constructDecisionTree(data, featureMap, continuousFeatures)
    Args: 2D list, feature-map, list of non-categorical features (except outcome)
    Returns: dict, decision tree

predict(tree, record, featureMap)
    Args: decision tree, instance of data (1-D list), feature-map
    Returns: outcome

accuracy(tree, data, featureMap)
    Args: decision tree, 2D list of data, feature-map
    Returns: float, accuracy of tree in range [0,1]. 1 being highest

prune(tree, data, featureMap)
    Args: decision tree, 2D list of data, feature-map
    Returns: pruned decision tree


Classes
-------

Descriptions of class methods are exactly same as above.

Bagging(sampleSize, classifiersCount)
    > construct(data, featureMap, continuousFeatures)
    > predict(row, featureMap)
    > accuracy(data, featureMap)

Boosting(iterationsCount, sampleSize)
    > construct(data, featureMap, continuousFeatures)
    > predict(row, featureMap)
    > accuracy(data, featureMap)


Decision Tree Data Structure
============================

This is explained in the constructDecisionTree() docstring. Decision tree is a
dictionary and each node is recursively a dictionary. Following is the format

{
    // not available for the leaf node; leaf node represents outcome
    "feature": <feature name of the node>,
    
    // key: condition tuple (operator, value) applied on the feature
    // value: dictionary, sub-tree
    //
    // operator: <=, >, ==
    // value: integer/string/float
    //
    "children": {
        (operator, value) : {}
    },

    // only available at leaf node
    // represents `set` of outcome values.
    "outcome": [ set of outcome values ]
}
