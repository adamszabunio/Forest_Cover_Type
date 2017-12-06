#!/usr/bin/env python3
'''
Decision Tree and Random Forest Classifier algorithims coded for 
Machine Learning Class (second semester of Master's program)
'''

from collections import Counter
import numpy as np

class TreeNode(object):
    '''
    A node class for a decision tree.
    '''
    def __init__(self):
        self.column = None  # (int) index of feature to split on
        self.value = None  # value of the feature to split on
        self.categorical = True  # (bool) whether or not node is split on
                                 # categorial feature
        self.name = None    # (string) 
        self.left = None    # (TreeNode) left child
        self.right = None   # (TreeNode) right child
        self.leaf = False   # (bool) true if node is a leaf, false otherwise
        self.classes = Counter()  

    def predict_one(self, x):
        '''
        INPUT:
            - x: 1d numpy array (single data point)
        OUTPUT:
            - y: predicted label
        Return the predicted label for a single data point.
        '''
        if self.leaf:
            return self.name
        col_value = x[self.column]

        if self.categorical:
            if col_value == self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)
        else:
            if col_value < self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)


class DecisionTree(object):
    '''
    A decision tree class.
    '''

    def __init__(self, impurity_criterion='entropy'):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None  # root Node
        self.feature_names = None  # string names of features 
        self.categorical = None # Boolean array
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini

    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None
        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        is_categorical = lambda x: isinstance(x, str) or isinstance(x, bool)
        # TODO: add updates to make robust toa variety of datatypes (np.array...)
            
        # Each variable (organized by index) is given a label categorical or not
        self.categorical = np.vectorize(is_categorical)(X[0])

        # Call the build_tree function
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - TreeNode
        Recursively build the decision tree. Return the root node.
        '''

        # initialize a root TreeNode
        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        # if no index is returned from the split index or we cannot split
        if index is None or len(np.unique(y)) == 1:
            # set the node to be a leaf
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]

        else: # otherwise we can split 
            X1, y1, X2, y2 = splits
            # the node column should be set to the index coming from split_index
            node.column = index
            # the node name is the feature name as determined by the index (column name)
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]

            # continue recursing down both branches of the split
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)

        return node

    def _entropy(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float
        Return the entropy of the array y.
        '''

        total = 0

        for c in np.unique(y):
            p_C = np.sum(y == c) / float(len(y))
            total += p_C * np.log(p_C)
            
        return -total

    def _gini(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float
        Return the gini impurity of the array y.
        '''

        total = 0

        for c in np.unique(y):
            p_C = np.sum(y == c) / float(len(y))
            total += p_C**2
            
        return 1 - total

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)
        Return the two subsets of the dataset achieved by the given feature and
        value to split on.
        Call the method like this:
        X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)
        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''

        split_column = X[:, split_index]

        if self.categorical[split_index]:
            # select the indices of the rows in the column
            # with the split_value (T/F) into one set of indices 
            A = split_column == split_value
            B = split_column != split_value
        # else if the variable is not categorical
        else:
            A = split_column < split_value
            B = split_column >= split_value
            
        return X[A], y[A], X[B], y[B]

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float
        Return the information gain of making the given split.
        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''
        # * set total equal to the impurity_criterion
        total = self.impurity_criterion(y)
        
        avg_ent = len(y1)/len(y)*self.impurity_criterion(y1) \
             + len(y2)/len(y)*self.impurity_criterion(y2)
        total -= avg_ent

        return total

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)
        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.
        Return None, None, None if there is no split which improves information
        gain.
        Call the method like this:
        index, value, splits = self._choose_split_index(X, y)
        X1, y1, X2, y2 = splits
        '''

        split_index, split_value, split = None, None, None
        # keep track of the maximum gain
        max_gain = 0

        for col in range(X.shape[1]):
            values = np.unique(X[:, col])
            if len(values) < 2:
                continue

            for val in values:
                # make a temporary split (using the column index and V)
                temporary_split = self._make_split(X, y, col, val)
                # calculate the information gain between the original y, y1 and y2
                X1, y1, X2, y2 = temporary_split
                gain = self._information_gain(y, y1, y2)
                if gain > max_gain:

                    # set max_gain, split_index, and split_value to be equal
                    # to the current max_gain, column and value
                    # set the output splits to the current split setup (X1, y1, X2, y2)
                    split = temporary_split
                    max_gain, split_index, split_value = gain, col, val
                   
        return split_index, split_value, split 
    
    
    def predict(self, X):
        '''
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array
        Return an array of predictions for the feature matrix X.
        '''

        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree. This will allow you to $:print tree
        '''
        return str(self.root)
    
class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features, impurity_criterion='entropy'):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None
        self.impurity_criterion = impurity_criterion

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):

        # Return a list of num_trees DecisionTrees.
        row, col = X.shape
        forest = []
        
        for tree in range(num_trees):
            # create a random set of X_samples with replacement
            X_samp = np.random.randint(row, size=num_samples)
            # create a random permutation of features (list)[sliced to length num_features]
            y_samp = np.random.permutation(col)[:num_features]
            X_tree = X[X_samp,:][:,y_samp]
            y_tree = y[X_samp]
            tree = DecisionTree(self.impurity_criterion)
            tree.fit(X_tree, y_tree, feature_names=y_samp)
            forest.append(tree)
            
        return forest

    def predict(self, X):

        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
       
        # * Each one of the trees is allowed to predict on the same row of input data. The majority vote
        # is the output of the whole forest. This becomes a single prediction.
        
        predictions  = []
        for tree in self.forest:
            predict = tree.predict(X[:,tree.feature_names])
            predictions.append(predict)
        # HT to Tristan! Thank you for the tip on np.apply_along_axis
        # Count along the columns and find the most common class    
        return np.array(np.apply_along_axis(lambda col: Counter(col).most_common()[0][0], \
                                            arr=predictions, axis=0))

    
    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data.
        '''
        
        return sum(self.predict(X) == y) / len(y)