from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import numpy as np
import math
import pandas as pd


class Node:
    """A decision tree node."""
    def __init__(self, gini, entropy, num_samples, 
            num_samples_per_class, predicted_class):
        self.gini = gini
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self,criterion='gini', max_depth=4):
        self.criterion = criterion
        self.max_depth = max_depth

    def _gini(self,sample_y,n_classes):
        # TODO: calculate the gini index of sample_y
        # sample_y represent the label of node
        final_gini, each_gini = 0, 0
        num_samples_per_class = [np.sum(sample_y == i) for i in range(self.n_classes_)]  # num_samples_per_class 輸出 y是0 的sample有幾個, y是1 的sample有幾個, y是2 的sample有幾個
        # print("num_samples_per_class")
        # print(num_samples_per_class)
        total_class = np.sum(num_samples_per_class)

        if (total_class == 0) :
          return final_gini
        for idx in range(len(num_samples_per_class)-1):
          each_gini += -((num_samples_per_class[idx]/total_class)**2)
        final_gini = 1 - each_gini
        return final_gini
         
    def _entropy(self,sample_y,n_classes):
        # TODO: calculate the entropy of sample_y 
        # sample_y represent the label of node
        entropy = 0
        if len(sample_y)!=0:
          num_samples_per_class = [np.sum(sample_y == i) for i in range(self.n_classes_)]
          total_class = 0
          for i in range(len(num_samples_per_class)):
            total_class += num_samples_per_class[i]
          for i in range(n_classes):
            if (num_samples_per_class[i]==0):
              each_entropy = 0
            else:
              each_entropy = -(num_samples_per_class[i]/total_class)*np.log2(num_samples_per_class[i]/total_class)
            entropy += each_entropy
        else:
          entropy = 0
        return entropy

    def _feature_split(self, X, y,n_classes):
        InfoGain = 0
        # Returns:
        #  best_idx: Index of the feature for best split, or None if no split is found.
        #  best_thr: Threshold to use for the split, or None if no split is found.
        m = y.size
        best_idx, best_thr = 0, 0
        if m <= 1:
            return None, None

        # Gini or Entropy of current node.
        if self.criterion == "gini":
            best_criterion = self._gini(y,n_classes)
            gini_Max = 0
            for feature_idx in range(X.shape[1]):
              sort_X_index = np.argsort(X[:,feature_idx])
              sort_X = X[sort_X_index]
              sort_y = y[sort_X_index]
              for sample_num in range(X.shape[0]-1):
                if (sort_X[sample_num][feature_idx]) == (sort_X[sample_num+1][feature_idx]):
                  continue
                else:
                  split_value = (sort_X[sample_num,feature_idx]+sort_X[sample_num+1,feature_idx])/2
                  sort_y_left = sort_y[:sample_num]
                  sort_y_right = sort_y[sample_num:]
                  left_gini = self._gini(sort_y_left, n_classes)
                  right_gini = self._gini(sort_y_right, n_classes)                            #print(X[ X[:,0]>0, 0 ])
                  final_gini = (len(sort_y_left)/(len(sort_y_left)+len(sort_y_right)))*left_gini + (len(sort_y_right)/(len(sort_y_left)+len(sort_y_right)))*right_gini
                if (final_gini > gini_Max):
                  gini_Max = final_gini
                  best_idx = feature_idx
                  best_thr = split_value
            return best_idx, best_thr
        else:
            InfoGain = 0
            best_criterion = self._entropy(y,n_classes)
            for feature_idx in range(X.shape[1]):
              sort_X_index = np.argsort(X[:,feature_idx])
              sort_X = X[sort_X_index]
              sort_y = y[sort_X_index]
              for sample_num in range(X.shape[0]-1):
                if (sort_X[sample_num][feature_idx]) == (sort_X[sample_num+1][feature_idx]):
                  continue
                else:
                  split_value = (sort_X[sample_num,feature_idx]+sort_X[sample_num+1,feature_idx])/2
                  sort_y_left = sort_y[:sample_num]
                  sort_y_right = sort_y[sample_num:]
                  left_en = self._entropy(sort_y_left, n_classes)
                  right_en = self._entropy(sort_y_right, n_classes)
                  newInfoGain = best_criterion-(left_en*len(sort_y_left)/len(sort_y) + right_en*len(sort_y_right)/len(sort_y))
                  # print(newInfoGain)
                  # print(InfoGain)
                  # print("---------")
                  if (newInfoGain > InfoGain):
                    InfoGain = newInfoGain
                    best_idx = feature_idx
                    best_thr = split_value

        # TODO: find the best split, loop through all the features, and consider all the
        # midpoints between adjacent training samples as possible thresholds. 
        # Computethe Gini or Entropy impurity of the split generated by that particular feature/threshold
        # pair, and return the pair with smallest impurity.
              return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y,self.n_classes_),
            entropy = self._entropy(y,self.n_classes_),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
 
        if depth < self.max_depth:
            idx, thr = self._feature_split(X, y,self.n_classes_)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    def fit(self,X,Y):
        # Fits to the given training data
        self.n_classes_ = len(np.unique(Y)) 
        self.n_features_ = X.shape[1]
        
        # if user entered a value which was neither gini nor entropy
        if self.criterion != 'gini' :
            if self.criterion != 'entropy':
                self.criterion='gini'         
        self.tree_ = self._build_tree(X, Y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

def load_train_test_data(test_ratio=.3, random_state = 1):
    balance_scale = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
            names=['Class Name', 'Left-Weigh', 'Left-Distance', 'Right-Weigh','Right-Distance'],header=None)
    
    class_le = LabelEncoder()
    balance_scale['Class Name'] = class_le.fit_transform(balance_scale['Class Name'].values)
    X = balance_scale.iloc[:,1:].values
    y = balance_scale['Class Name'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std , X_test_std

def accuracy_report(X_train_scale, y_train,X_test_scale,y_test,criterion='gini',max_depth=4):
    tree = DecisionTreeClassifier(criterion = criterion, max_depth=max_depth)
    tree.fit(X_train_scale, y_train)
    pred = tree.predict(X_train_scale)
    print(criterion + " tree train accuracy: %f" 
        % (sklearn.metrics.accuracy_score(y_train, pred )))
    pred = tree.predict(X_test_scale)
    print(criterion + " tree test accuracy: %f" 
        % (sklearn.metrics.accuracy_score(y_test, pred )))
    
def main():
    X_train, X_test, y_train, y_test = load_train_test_data(test_ratio=.3,random_state=1)
    X_train_scale, X_test_scale = scale_features(X_train, X_test)
    #gini tree
    # accuracy_report(X_train_scale, y_train,X_test_scale,y_test,criterion='gini',max_depth=4)
    #entropy tree
    accuracy_report(X_train_scale, y_train,X_test_scale,y_test,criterion='entropy',max_depth=4) 

if __name__ == "__main__":
    main()
