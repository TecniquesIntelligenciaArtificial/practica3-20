import csv
import numpy as np

from sklearn import metrics

# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

# reading wine data
with open('wine.csv', 'r') as f:
  reader = csv.reader(f, delimiter=',')
  wine = list(reader)

# convert data to numpy array and separate data from class. Also convert data to float
wineA = np.array(wine)
wine_data = wineA[1:,1:13].astype(np.float64)
wine_class = wineA[1:,0]


# Create a preprocessor to standardize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Create a preprocessor to use Linear Discriminant Analysis:
# It reduces the number of atributes to 2 = number of classes -1
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)

# k nearest neighbor as classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Gaussian Naive Bayes
# Gaussian is used to deal with continuous attributes that (we assume) follow a normal distribution
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# Decision tree classifier
from sklearn import tree
dtc = tree.DecisionTreeClassifier()

# Classifies data with the given classifier. Performs a k-folder cross validation. If specified can preprocess (transform) data
(None, 'normalize' and 'linearDiscriminatAnalysis')
def k_folder_cross_classification(classifier, data, target_class, transform = None, folds=10, normalize=sc, lda = lda):
  # prepare cross validation
  kfold = KFold(folds, True, 1)
  scores = [None] * folds
  # enumerate splits
  for idx, (train, test) in enumerate(kfold.split(data)):
    if transform == None:
      dataTrain = data[train]
      dataTest = data[test]
    elif transform == 'normalize':
      dataTrain = normalize.fit_transform(data[train], target_class[train])
      dataTest = normalize.transform(data[test])
    elif transform == 'linearDiscriminatAnalysis':
      dataNormTrain = normalize.fit_transform(data[train], target_class[train])
      dataNormTest = normalize.transform(data[test])
      dataTrain = lda.fit_transform(dataNormTrain, target_class[train])
      dataTest = lda.transform(dataNormTest)
    classifier.fit(dataTrain, target_class[train])
    predicted = classifier.predict(dataTest)
    scores[idx] = metrics.classification_report(target_class[test], predicted, output_dict=True)

  return scores

def average_metrics(scores):
  score_keys = list(scores[0]['macro avg'].keys())
  average = {}
  for sk in score_keys:
    average[sk]=0.0

  for i in range(0, len(scores)):
    for score_key, score_value in scores[i]['macro avg'].items():
      average[score_key] = average[score_key] + score_value

  for score_key, score_value in average.items():
    average[score_key] = score_value/len(scores)

  return average

# use Decision Tree classifier
result = k_folder_cross_classification(dtc, wine_data, wine_class)
print('DTC. raw data: ', average_metrics(result))

result = k_folder_cross_classification(dtc, wine_data, wine_class, transform='normalize')
print('DTC. normalized data: ', average_metrics(result))

result = k_folder_cross_classification(dtc, wine_data, wine_class, transform='linearDiscriminatAnalysis')
print('DTC. linear discriminant analysis data: ', average_metrics(result))
print()
