import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data\data.csv')

# Splitting
training = data.head(5000)
testing = data.tail(5000)
holdout_set = training.sample(5000, random_state=1) # pick 5000 observations randomly
training = training.drop(holdout_set.index)         # remove holdout from training data

# Build a classifier
classifier = DummyClassifier(strategy='most_frequent')
classifier.fit(training.drop('Genre', axis=1), training['Genre'])

# Estimate accuracy
pred = classifier(holdout_set.drop('Genre', axis=1))
estimated_accuracy = accuracy_score(holdout_set['Genre'], pred)
pd.Series(estimated_accuracy).to_csv('ea(1).csv', index=False, header=False)

# Predicting testing set
pred = classifier.predict(holdout_set.drop('Genre', axis=1))
pred = pd.Series(pred).to_csv('pred(1).csv', index=False, header=False)