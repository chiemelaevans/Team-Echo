import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import pickle

# Load the csv file
iris_cleaned = pd.read_csv("iris_dataset_cleaned.csv")

#Split into features and labels
# Features (X) and Labels (y)
X = iris_cleaned.drop(columns='class')
y = iris_cleaned['class']

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert to pandas DataFrames (this step is redundant as train_test_split already returns pandas DataFrames)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.DataFrame(y_train, columns=['class'])
y_test = pd.DataFrame(y_test, columns=['class'])

from sklearn.ensemble import RandomForestClassifier

# Create an instance of RandomForestClassifier with default parameters
forest = RandomForestClassifier(random_state=42)

# Train the model
forest.fit(X_train, y_train)

# Make predictions on the test data
y_pred = forest.predict(X_test)


# Make pickle file of our model
pickle.dump(forest, open("model.pkl", "wb"))
