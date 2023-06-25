#!/usr/bin/env python3

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import settings

df = pd.read_csv(settings.DATASET_PATH)

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# changing from str to numbers
X['gender'] = X['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
X['smoking_history'] = X['smoking_history'].map({'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=5)

tree.fit(X=X_train, y=y_train)

y_predicted = tree.predict(X_test)

accuracy_tree = accuracy_score(y_test, y_predicted)

accuracy_tree_percentage = round(accuracy_tree * 100, 2)

print(f'Accuracy Precision: {accuracy_tree_percentage}%')

joblib.dump(tree, settings.WEIGHTS_PATH)
