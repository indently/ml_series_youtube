import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier


def wine_decision(wine_type, score, price):
    df = pd.read_csv('sample_wine_data.csv')

    # We convert the sample data into 1s and 0s
    le = LabelEncoder()
    en_wine, en_score, en_price = le.fit_transform(df.wine), le.fit_transform(df.score), le.fit_transform(df.price)
    en_bought = le.fit_transform(df.bought)

    # Translate the data into something we can later remember
    transl = {
        'Red': 0, 'White': 1,
        'High': 0, 'Low': 1,
        'Yes': 0, 'No': 1
    }

    # We combine the features into a single list
    features = list(zip(en_wine, en_score, en_price))

    # We follow the naming convention for our input and outputs
    X = np.array(features)
    y = np.array(en_bought)

    # Split the data into testing and training data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2)

    # We create a Decision Tree Classifier & Random Forest Classifier
    clf = tree.DecisionTreeClassifier()
    rf_clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    rf_clf.fit(X, y)

    # Get the score
    print('Score (TREE): ', clf.score(X, y))
    print('Score (FOREST): ', rf_clf.score(X, y))

    # We make and format the prediction
    prediction = clf.predict([[transl[wine_type], transl[score], transl[price]]])
    value = int(prediction[0])
    print(value)
    print('I buy the wine.' if value == 1 else 'No wine for me today.')


wine_decision(wine_type='Red', score='High', price='High')
