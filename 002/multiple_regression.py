import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def predict_distance(fuel_consumption_l, percent_electric):
    raw_data = {
        'fuel_consumption_l': [1, 2, 2, 3, 4, 5, 6],
        'percent_electric': [.20, .20, .60, .90, .90, .50, .90],
        'total_distance_km': [200, 240, 560, 990, 1200, 800, 1400]
    }

    df = pd.DataFrame(raw_data)

    X = np.array(df[['fuel_consumption_l', 'percent_electric']])
    y = np.array(df['total_distance_km']).reshape(-1, 1)

    # Split into test and training data
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=.20)

    # Initialise the model
    model = LinearRegression()
    model.fit(train_X, train_y)

    # Make a prediction
    y_prediction = model.predict([[fuel_consumption_l, percent_electric]])
    print('Prediction: ', y_prediction)

    # Used to check how accurate the model is
    y_test_prediction = model.predict(test_X)

    print('MAE: ', mean_absolute_error(test_y, y_test_prediction))
    print('r2: ', r2_score(test_y, y_test_prediction))


predict_distance(5, .80)
