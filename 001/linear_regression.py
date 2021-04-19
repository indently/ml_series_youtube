import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def predict_salary(years_of_experience):
    # Data
    raw_data = {
        'years_worked': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'salary': [60, 100, 130, 150, 180, 230, 260, 270, 290, 330]
    }

    df = pd.DataFrame(raw_data)

    # Get the data we want to predict
    X = np.array(df['years_worked']).reshape(-1, 1)
    y = np.array(df['salary']).reshape(-1, 1)

    # Splits the testing data and the training data
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=.20)

    # Initialise the model
    model = LinearRegression()
    model.fit(train_X, train_y)

    # Make the prediction
    y_prediction = model.predict([[years_of_experience]])
    print('PREDICTION: ', y_prediction)

    # Used to check how accurate the model is
    y_test_prediction = model.predict(test_X)
    y_line = model.predict(X)

    # Extra info
    print('Slope', model.coef_)
    print('Intercept', model.intercept_)
    print('MAE', mean_absolute_error(test_y, y_test_prediction))
    print('r2', r2_score(test_y, y_test_prediction))

    # Plot the data
    plt.scatter(X, y, s=12)
    plt.xlabel('Years (Exp)')
    plt.ylabel('Salary')
    plt.plot(X, y_line, color='r')
    plt.show()


predict_salary(20)
