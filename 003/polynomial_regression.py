import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def predict_sales(time):
    raw_data = {
        'years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'sales': [10, 30, 50, 60, 50, 54, 67, 68, 80, 100]
    }

    df = pd.DataFrame(raw_data)

    X = np.array(df['years'].tolist())
    y = np.array(df['sales'].tolist())

    model = np.poly1d(np.polyfit(X, y, deg=3))

    y_prediction = model(time)
    y_prediction_test = model(X)

    print('Prediction: ', y_prediction)

    print('MAE: ', mean_absolute_error(y, y_prediction_test))
    print('r2: ', r2_score(y, y_prediction_test))

    curvy_line = np.linspace(1, 10, 100)

    plt.scatter(X, y)
    plt.plot(curvy_line, model(curvy_line))
    plt.show()


predict_sales(15.5)
