import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create some sample data
raw_data = {
    'weather': ['Sunny', 'Sunny', 'Sunny', 'Rainy', 'Rainy', 'Rainy', 'Clear', 'Clear', 'Clear', 'Clear'],
    'temp': ['Hot', 'Cold', 'Hot', 'Hot', 'Cold', 'Cold', 'Hot', 'Hot', 'Cold', 'Cold'],
    'go_outside': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No']
}


# Create the function
def predict_outside(now_weather, now_temp):
    # Map the values to numbers
    map_no = {'Sunny': 2, 'Rainy': 1, 'Clear': 0,
              'Hot': 1, 'Cold': 0}

    df = pd.DataFrame(raw_data)
    print(df)

    # Encode the parameters
    le = LabelEncoder()
    en_weather, en_temp = le.fit_transform(df.weather), le.fit_transform(df.temp)
    en_go = le.fit_transform(df.go_outside)

    # Combine the inputs
    features = tuple(zip(en_weather, en_temp))

    # Inputs and outputs
    X = np.array(features)
    y = np.array(en_go)

    # Create the model
    model = GaussianNB()

    # Split the data into testing and training data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2)

    # Fit the model
    model.fit(X, y)

    # Get the score
    print('Score: ', model.score(X, y))

    # Make the prediction
    prediction = model.predict([[map_no[now_weather], map_no[now_temp]]])
    value = int(prediction[0])
    print(f'It is {now_temp} and {now_weather}. Go outside?')
    print('Prediction: ', 'Yes' if value == 1 else 'No')


predict_outside('Rainy', 'Cold')
