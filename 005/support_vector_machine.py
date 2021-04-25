import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Creates the sample data we will be using
def create_samples(samples):
    # Generate two lists of random people. (NOTE: Samples will be multiplied by 2)
    green_people = np.random.randint(0, 100, size=(samples, 2)).tolist()
    red_people = np.random.randint(50, 150, size=(samples, 2)).tolist()

    # Tells the programme that: 0 = Green Person, 1 = Red Person
    colour = np.concatenate((np.zeros(samples), np.ones(samples))).flatten().tolist()

    return {'green_people': green_people, 'red_people': red_people, 'colour': colour}


def determine_colour(a, b):
    # Prepare data
    data = create_samples(100)
    green_people, red_people = data['green_people'], data['red_people']
    people = green_people + red_people
    colour = data['colour']

    # Input and outputs
    X = np.array(people)
    y = np.array(colour)

    # Split the data into testing and training data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2)

    # Creates a Support Vector Machine & a KNN Classifier
    clf = SVC(kernel='linear', C=5)
    knn = KNeighborsClassifier(n_neighbors=15)

    # Fits the data into each of them
    clf.fit(train_X, train_y)
    knn.fit(train_X, train_y)

    # Print the score of the model
    print('Score (SVC): ', clf.score(test_X, test_y))
    print('Score (kNN): ', knn.score(test_X, test_y))

    # Predict with out input values
    prediction_clf = clf.predict([[a, b]])
    prediction_knn = knn.predict([[a, b]])

    # Convert to Int
    clf_value, knn_value = int(prediction_clf[0]), int(prediction_knn[0])

    # Prints the prediction for both algorithms
    print(f'SVC ({clf_value}): ', 'Red' if clf_value == 1 else 'Green')
    print(f'kNN ({knn_value}): ', 'Red' if knn_value == 1 else 'Green')

    # Colour the plots and scatter them
    red_scatter = [np.array(red_people)[:, 0], np.array(red_people)[:, 1]]
    green_scatter = [np.array(green_people)[:, 0], np.array(green_people)[:, 1]]

    plt.scatter(red_scatter[0], red_scatter[1], c='r')
    plt.scatter(green_scatter[0], green_scatter[1], c='g')

    # Get the support vectors
    support_vectors = clf.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='b', marker='o')

    # Plot our a & b
    plt.scatter(a, b, color='black', s=200, marker="X")
    plt.show()


determine_colour(80, 80)  # (80,80) is close to the line so results will be different for kNN and SVC
determine_colour(100, 100)  # 100% will be red people
determine_colour(30, 30)  # 100% will be green people
