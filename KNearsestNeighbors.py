from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def KNNClassifier(dataPath: str):
    heart_data = pd.read_csv(dataPath)
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    # instanciate our model
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    # predict on test dataset
    predicted_model = knn.predict(X_test)
    print(predicted_model.shape)
    # accuracy score achieved
    knn_score = round(accuracy_score(predicted_model, Y_test) * 100, 2)
    print(f'the percentage of accuracy achieved using knn is {str(knn_score)} %')
    return knn_score


if __name__ == '__main__':
    KNNClassifier("data/heart_disease_data.csv")

