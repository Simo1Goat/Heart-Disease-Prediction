from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def N_Bayes(dataPath: str):
    # instanciate the Gaussian Naive Bayes
    heart_data = pd.read_csv(dataPath)
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    nb_model = GaussianNB()
    nb_model.fit(X_train, Y_train)
    nb_predict = nb_model.predict(X_test)
    nb_accuracy = round(accuracy_score(nb_predict, Y_test) * 100, 2)
    print(f'the percentage of accuracy using Naive Bayes is {str(nb_accuracy)} %')
    return nb_accuracy


if __name__ == '__main__':
    N_Bayes("data/heart_disease_data.csv")

