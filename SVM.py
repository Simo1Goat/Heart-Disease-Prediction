from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


def SupportVectorMachine(dataPath: str):
    heart_data = pd.read_csv(dataPath)
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, Y_train)
    predicted_model = svm_model.predict(X_test)
    print(predicted_model.shape)
    score_svm = round(accuracy_score(predicted_model, Y_test) * 100, 2)
    print(f'the percentage achieved using SVM {str(score_svm)} %')
    return score_svm


if __name__ == '__main__':
    SupportVectorMachine("data/heart_disease_data.csv")
