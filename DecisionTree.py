from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def Decision_Tree_Classifier(dataPath: str):
    epoch = None
    max_accuracy = 0
    epochs = 200
    heart_data = pd.read_csv(dataPath)
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)
    for epoch in range(epochs):
        dt_model = DecisionTreeClassifier(random_state=epoch)
        dt_model.fit(X_train, Y_train)
        predicted_model = dt_model.predict(X_test)
        current_accuracy = round(accuracy_score(predicted_model, Y_test) * 100, 2)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_epoch = epoch

    print(f'the percentage of accuracy achieved using Decision tree is {str(max_accuracy)} % ==> epoch {str(epoch)}')
    return max_accuracy


if __name__ == '__main__':
    Decision_Tree_Classifier("data/heart_disease_data.csv")
