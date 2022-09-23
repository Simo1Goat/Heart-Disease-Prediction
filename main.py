# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    heart_data = pd.read_csv("data/heart_disease_data.csv")
    # checking the missing values: -- no missing values in or data --
    print(heart_data.isnull().sum())
    # statistical information about data
    print(heart_data.describe())
    # see how many people have heart disease: #1: 165 affected #0: 138 not affected.
    # print(heart_data["target"].value_counts())
    # split our data into test and train
    X = heart_data.drop(columns="target", axis=1)
    Y = heart_data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
    # shape of new split data
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    # start working with logistic regression
    model = LogisticRegression(solver="lbfgs", max_iter=100)
    model.fit(X_train, Y_train)
    # accuracy score on the training data
    training_data_accuracy = model.predict(X_train)
    train_accuracy = accuracy_score(training_data_accuracy, Y_train)
    # accuracy score on the testing data
    testing_data_accuracy = model.predict(X_test)
    test_accuracy = accuracy_score(testing_data_accuracy, Y_test)
    print(f'train_accuracy {train_accuracy}, testing accuracy {test_accuracy}')
    # confusion matrix to check the accuracy of the predicted test:
    cm_test = confusion_matrix(testing_data_accuracy, Y_test)
    sns.heatmap(cm_test, cmap="Greens", annot=True,
                cbar_kws={"orientation": "vertical", "label": "degree of accuracy"},
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
