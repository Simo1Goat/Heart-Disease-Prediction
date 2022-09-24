import main, NaiveBayes, SVM, KNearsestNeighbors, DecisionTree, Forest_Random
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataPath = "data/heart_disease_data.csv"
    lr_score = main.logisticsRegression(dataPath)
    nb_score = NaiveBayes.N_Bayes(dataPath)
    svm_score = SVM.SupportVectorMachine(dataPath)
    knn_score = KNearsestNeighbors.KNNClassifier(dataPath)
    dt_score = DecisionTree.Decision_Tree_Classifier(dataPath)
    fr_score = Forest_Random.Random_Forest_Classifier(dataPath)
    algorithms = ["logistic regression", "Naive Bayes", "SVM", "KNN", "Decision Tree", "Forest Random"]
    algorithms_score = [lr_score, nb_score, svm_score, knn_score, dt_score, fr_score]

    for i in range(len(algorithms_score)):
        print(f'the accuracy score achieved using {algorithms[i]} is {algorithms_score[i]}%')
    sns.set(rc={'figure.figsize': (15, 8)})
    plt.xlabel("Algorithms")
    plt.ylabel("Algorithms Score")
    sns.barplot(algorithms, algorithms_score)
    plt.show()
