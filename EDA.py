import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def EDA(dataPath: str):
    heart_data = pd.read_csv(dataPath)
    # printing few rows from dataset
    print(heart_data.sample())
    # describing our data
    print(heart_data.describe())
    # printing columns type
    print(heart_data.info())
    # checking missing values
    print(heart_data.isnull().sum())
    # luckily we don't have missing values
    # checking correlation between columns
    corr_matrix = heart_data.corr()
    sns.heatmap(corr_matrix, cmap="Reds", annot=True, cbar_kws={"orientation": "vertical", "label": "correlation degree"})
    plt.show()
    # Analyzing the target column
    sns.countplot(heart_data.target)
    plt.show()
    print(f'Percentage of patience without heart problems: {str(round(heart_data.target.value_counts()[0] * 100 / heart_data.shape[0], 2))}%')
    print(f'Percentage of patience with heart problems: {str(round(heart_data.target.value_counts()[1] * 100 / heart_data.shape[0], 2))}%')
    # Analyzing the 'Chest Pain Type' feature
    print(heart_data.cp.unique())
    sns.barplot(heart_data.cp, heart_data.target)
    plt.show()
    # we notice that the people with chest pain 0 are much less likely to have heart problems
    # Analyzing the restcg feature
    print(heart_data.restecg.unique())
    sns.barplot(heart_data.restecg, heart_data.target)
    plt.show()
    # as we can see from the bar plot people with 0, 1 are more likely to have heart problem
    # Analyzing the exang feature
    print(heart_data.exang.unique())
    sns.barplot(heart_data.exang, heart_data.target)
    plt.show()
    # people with exang = 1 are much less likely to have heart problems
    # Analysing the 'ca' feature
    print(heart_data["ca"].unique())
    sns.countplot(heart_data.ca)
    plt.show()
    sns.barplot(heart_data.ca, heart_data.target)
    plt.show()
    # ca = 4 has astonishgly large number of heart patients
    # Analyzing the "thal" feature
    sns.barplot(heart_data["thal"], heart_data["target"])
    plt.show()


if __name__ == '__main__':
    EDA("data/heart_disease_data.csv")
