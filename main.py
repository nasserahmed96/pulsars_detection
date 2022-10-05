import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
from sklearn.decomposition import PCA


def read_data_from_csv(file_name):
    #Check for the dataset
    print('Current dir: ', os.getcwd())
    print('Dataset file name: ', os.listdir('../Dataset'))
    return pd.read_csv(f'../Dataset/{file_name}')


def get_data_with_columns(data: pd.DataFrame, required_columns: list):
    """
    Get data for specific columns
    :param required_columns: List of required columns to be selected in order
    :param data: A Dataframe containing the required data
    :return:
    """
    data.columns = required_columns
    data.info()
    return data


def draw_heatmap(data: pd.DataFrame):
    corr = data.corr()
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, center=0, square=True, linewidth=.5)
    plt.show()


def draw_correlations(data):
    print('Correlations')
    print(data.corr())
    print('Mean corresponding to status col')
    print(data.groupby('target_class').mean())
    print('Median corresponding to status col')
    print(data.groupby('target_class').median())


def draw_box_plot(data):
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(0, len(data.columns)-1):
        sns.boxplot(x='target_class', y=data.iloc[:,i], data=data, orient='v', ax=axes[i])
    plt.tight_layout()
    plt.show()


def data_split(X, y):
    """
    Split the data into train and test
    :param data:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=0)
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
    scalar.fit(X_train)
    train_scaled = scalar.transform(X_train)
    test_scaled = scalar.transform(X_test)
    return train_scaled, test_scaled, y_train.astype(int), y_test.astype(int)


def logistic_regression(x, y):
    logreg = LogisticRegression().fit(x, y)
    return logreg


def training(X, y):
    X_train_scaled, X_test_scaled, y_train, y_test = data_split(X, y)
    logreg_result = logistic_regression(X_train_scaled, y_train)
    print("Training set score: {:0.3f}".format(logreg_result.score(X_train_scaled, y_train)))
    print("Test set score: {:0.3f}".format(logreg_result.score(X_test_scaled, y_test)))
    logit_model = sm.Logit(y_train, X_train_scaled)
    result = logit_model.fit()
    print(result.summary2())
    return logreg_result, X_test_scaled, y_test


def predict(logreg_result, X_test_scaled):
    y_pred = logreg_result.predict(X_test_scaled)
    y_pred_string = y_pred.astype(str)
    y_pred_string[np.where(y_pred_string == '0')] = '0'
    y_pred_string[np.where(y_pred_string == '1')] = '1'
    return y_pred_string


def test(y_test):
    y_test_string = y_test.astype(str)
    y_test_string[np.where(y_test_string == '0')] = '0'
    y_test_string[np.where(y_test_string == '1')] = '1'
    return y_test_string

def draw_confusion_matrix(y_test_string, y_pred_string):
    ax = plt.subplot()
    labels = ['Pulsar', 'Not pulsar']
    cm = confusion_matrix(y_test_string, y_pred_string)
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Pulsar', 'Not pulsar'])
    ax.yaxis.set_ticklabels(['Pulsar', 'Not pulsar'])
    plt.show()


def run():
    data = pd.concat([read_data_from_csv('pulsar_data_train.csv'), read_data_from_csv('pulsar_data_test.csv')])
    data = data.dropna()
    draw_heatmap(data)
    draw_correlations(data)
    draw_box_plot(data)
    X = data.loc[:, data.columns != 'target_class']
    pca = PCA(n_components=len(X.columns))
    X = pca.fit_transform(X)
    y = data.loc[:, data.columns == 'target_class']
    logreg_result, X_test_scaled, y_test = training(X, y)
    y_pred_string = predict(logreg_result, X_test_scaled)
    y_test_string = test(y_test)
    print('Accuracy: ', accuracy_score(y_pred_string, y_test_string))
    draw_confusion_matrix(y_test_string, y_pred_string)



if __name__ == '__main__':
    run()