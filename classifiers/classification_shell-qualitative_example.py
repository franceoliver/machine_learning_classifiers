# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:59:51 2019

@author: 939035

Classifiers
"""
# %% 1)Importing packages
import seaborn as sns
import pandas as pd
import numpy as np
# Handling SSL error when trying to connect from the office!
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Handing sns not showing plot error
import matplotlib.pyplot as plt
# ML models
# kernal SVM
from sklearn.svm import SVC
# RandomForrestModel
from sklearn.ensemble import RandomForestClassifier
# MLPClassifier (neural_network)
from sklearn.neural_network import MLPClassifier
# Gradient Boosting Tree
from sklearn.ensemble import GradientBoostingClassifier
# Training the model (speed)
# Decision Tree Classificr
from sklearn.tree import DecisionTreeClassifier
# Logisitc Regression
from sklearn.linear_model import LogisticRegression
# Data Is too Large
##Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# other random ones
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class machine_learning_classifier():
    ''' A class that contains a classifier loop '''

    def __init__(self):
        # Variables to alter
        self.df = sns.load_dataset('titanic')
        # List the qualitative variables you wish to create dummies from
        self.qualitative_vars = ['sex', 'who', 'deck', 'embark_town', 'alive']
        # Give the string of the y variable
        self.y_var = 'survived'
        # Do not alter
        self.df_feat = pd.DataFrame()
        self.dummies = pd.DataFrame


    def inital_variable_removal(self, inital_vars_to_drop):
        # Dropping duplicate variable e.g qualitative variable Class and quantitative equivalent pclass
        self.df = self.df.drop(inital_vars_to_drop, axis=1)
        return self.df

    def remove_na(self):
        # Dropping nan or na rows
        self.df = self.df.dropna().reset_index().drop('index', axis=1)
        return self.df


    def exploring_data(self, y_var_category, var1, var2):
        # ## Basic Pairplot
        pp = sns.pairplot(self.df, hue=self.y_var)
        plt.show()

        # creating kde plot of sepal_lenght vs sepal width for setosa species of flower
        kde = self.df[self.df[self.y_var] == y_var_category]
        kdeplot = sns.kdeplot(kde[var1], kde[var2], cmap='plasma', shade='true'
                    , shade_lowest=False)
        plt.show()
        return pp, kdeplot

    def creating_dummies(self):
        # 4)Creating Dummys from qualitative variables (optional)
        self.dummies = pd.get_dummies(self.df[self.qualitative_vars])
        ### dropping qualitative variables before standardising
        self.df = self.df.drop(self.qualitative_vars, axis=1)
        return self.df

    def standardising(self):
        # Splitting the DataFrame into the dummies and then the standard varibales
        from sklearn.preprocessing import StandardScaler
        # standardising the data to the same scale
        # why - larger scale data will have a greater effect on the results
        scaler = StandardScaler()
        # fitting the data minus the dependent variable
        scaler.fit(self.df.drop(self.y_var, axis=1))
        # creating the variable scaled featers (returns a array)
        scaled_features = scaler.transform(self.df.drop(self.y_var, axis=1))
        # Creating a df of the array'd scaled features
        self.df_feat = pd.DataFrame(scaled_features, columns=self.df.drop(self.y_var, axis=1).columns)

        return self.df_feat

    def readding_dummies(self):
        # %% 6) Re adding dummies after standardising
        ## adding dummies back on after standaridiation of the rest of the data
        self.df_feat = pd.concat([self.df_feat, self.dummies], axis=1)
        return self.df_feat

    def correlations(self):
        # %% 7) Find correlation among variables.
        # after standardising
        correlation_matrix = self.df_feat.corr()

        return correlation_matrix

    def dropping_highly_correlated_variables(self, list_of_vars_to_drop):
        self.df_feat = self.df_feat.drop(list_of_vars_to_drop, axis=1)
        return self.df_feat

    def setting_y(self):
        # Setting X and y
        self.y = self.df[self.y_var]
        return self.y


    def feature_selection(self):
        # https://scikit-learn.org/stable/modules/feature_selection.html
        import sklearn.feature_selection

    def model_loop(self):

        # model selection by cross validation.
        from sklearn.model_selection import cross_val_score

        models = [SVC(),
                  RandomForestClassifier(),
                  MLPClassifier(),
                  GradientBoostingClassifier(),
                  DecisionTreeClassifier(),
                  LogisticRegression(),
                  GaussianNB(),
                  KNeighborsClassifier()]

        classification_results = pd.DataFrame(columns=['model',
                                                       'corss_val_scores',
                                                       'cvs_mean'
                                                       ])

        for m in models:
            model = m
            cvs = cross_val_score(model, self.df_feat, self.y, cv=10, scoring='accuracy')
            cvsm = cvs.mean()

            classification_results = classification_results.append({'model': m,
                                                                    'corss_val_scores': cvs,
                                                                    'cvs_mean': cvsm,
                                                                    }
                                                                   , ignore_index=True)

        return classification_results

    def model_tuning(self):
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001]}

        grid = GridSearchCV(SVC(), param_grid, verbose=2)
        grid.fit(self.df_feat, self.y)

        grid_predictions = grid.predict(self.df_feat)

        cm = (confusion_matrix(self.y, grid_predictions))

        cr = (classification_report(self.y, grid_predictions))

        return cm, cr


def main():
    mlc = machine_learning_classifier()
    mlc.inital_variable_removal(inital_vars_to_drop = ['class', 'embarked'])
    mlc.remove_na()
    mlc.exploring_data(y_var_category = 1, var1 = 'pclass', var2 = 'fare')
    mlc.creating_dummies()
    mlc.standardising()
    mlc.readding_dummies()
    correlation_matrix = mlc.correlations()
    mlc.dropping_highly_correlated_variables(list_of_vars_to_drop=['who_man'])
    mlc.setting_y()
    classification_results = mlc.model_loop()
    confusion_matrix, classification_report = mlc.model_tuning()