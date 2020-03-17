#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:10:43 2019

@author: ofrance002

Keras Exercise
Predict political party based on votes
As a fun little example, we'll use a public data set of how US congressmen voted on 17 different issues in the year 1984. Let's see if we can figure out their political party based on their votes alone, using a deep neural network!

For those outside the United States, our two main political parties are "Democrat" and "Republican." In modern times they represent progressive and conservative ideologies, respectively.

Politics in 1984 weren't quite as polarized as they are today, but you should still be able to get over 90% accuracy without much trouble.

Since the point of this exercise is implementing neural networks in Keras, I'll help you to load and prepare the data.

Let's start by importing the raw CSV file using Pandas, and make a DataFrame out of it with nice column labels:
"""

#Data collection and cleaning

import pandas as pd

feature_names =  ['party','handicapped-infants', 'water-project-cost-sharing', 
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                    'el-salvador-aid', 'religious-groups-in-schools',
                    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                    'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                    'education-spending', 'superfund-right-to-sue', 'crime',
                    'duty-free-exports', 'export-administration-act-south-africa']

voting_data = pd.read_csv('/Users/ofrance002/Documents/machine_learning/neural_networks/MLCourse/house-votes-84.data.txt', na_values=['?'], 
                          names = feature_names)

voting_data.head()

voting_data.describe()

'''
We can see there's some missing data to deal with here; some politicians abstained on some votes, or just weren't present when the vote was taken. We will just drop the rows with missing data to keep it simple, but in practice you'd want to first make sure that doing so didn't introduce any sort of bias into your analysis (if one party abstains more than another, that could be problematic for example.)
'''

voting_data.dropna(inplace=True)
voting_data.describe()

'''Our neural network needs normalized numbers, not strings, to work. So let's replace all the y's and n's with 1's and 0's, and represent the parties as 1's and 0's as well.'''

voting_data.replace(('y', 'n'), (1, 0), inplace=True)
voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)

voting_data.head()

'''
Finally let's extract the features and labels in the form that Keras will expect:
'''
all_features = voting_data[feature_names].drop('party', axis=1).values
all_classes = voting_data['party'].values

#%% Machine learning aspect


from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score

def create_model():
    model = Sequential()
    #16 feature inputs (votes) going into an 32-unit layer 
    model.add(Dense(32, input_dim=16, kernel_initializer='normal', activation='relu'))
    # Another hidden layer of 16 units
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (Democrat or Republican political party)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Wrap our Keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
cv_scores.mean()

'''
94% without even trying too hard! Did you do better? Maybe more neurons, more layers, or Dropout layers would help even more.
'''



