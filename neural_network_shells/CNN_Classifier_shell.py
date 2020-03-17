## 1 Load libraries
import numpy as np
import pandas as pd
import seaborn as sns
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras import models




## 2 load data
df = sns.load_dataset('iris')

## 3 Explority data analysis
# Basic Pairplot
sns.pairplot(df, hue='species')
# creating kde plot of sepal_lenght vs sepal width for setosa species of flower
setosa = df[df['species']=='setosa']
sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma', shade='true'
            , shade_lowest =False)

## 5 standardising the data (optional)
# Splitting the DataFrame into the dummies and then the standard varibales
from sklearn.preprocessing import StandardScaler
# standardising the data to the same scale
# why - larger scale data will have a greater effect on the results
scaler = StandardScaler()
# fitting the data minus the dependent variable
scaler.fit(df.drop('species',axis=1))
# creating the variable scaled featers (returns a array)
scaled_features = scaler.transform(df.drop('species',axis=1))
# Creating a df of the array'd scaled features
df_feat = pd.DataFrame(scaled_features, columns = df.drop('species',axis=1).columns)

## 7 Find correlation among variables.
# before standardisation
corr_pre = df.corr()
# after standardising
corr_pre_standadised = df_feat.corr()

## 8 Setting X and y
X = df_feat
y = df['species']

# 9 Creating the model
# Create function returning a compiled network
def create_network():
    # Create function returning a compiled network
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu', input_shape=(X.columns.nunique(),)))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy',  # Cross-entropy
                    optimizer='rmsprop',  # Root Mean Square Propagation
                    metrics=['accuracy'])  # Accuracy performance metric

    # Return compiled network
    return network



# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=create_network,
                                 epochs=10,
                                 batch_size=100,
                                 verbose=0)

# Evaluate neural network using 10-fold cross-validation
results = cross_val_score(neural_network, X, y, cv=10)

results.mean()