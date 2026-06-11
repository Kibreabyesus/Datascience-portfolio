
# example how to use sklearn for feature engineering (scaling)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('penguins_simple.csv', sep=';')

X = df[['Culmen Length (mm)']]
y = df['Species']

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y)



scaler = MinMaxScaler()

scaler.fit(Xtrain)  # to learn the min and max of our data

Xtrain_scaled = scaler.transform(Xtrain)  # apply the transformation / scaling

Xtrain_scaled = pd.DataFrame(Xtrain_scaled)  # convert to pandas dataframe

print(Xtrain)
print(Xtrain_scaled)
