import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sea
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import keras.callbacks
from datetime import datetime
#Importing the dataset
diabetes = pd.read_csv('diabetes_con_headers.csv')
dataset = diabetes
dataset.info()
#Example of violinplot using library Seaborn
graph1 = sea.violinplot(x='Outcome', y='Pregnancies', data=dataset, palette='muted', split=True)
plt.show()
#Correlation between characteristics of dataset
correl = dataset.corr()
print(correl)
#Subdivision of the dataset
Y= diabetes.Outcome
X = diabetes.drop('Outcome',axis=1)
columns = X.columns
#Standardization of the dataset
scaler = StandardScaler()
XScaler = scaler.fit_transform(X)
dataXScaler = pd.DataFrame(XScaler, columns = columns)
# Define the Sequential keras Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(XScaler, Y, epochs=150, batch_size=10)
_,accuracy = model.evaluate(XScaler, Y)
print(str(accuracy))
# costruire Y  ossia predictions con il modello della rete neurale
predictions = model.predict(XScaler)
IndexValues = []
RealValues = []
PredictValues = []
TotalValues = []
Ypredictions = [round(x[0]) for x in predictions]
for i in range(30):
    #print('%s => %d (expected %d)' % (XScaler[i].tolist(), Ypredictions[i], Y[i]))
    IndexValues.append(i)
    RealValues.append(Y[i])
    PredictValues.append(Ypredictions[i])
    TotalValues.append([i,Y[i],Ypredictions[i]])
df = pd.DataFrame(TotalValues, columns =['Nr', 'RealValue','PredicValue'])
df2= df.melt('Nr', var_name='cols', value_name='vals')
sea.lineplot(x="Nr", y="vals", hue="cols", data=df2,alpha=.5)
plt.savefig("sea_1_.png")
