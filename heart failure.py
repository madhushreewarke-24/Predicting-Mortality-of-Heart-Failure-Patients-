
"""importing libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks

from sklearn.metrics import precision_score, recall_score, confusion_matrix, \
accuracy_score, f1_score, classification_report

"""import data and analyze"""

data_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
data_df

data_df.info()

data_df.isna().any().sum()

cols = ["#00EE00","#EE0000"]

ax = sns.countplot(x=data_df["DEATH_EVENT"],palette = cols)

ax = sns.countplot(x="DEATH_EVENT",hue="DEATH_EVENT", data=data_df)

plt.figure(figsize=(20,20))
sns.heatmap(data_df.corr(),cmap = "GnBu" ,annot=True)
plt.show()

plt.figure(figsize=(20,10))
sns.countplot(x=data_df["age"],data=data_df,hue = "DEATH_EVENT")
plt.show()

features = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]
for i in features:
    plt.figure(figsize=(10,7))
    sns.swarmplot(x=data_df["DEATH_EVENT"],y=data_df[i],color="black",alpha=0.7)
    sns.boxenplot(x=data_df["DEATH_EVENT"],y=data_df[i],palette=cols)
    plt.show()

data_df.isna()

"""data processing"""

x = data_df.drop("DEATH_EVENT",axis=1)
y = data_df["DEATH_EVENT"]

col_names = list(x.columns)
ss = StandardScaler()
x_scaled = ss.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled,columns=col_names)

x_scaled.describe().T

plt.figure(figsize=(20,10))
sns.boxenplot(data=x_scaled)
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.3)

"""model building
SVM
"""

model1 = svm.SVC()
model1.fit(x_train,y_train)

y_pred = model1.predict(x_test)

y_pred

np.array(y_test)

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="Blues")

accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))

"""ANN"""

early_stopping = callbacks.EarlyStopping(min_delta=0.01,patience=10,restore_best_weights=True)
model = Sequential()
model.add(Dense(units = 38,activation="relu",input_dim=12))

model.add(Dropout(0.25))
model.add(Dense(units = 8,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.summary()

history = model.fit(x_train,y_train,batch_size=20,epochs=100,validation_data=(x_test,y_test),callbacks=[early_stopping])

history_df = pd.DataFrame(history.history)
plt.plot(history_df.loc[:,["loss"]],label="trainin loss")
plt.plot(history_df.loc[:,["val_loss"]],label="val_loss")
plt.legend()
plt.show()

plt.plot(history_df.loc[:,['accuracy']],label="trainin accuracy")
plt.plot(history_df.loc[:,["val_accuracy"]],label="val_accuracy")
plt.legend()
plt.show()

y_pred = model.predict(x_test)

sns.heatmap(confusion_matrix(y_test,y_pred.round()),annot=True,cmap="Blues")

y_pred = (y_pred > 0.5)

print(classification_report(y_test,y_pred))

