import sys
import pandas as pd
from numpy import around

from sklearn.linear_model import LogisticRegression

# Nous importons le fichiers comprennat les 4 variables de base
df_pred = pd.read_csv(sys.argv[1])
df_train = pd.read_csv('./Data/notes.csv')

# prepare the dataset
df_X = df_train.iloc[:, 1:]
df_Y = df_train.iloc[:, 0]

df_X_train =  df_pred.iloc[:, :-1]                     

# define the model
model = LogisticRegression()

# fit the model
model.fit(df_X, df_Y)

# make predictions
df_pred['is_genuine'] = model.predict(df_X_train)
df_pred['False %'] = around(model.predict_proba(df_X_train)[:, 0], 3)
df_pred['True %'] = around(model.predict_proba(df_X_train)[:, 1], 3)

print(df_pred.iloc[:, -4:])