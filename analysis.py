import numpy
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score




def apply_perturbation(df,percent,seed):
  #aplica ruido a cada instâcia do atributo da vez
  number_of_instances = len(df.index)
  for c in df.columns:
      seed = seed + 1  
      numpy.random.seed(seed)  
      noise = numpy.random.normal(0, percent, number_of_instances)
      df[c] = df[c] + noise
  return df

def model_output_analysis(y_test,y_pred):
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      f1 = f1_score(y_test,y_pred)
      roc_auc = roc_auc_score(y_test,y_pred)
      return accuracy, precision, recall, f1, roc_auc


def normalize(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
      if df[column].dtype != 'category':
        if(len(df_norm[column].unique()) > 1): #fix NaN generation
          df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        else:
          df_norm[column] = 0
    return df_norm

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()


def y_as_binary(y):
   return  y.map({"tested_positive":1, "tested_negative":0})


