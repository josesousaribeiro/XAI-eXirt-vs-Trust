#xai methods imports
import eli5
import shap
import dalex as dx
import ciu
from eli5.sklearn import PermutationImportance
from lofo import LOFOImportance, FLOFOImportance, Dataset
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

#dataset import
import openml

#models imports
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier

#analysis imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


#analysis data

from analysis import * 
import pandas as pd


seed = 42

#initialize models
models = {
          'mlp': MLPClassifier(verbose=False),
          'lgbm':lgb.LGBMClassifier(verbosity=-1),
          'knn': KNeighborsClassifier(),
          'dt':tree.DecisionTreeClassifier()
          }




dataset = openml.datasets.get_dataset('37') #37 is dibates dataset

X, Y, categorical_indicator, attribute_names = dataset.get_data(
                  dataset_format="dataframe", target=dataset.default_target_attribute)


X = normalize(X)
Y = y_as_binary(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)


#tunning models
for key in models:
  if key == 'mlp':
    params_grid = {'max_iter' : [3000],
                   'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                   'solver' : ['sgd', 'adam'],
                   'alpha' : [0.005, 0.01, 0.015],
                   'hidden_layer_sizes': [
                    (5,),(10,),(15,),(5,5,),(5,10),(5,15,),(10,5),(10,10,),(10,15,),(15,5,),(15,10,),(15,15,)
                  ]}
  else:
    if key == 'lgbm':
      params_grid =  {'learning_rate': [0.1, 0.15, 0.2],
                      'max_depth': [2, 3, 4, 5, 6],
                      'n_estimators': [50, 100,200,300, 400],
                      'min_data_in_leaf': [40, 60,80]}
    else:
      if key == 'knn':
        params_grid = {'leaf_size': [1, 10, 20, 40],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                      'metric': ['minkowski','cityblock','euclidean'],
                      'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16]}
      else:
        if key == 'dt':
          params_grid = {'min_samples_leaf': [1, 10, 20],
                        'max_depth': [1, 6, 12],
                        'criterion': ['gini','entropy'],
                        'splitter': ['best', 'random'],
                        'min_samples_split': [2, 5, 15, 20, 30]}

  grid_search = GridSearchCV(estimator = models[key],
                                param_grid = params_grid,
                                cv = StratifiedKFold(4), n_jobs = 1,
                                verbose = 0, scoring = 'roc_auc')

  # Fit the grid search to the data
  print()
  print('Apply gridsearch in '+key+'...best models is:')
  grid_search.fit(X, Y) #execute the cv in all instances of data
  models[key] = grid_search.best_estimator_
  models[key].fit(X_train,y_train) #fit the data with correct train split
  print(models[key])
  print()
#generate perturbed datasets to test

tests = {
    'x_test_0%': X_test.copy(),
    'x_test_15%': apply_perturbation(X_test.copy(), 0.15, 1),
    'x_test_30%': apply_perturbation(X_test.copy(), 0.3, 100)}

df_analysis = pd.DataFrame(index=['accuracy','precision','recall','f1','roc_auc'])
for i in models:
  for j in tests:

    y_pred = models[i].predict(tests[j])
    
    accuracy, precision, recall, f1, roc_auc = model_output_analysis(y_test, y_pred)

    df_analysis.at['accuracy',i+'_'+j] = accuracy
    df_analysis.at['precision',i+'_'+j] = precision
    df_analysis.at['recall',i+'_'+j] = recall
    df_analysis.at['f1',i+'_'+j] = f1
    df_analysis.at['roc_auc',i+'_'+j] = roc_auc

print(df_analysis)