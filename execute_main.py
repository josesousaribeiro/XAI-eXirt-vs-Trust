
import os
import wget

#dataset import
import openml
import util

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
from explanable_tools import explainRankByEli5, explainRankByEXirt, explainRankByKernelShap, explainRankByLofo, explainRankDalex, explainRankSkater, explainRankNewCiu
import pandas as pd


if os.path.isfile(os.path.join(os.getcwd(),'decodIRT_MLtIRT.py')) == False:
                  wget.download('https://raw.githubusercontent.com/josesousaribeiro/eXirt/main/pyexirt/decodIRT_MLtIRT.py')

if os.path.isfile(os.path.join(os.getcwd(),'decodIRT_analysis.py')) == False:
                  wget.download('https://raw.githubusercontent.com/josesousaribeiro/eXirt/main/pyexirt/decodIRT_analysis.py')


bar = util.bar_system()

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


X = z_score(X)
Y = y_as_binary(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=seed)


#tunning models
for key in models:
  if key == 'mlp':
    params_grid = {'max_iter' : [3000],
                   'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                   'solver' : ['sgd', 'adam'],
                   'alpha' : [0.005, 0.01, 0.015],
                   'hidden_layer_sizes': [
                    (5,),(10,),(15,),(5,5,),(5,10),(5,15,),(10,5),(10,10,),(10,15,),(15,5,),(15,10,),(15,15,)
                  ]
    }
  else:
    if key == 'lgbm':
      params_grid =  {'learning_rate': [0.01, 0.015, 0.02],
                      'max_depth': [2, 3, 4, 5, 6],
                      'n_estimators': [200,300, 400, 500],
                      'min_data_in_leaf': [40, 60,80],
                      'colsample_bytree': [0.7, 1]}
    else:
      if key == 'knn':
        params_grid = {'leaf_size': [5, 10, 20, 40, 50],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                      'metric': ['minkowski','cityblock','euclidean'],
                      'n_neighbors': [2, 4, 6, 8, 10, 12]}
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
  print('Apply gridsearch in '+key+'...best model is:')
  grid_search.fit(X, Y) #execute the cv in all instances of data
  models[key] = grid_search.best_estimator_
  models[key].fit(X_train,y_train) #fit the data with correct train split
  print(models[key])
  print()
#generate perturbed datasets to test



tests = {
    'x_test_original': X_test.copy(),
    'x_test_10%_permute': apply_perturbation_permute(X_test.copy(deep=True), 0.1, 10),
    'x_test_20%_permute': apply_perturbation_permute(X_test.copy(deep=True), 0.2, 20),
    'x_test_30%_permute': apply_perturbation_permute(X_test.copy(deep=True), 0.3, 30)
    }

df_performance_analysis = pd.DataFrame(index=['accuracy','precision','recall','f1','roc_auc'])
for i in models:
  for j in tests:

    y_pred = models[i].predict(tests[j])
    
    accuracy, precision, recall, f1, roc_auc = model_output_analysis(y_test, y_pred)

    df_performance_analysis.at['accuracy',i+'_'+j] = round(accuracy,3)
    df_performance_analysis.at['precision',i+'_'+j] = round(precision,3)
    df_performance_analysis.at['recall',i+'_'+j] = round(recall,3)
    df_performance_analysis.at['f1',i+'_'+j] = round(f1,3)
    df_performance_analysis.at['roc_auc',i+'_'+j] = round(roc_auc,3)

print(df_performance_analysis)
df_performance_analysis.to_csv('.'+bar+'output'+bar+'csv'+bar+'df_performance_analysis.csv',sep=',')


df_explanation_analysis = pd.DataFrame()
for i in models:
  for j in tests: 
  
    #explanation by exirt
    #print('eXirt explaning...')
    #print('Explaining M1...')
    #df_feature_rank['exirt_m1'], temp = explainer.explainRankByEXirt(model_m1, X_data_train, X_data_test, y_data_train, y_data_test,code_datasets[i],model_name='m1')
    
    X_data = X_test.copy(deep=True)

    #explanation by skater

    #print('ciu explaning...'+i+'_'+j)
    #df_explanation_analysis['ciu_'+i+'_'+j] = explainRankNewCiu(models[i],X, X_train.copy(), y_train.copy(), tests[j].copy(deep=True))

    
    print('Shap explaning...'+i+'_'+j)
    df_explanation_analysis['shap_'+i+'_'+j] = explainRankByKernelShap(models[i], tests[j].columns, tests[j].copy(deep=True))

    print('EXirt explaing...'+i+'_'+j)
    df_explanation_analysis['eXirt_'+i+'_'+j] = explainRankByEXirt(models[i],X_train,tests[j],y_train, y_test, 'diabetes_'+i+'_'+j)
    
    print('Skater explaning...'+i+'_'+j)
    df_explanation_analysis['skater_'+i+'_'+j] = explainRankSkater(models[i], tests[j].copy(deep=True))

    print('Eli5 explaning...'+i+'_'+j)
    df_explanation_analysis['eli5_'+i+'_'+j] = explainRankByEli5(models[i], tests[j].copy(deep=True), y_test)
    
    print('Dalex explaning...'+i+'_'+j)
    df_explanation_analysis['dalex_'+i+'_'+j] = explainRankDalex(models[i],tests[j].copy(deep=True), y_test)

    print('Lofo explaning...'+i+'_'+j)
    df_explanation_analysis['lofo_'+i+'_'+j] = explainRankByLofo(models[i], tests[j].copy(deep=True), y_test, tests[j].columns)

    

    #eXirt
    
df_explanation_analysis.to_csv('.'+bar+'output'+bar+'csv'+bar+'df_explanation_analysis.csv',sep=',')