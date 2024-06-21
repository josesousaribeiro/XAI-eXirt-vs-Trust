
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
                   'activation' : ['sigmoid','identity', 'logistic', 'tanh', 'relu'],
                   'solver' : ['sgd', 'adam'],
                   'alpha' : [0.005, 0.01, 0.015],
                   'hidden_layer_sizes': [
                    (4,),(8,),(16,),(4,4,),(4,8),(4,16,),(8,4),(8,8,),(8,16,),(16,4,),(16,8,),(16,16,)
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
        params_grid = {'leaf_size': [5, 10, 15, 120, 25],
                      'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                      'metric': ['minkowski','cityblock','euclidean'],
                      'n_neighbors': [2, 3, 4, 5,6]}
      else:
        if key == 'dt':
          params_grid = {'min_samples_leaf': [1,2,3,4],
                        'max_depth': [1, 2, 3],
                        'criterion': ['gini','entropy'],
                        'min_samples_split': [1, 2, 3, 4]}

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
  print(models[key].get_params())
  print()
#generate perturbed datasets to test


df_0 = X_test.copy(deep=True)

df_1 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.01, 10)
df_2 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.02, 20)
df_3 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.03, 30)
df_4 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.04, 40)
df_5 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.05, 50)
df_6 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.06, 60)
df_7 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.07, 70)
df_8 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.08, 80)
df_9 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.09, 90)
df_10 = apply_perturbation_in_sample(X_test.copy(deep=True), 0.10, 100)



tests = {
    'x_test_original': df_0,
    'x_test_1%_permute': df_1,
    'x_test_2%_permute': df_2,
    'x_test_3%_permute': df_3,
    'x_test_4%_permute': df_4,
    'x_test_5%_permute': df_5,
    'x_test_6%_permute': df_6,
    'x_test_7%_permute': df_7,
    'x_test_8%_permute': df_8,
    'x_test_9%_permute': df_9,
    'x_test_10%_permute': df_10
    }

for i in tests:
    tests[i].to_csv('.'+bar+'output'+bar+'csv'+bar+i+'.csv',sep=',')

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
    
    #explanation by skater

    #print('ciu explaning...'+i+'_'+j)
    #df_explanation_analysis['ciu_'+i+'_'+j] = explainRankNewCiu(models[i],X, X_train.copy(), y_train.copy(), tests[j].copy(deep=True))

    
    print('Shap explaning...'+i+'_'+j)
    df_explanation_analysis['shap_'+i+'_'+j] = explainRankByKernelShap(models[i], tests[j].columns, tests[j].copy(deep=True))

    print('EXirt explaing...'+i+'_'+j)
    df_explanation_analysis['eXirt_'+i+'_'+j] = explainRankByEXirt(models[i],X_train,tests[j].copy(deep=True),y_train, y_test, 'diabetes_'+i+'_'+j)
    
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