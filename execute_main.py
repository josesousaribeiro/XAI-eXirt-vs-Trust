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

#analisys imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold



#initialize models
models = {'MLP': MLPClassifier(max_iter=100),
          'LGBM':lgb.LGBMClassifier(),
          'KNN': KNeighborsClassifier(n_neighbors=1),
          'DT':tree.DecisionTreeClassifier()}

print(models)


dataset = openml.datasets.get_dataset('37') #37 is dibates dataset

X, Y, categorical_indicator, attribute_names = dataset.get_data(
                  dataset_format="dataframe", target=dataset.default_target_attribute)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

#tunning models
for key in models:
  if key == 'MLP':
    params_grid = {'hidden_layer_sizes': [(10,),(10,20),(10,20,30)],
                   'activation': ['tanh', 'relu'],
                   'solver': ['sgd', 'adam'],
                   'alpha': [0.0001, 0.05],
                   'learning_rate': ['constant','adaptive']}
    
    grid_search = GridSearchCV(estimator = models[key],
                                param_grid = params_grid,
                                cv = StratifiedKFold(4), n_jobs = 1,
                                verbose = 0, scoring = 'roc_auc')

    # Fit the grid search to the data
    grid_search.fit(X, Y) #execute the cv in all instances of data
    print(grid_search.best_params_)
    models[key] = grid_search.best_estimator_
    models[key].fit(X_train,y_train) #fit the data with correct train split

  if key == 'LGBM':
    params_grid =  {'learning_rate': [0.1,0.5],
                    'max_depth': [1,2,3,4],
                    'n_estimators': [100,200,300],
                    'min_data_in_leaf': [40, 60,80]}

    grid_search = GridSearchCV(estimator = models[key],
                                param_grid = params_grid,
                                cv = StratifiedKFold(4), n_jobs = 1,
                                verbose = 0, scoring = 'roc_auc')
    break
    # Fit the grid search to the data
    grid_search.fit(X, Y) #execute the cv in all instances of data
    print(grid_search.best_params_)
    models[key] = grid_search.best_estimator_
    models[key].fit(X_train,y_train) #fit the data with correct train split
  if key == 'KNN':
    params_grid = {'leaf_size': [1, 10, 20, 40],
                   'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                   'metric': ['str', 'callable','minkowski'],
                   'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16]}

    grid_search = GridSearchCV(estimator = models[key],
                                 param_grid = params_grid,
                                 cv = StratifiedKFold(4), n_jobs = 1,
                                 verbose = 0, scoring = 'roc_auc')

    # Fit the grid search to the data
    grid_search.fit(X, Y) #execute the cv in all instances of data
    print(grid_search.best_params_)
    models[key] = grid_search.best_estimator_
    models[key].fit(X_train,y_train) #fit the data with correct train split
  if key == 'DT':
    params_grid = {'min_samples_leaf': [1, 10, 20, 40],
                   'max_depth': [1, 6, 12],
                   'criterion': ['gini','entropy'],
                   'splitter': ['best', 'random'],
                   'min_samples_split': [2, 5, 15, 20, 30]}

    grid_search = GridSearchCV(estimator = models[key],
                                 param_grid = params_grid,
                                 cv = StratifiedKFold(4), n_jobs = 1,
                                 verbose = 0, scoring = 'roc_auc')

    # Fit the grid search to the data
    grid_search.fit(X, Y) #execute the cv in all instances of data
    print(grid_search.best_params_)
    models[key] = grid_search.best_estimator_
    models[key].fit(X_train,y_train) #fit the data with correct train split


for key in models:
  print(models[key].fit(X_train,y_train))