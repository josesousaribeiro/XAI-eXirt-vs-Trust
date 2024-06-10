#xai methods imports
import eli5
import shap
import dalex as dx
import ciu
from eli5.sklearn import PermutationImportance
from lofo import LOFOImportance, FLOFOImportance, Dataset
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

import numpy as np
import pandas as pd


def explainRankByLofo(model,X,Y,names_x_attributes):
    df = X.copy()
    df['class'] = Y.to_list()
    dataset = Dataset(df=df, target="class", features=names_x_attributes)
    fi = LOFOImportance(dataset, scoring='accuracy', model=model)
    importances = fi.get_importance()
    importances = importances.sort_values(by=['importance_mean','feature'],ascending=False) #fix problem of equals values of explaination
    return importances['feature'].to_list()

def explainRankByEli5(model, X, Y):
    perm = PermutationImportance(model, random_state=42).fit(X, Y)
    rank = eli5.show_weights(perm, feature_names = X.columns.tolist(),top=-1)
    rank = pd.read_html(rank.data)[0]
    rank = rank.sort_values(by=['Weight','Feature'], ascending=False) #fix problem of equals values of explaination
    return rank['Feature'].to_list()


def explainRankByKernelShap(model,x_features_names, X,is_gradient=False): # shap.sample(data, K) or shap.kmeans(data, K)
    np.random.seed(0)
    explainer = shap.KernelExplainer(model.predict_proba, X[:],nsamples=len(x_features_names))
    shap_values = explainer.shap_values(X[:])
    if is_gradient == False:
        vals= np.abs(shap_values).sum(1)
    else:
        vals= np.abs([shap_values]).sum(1) #correction []
    temp_df = pd.DataFrame(list(zip(x_features_names, sum(vals))), columns=['feat_name','shap_value'])
    temp_df = temp_df.sort_values(by=['shap_value','feat_name'], ascending=False) #fix problem of equals values of explaination
    return list(temp_df['feat_name'])


def explainRankByTreeShap(model, x_features_names, X, is_gradient=False):
    np.random.seed(0)
    shap_values = shap.TreeExplainer(model, feature_perturbation='interventional').shap_values(X)
    if is_gradient == False:
        vals= np.abs(shap_values).mean(0)
    else:
        vals= np.abs([shap_values]).mean(0) #correction []
        temp_df = pd.DataFrame(list(zip(x_features_names, sum(vals))), columns=['feat_name','shap_value'])
        temp_df = temp_df.sort_values(by=['shap_value','feat_name'], ascending=False) #fix problem of equals values of explaination

    return temp_df['feat_name'].to_list()

def explainRankByCiu(model, x_test, X, feature_names,context_dic,rank):

    def _makeRankByCu(ciu):
        df_cu = pd.DataFrame(list(ciu.cu.items()), columns=['attribute', 'cu'])
        df_cu = df_cu.sort_values(by='cu', ascending=False)
        #ciu.plot_cu()
        return df_cu['attribute'].to_list()

    def _makeRankByCi(ciu):
        df_ci = pd.DataFrame(list(ciu.ci.items()), columns=['attribute', 'ci'])
        df_ci = df_ci.sort_values(by=['ci','attribute'], ascending=False) #fix problem of equals values of explaination
        return df_ci['attribute'].to_list()

    case = x_test.values[0]
    example_prediction = model.predict([x_test.values[0]])
    example_prediction_probs = model.predict_proba([x_test.values[0]])
    prediction_index = list(example_prediction_probs[0]).index(max(example_prediction_probs[0]))

    ciu = determine_ciu(
        x_test.iloc[[1]],
        model.predict_proba,
        X.to_dict('list'),
        samples = 1000,
        prediction_index = 1)

    if rank == 'ci':
        result = _makeRankByCi(ciu)
    else:
        if rank == 'cu':
            result = _makeRankByCu(ciu)
        else:
            result = {}

    #ciu
    return result


def explainRankSkater(model, X):
    interpreter = Interpretation(X.to_numpy(), feature_names=X.columns.to_list())

    model_new = InMemoryModel(model.predict_proba,
                                examples=X.to_numpy(),
                                unique_values=model.classes_)

    rank = interpreter.feature_importance.feature_importance(model_new,
                                                                ascending=False,
                                                                progressbar=False
                                                                #n_jobs=1
                                                                )
    rank = rank.to_frame(name='values')
    rank = rank.reset_index()
    rank = rank.rename(columns={'index':'variable','values':'values'})
    rank = rank.sort_values(by=['values','variable'], ascending=False) #fix problem of equals values of explaination
    return rank['variable'].to_list()

def explainRankDalex(model, X_train, y_train):
    explainer = dx.Explainer(model, X_train, y_train,verbose=False)
    explanation = explainer.model_parts()
    rank = explanation.result
    rank = rank[rank.variable != '_baseline_']
    rank = rank[rank.variable != '_full_model_']
    rank = rank.sort_values(by=['dropout_loss','variable'], ascending=False) #fix problem of equals values of explaination
    return rank['variable'].tolist()