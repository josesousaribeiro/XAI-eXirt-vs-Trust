#xai methods imports
import eli5
import shap
import dalex as dx
import ciu




import numpy as np
import pandas as pd

from pyexirt.eXirt import Explainer
from eli5.sklearn import PermutationImportance
from lofo import LOFOImportance, FLOFOImportance, Dataset
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel


from io import StringIO
from bs4 import BeautifulSoup




def parse_html(box_scores):
    with open(box_scores) as f: 
        html = f.read()
    
    soup = BeautifulSoup(html, features="lxml")
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.theader")]
    
    # Use StringIO to wrap the HTML content
    html_buffer = StringIO(html)
    return BeautifulSoup(html_buffer, features="lxml")

def explainRankByEXirt(model, X_train, X_test, y_train, y_test, dataset_name_and_model):
    explainer = Explainer()
    rank, temp = explainer.explainRankByEXirt(model, X_train, X_test, y_train, y_test, dataset_name_and_model)
    return rank

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
    rank = eli5.show_weights(perm, feature_names = X.columns.tolist())
    rank = pd.read_html(rank.data)[0]
    rank = rank.sort_values(by=['Weight','Feature'], ascending=False) #fix problem of equals values of explaination
    return rank['Feature'].to_list()


def explainRankByKernelShap(model,x_features_names, X): # shap.sample(data, K) or shap.kmeans(data, K)
    np.random.seed(0)
    explainer = shap.KernelExplainer(model.predict_proba, X[:],nsamples=len(x_features_names))
    shap_values = explainer.shap_values(X[:])
    vals= np.abs(shap_values).mean(0)
    temp_df = pd.DataFrame(list(zip(x_features_names, sum(vals.T))), columns=['feat_name','shap_value'])
    temp_df = temp_df.sort_values(by=['shap_value','feat_name'], ascending=False) #fix problem of equals values of explaination
    return temp_df['feat_name'].to_list()

def explainRankByKernelShap_fixed(
    model,
    x_features_names,
    X,
    *,
    background_method: str = "sample",  # "sample" | "kmeans"
    background_size: int = 100,
    nsamples: int = 100,  # nº de coalizões aproximadas (↑ = mais preciso, mais lento)
    random_state: int = 0,
):
    """
    Retorna:
      - importance_df: ranking global por |SHAP| médio
      - shap_df: DataFrame (n_amostras x n_features) com os SHAP values
    """
    rng = np.random.default_rng(random_state)
    debug = True
    # --- Preparar X e nomes ---
    X_is_df = hasattr(X, "values") and hasattr(X, "index")
    X_np = X.values if X_is_df else np.asarray(X)
    if X_np.ndim != 2:
        raise ValueError(f"X deve ser 2D (amostras x features). Recebido shape={X_np.shape}.")

    n_rows, n_cols = X_np.shape
    index = X.index if X_is_df else pd.RangeIndex(n_rows)

    # Resolver nomes de features
    if x_features_names is not None:
        feat_names = list(x_features_names)
        if len(feat_names) != n_cols:
            raise ValueError(
                f"Quantidade de nomes ({len(feat_names)}) ≠ nº de colunas de X ({n_cols}). "
                "Garanta correspondência 1:1."
            )
    elif X_is_df:
        feat_names = list(X.columns)
        if len(feat_names) != n_cols:
            raise ValueError(
                f"X.columns tem {len(feat_names)} nomes, mas X possui {n_cols} colunas."
            )
    else:
        feat_names = [f"x{i}" for i in range(n_cols)]

    if debug:
        print(f"[DEBUG] X shape: {X_np.shape} | n_features esperadas: {len(feat_names)}")

    # --- Detectar tarefa e função f(x) ---
    is_classifier = getattr(model, "_estimator_type", None) == "classifier" or hasattr(model, "predict_proba")
    if is_classifier and hasattr(model, "predict_proba"):
        proba_shape = model.predict_proba(X_np[:1]).shape
        if len(proba_shape) == 2 and proba_shape[1] == 2:
            f = lambda Z: model.predict_proba(Z)[:, 1]
            multi = False
            if debug:
                print("[DEBUG] Tarefa: classificação binária (usando prob da classe positiva).")
        else:
            f = model.predict_proba
            multi = True
            if debug:
                print(f"[DEBUG] Tarefa: multiclasse (shape proba={proba_shape}).")
    else:
        f = model.predict
        multi = False
        if debug:
            print("[DEBUG] Tarefa: regressão ou classificador sem predict_proba (usando predict).")

    # --- Background para KernelSHAP ---
    if background_method == "kmeans":
        K = min(background_size, max(1, n_rows))
        background = shap.kmeans(X_np, K)
        if debug:
            print(f"[DEBUG] Background: kmeans com K={K}")
    else:
        if n_rows > background_size:
            idx = rng.choice(n_rows, size=background_size, replace=False)
            background = X_np[idx]
        else:
            background = X_np
        if debug:
            print(f"[DEBUG] Background: sample com shape {getattr(background,'shape',None)}")

    # --- Explainer & SHAP ---
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_np, nsamples=nsamples)

    # --- Construir shap_df e importância global ---
    if multi:
        # lista de arrays [n, m] por classe
        per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]          # k x m
        mean_abs = np.mean(np.vstack(per_class), axis=0)                     # m

        # média entre classes por amostra -> (n, m)
        sv_stack = np.stack(shap_values, axis=-1)                            # (n, m, k)
        sv_avg = sv_stack.mean(axis=2)                                       # (n, m)
        shap_df = pd.DataFrame(sv_avg, index=index, columns=feat_names)
    else:
        sv = shap_values if isinstance(shap_values, np.ndarray) else shap_values[0]
        if sv.ndim != 2 or sv.shape[1] != n_cols:
            raise ValueError(
                f"Formato inesperado de SHAP values: {sv.shape}. "
                f"Esperado (n_amostras, n_features)=({n_rows}, {n_cols})."
            )
        mean_abs = np.abs(sv).mean(axis=0)                                   # m
        shap_df = pd.DataFrame(sv, index=index, columns=feat_names)

    total = mean_abs.sum() or 1.0
    importance_df = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .assign(pct_contrib=lambda d: 100 * d["mean_abs_shap"] / total)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    if debug:
        print(f"[DEBUG] shap_df shape: {shap_df.shape} | importance_df linhas: {len(importance_df)}")

    return importance_df, shap_df

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


#create context dictionary (necessary to CIU)
def is_int(n):
    for i in range(len(n)):
      if isinstance(n[i],float):
        return False
    return True

def explainRankNewCiu(model,X, X_train, y_train, X_test):
    out_minmaxs = create_dic_ciu(X)
    CIU = ciu.CIU(model.predict, ['0','1'], data=X_train, out_minmaxs=out_minmaxs)
    CIUres = CIU.explain_all(X_test, do_norm_invals=True)
    print(CIUres)


def create_dic_ciu(X):
    attribute_names = X.columns
    context_dic = {}
    for k in range(len(attribute_names)):
        context_dic[attribute_names[k]] = [min(X[attribute_names[k]]), max(X[attribute_names[k]]), is_int(X[attribute_names[k]])]
    return context_dic

def explainRankByCiu(model, x_test, feature_names,context_dic,rank):

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