import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
import string
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# 1. Configurações Iniciais
bar = util.bar_system()
output_dataset_path = '' 

def bumpchart(df, ax, color_dic=None):
    """Versão com ajuste de margem Y para garantir respiro no topo e base."""
    left_yaxis = ax
    right_yaxis = left_yaxis.twinx()
    axes = [left_yaxis, right_yaxis]
    
    for col in df.columns:
        y = df[col]
        x = df.index.values
        for axis in axes[1:]: axis.plot(x, y, alpha=0)

        color = color_dic.get(col) if color_dic else None
        left_yaxis.plot(x, y, linewidth=1.5, alpha=0.7, color=color, solid_capstyle='round')
        left_yaxis.scatter(x, y, s=8, alpha=0.9, color=color)

    lines = len(df.columns)
    y_ticks = [*range(0, lines)]
    
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines - 0.5, -0.5))
    
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index
    left_yaxis.set_yticklabels(left_labels, fontsize=5)
    right_yaxis.set_yticklabels(right_labels, fontsize=5)
    return axes

def calculate_ard_data(df_master, test_name, models):
    """Calcula a métrica ARD e prepara os dados transformados antes da plotagem."""
    baseline_cols = []
    valid_model_names = []
    
    for model in models:
        col_name = f"{test_name}_{model}_x_test_original"
        if col_name not in df_master.columns:
            col_name = f"{test_name.lower()}_{model}_x_test_original"
        
        if col_name in df_master.columns:
            baseline_cols.append(col_name)
            valid_model_names.append(model.upper())

    if len(baseline_cols) > 1:
        df_plot = df_master[baseline_cols].copy()
        
        # Transformação para Posições de Rank
        ref_sequence = list(df_plot.iloc[:, 0]) # LGBM como referência
        df_transformed = df_plot.copy()
        for col in df_plot.columns:
            for idx1, val1 in enumerate(ref_sequence):
                for idx2, val2 in enumerate(df_plot[col]):
                    if val1 == val2:
                        df_transformed.at[idx1, col] = idx2
        
        df_transformed.index = ref_sequence
        
        # Cálculo da Métrica ARD
        baseline_vector = df_transformed.iloc[:, 0].values
        distances = []
        for i in range(1, len(df_transformed.columns)):
            comparison_vector = df_transformed.iloc[:, i].values
            dist = np.linalg.norm(baseline_vector - comparison_vector)
            distances.append(dist)
        
        ard_val = np.mean(distances)
        return ard_val, df_transformed, valid_model_names
    return None, None, None

# --- EXECUÇÃO ---

csv_path = f'.{bar}output{output_dataset_path}{bar}csv{bar}df_explanation_analysis.csv'
df_master = pd.read_csv(csv_path, sep=',', index_col=0)

# Cores consistentes baseadas nas features (index do dataframe)
features_list = df_master.index
palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
color_dic = {feat: palette[i % len(palette)] for i, feat in enumerate(features_list)}

models = ['lgbm', 'knn', 'mlp', 'dt']
tests = ['eXirt', 'shap', 'skater', 'dalex', 'eli5', 'lofo']

# 1. Pré-processamento: Calcular ARD para todos e armazenar
results = []
for test in tests:
    ard, data, model_names = calculate_ard_data(df_master, test, models)
    if ard is not None:
        results.append({
            'test_name': test,
            'ard': ard,
            'data': data,
            'model_names': model_names
        })

# 2. Ordenação Crescente por ARD
results_sorted = sorted(results, key=lambda x: x['ard'])

# 3. Plotagem do Mosaico 2x3
fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=300)
axes_flat = axes.flatten()

for i, res in enumerate(results_sorted):
    ax = axes_flat[i]
    letter = string.ascii_uppercase[i]
    
    # Customização do Título e Eixos
    ax.set_title(f"({letter}) {res['test_name'].upper()} | Model Dependence\n$ARD$: {round(res['ard'], 3)}", 
                 fontsize=7, fontweight='bold', pad=8)
    
    # Plotagem usando o bumpchart original
    res['data'].columns = res['model_names']
    bumpchart(res['data'].transpose(), ax, color_dic)
    ax.tick_params(axis='x', labelsize=6)

# Desativa slots excedentes
for j in range(len(results_sorted), len(axes_flat)):
    axes_flat[j].set_axis_off()

plt.tight_layout(pad=3.0)

path_output = f'.{bar}output{output_dataset_path}{bar}fig{bar}comparativo_dependencia_modelo_ordenado.pdf'
with PdfPages(path_output) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"Mosaico 2x3 ordenado por ARD crescente gerado com sucesso: {path_output}")