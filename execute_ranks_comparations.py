import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
import seaborn as sns
import string
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# 1. Configurações Iniciais de Caminho
bar = util.bar_system()
output_dataset_path = '' 

def bumpchart(df, ax, color_dic=None):
    """Versão ultra-compacta para publicações científicas."""
    left_yaxis = ax
    right_yaxis = left_yaxis.twinx()
    axes = [left_yaxis, right_yaxis]
    
    for col in df.columns:
        y = df[col]
        x = df.index.values
        for axis in axes[1:]: axis.plot(x, y, alpha=0)

        color = color_dic.get(col, '#7f7f7f') if color_dic else None
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
    left_yaxis.set_yticklabels(left_labels, fontsize=6)
    right_yaxis.set_yticklabels(right_labels, fontsize=6)
    return axes

def plot_into_grid(df_raw, model, test, ax, color_dic, letter):
    """Processa dados e desenha na grade com a letra no título."""
    original = df_raw.iloc[:, 0]
    sum_corr = 0
    for i in range(1, len(df_raw.columns)):
        rank = list(df_raw.iloc[:, i])
        corr, _ = stats.spearmanr(original, rank)
        sum_corr += corr

    ref_sequence = list(df_raw.iloc[:, 0])
    df_transformed = df_raw.copy()
    for col in df_raw.columns:
        for idx1, val1 in enumerate(ref_sequence):
            for idx2, val2 in enumerate(df_raw[col]):
                if val1 == val2:
                    df_transformed.at[idx1, col] = idx2
    
    df_transformed.index = ref_sequence
    df_transformed.columns = ['0%', '4%', '6%', '10%']
    
    # Título integrando a Letra, o Método, o Modelo e a Correlação
    ax.set_title(f"({letter}) {test.upper()} | {model.upper()} ($\sum c$: {round(sum_corr, 2)})", 
                 fontsize=6, fontweight='bold', pad=6)
    
    bumpchart(df_transformed.transpose(), ax, color_dic)
    ax.tick_params(axis='x', labelsize=6, pad=0)
    #ax.set_ylabel('Rank', fontsize=6, labelpad=0)
    return sum_corr

# 3. Execução Principal
csv_path = f'.{bar}output{output_dataset_path}{bar}csv{bar}df_explanation_analysis.csv'
df_master = pd.read_csv(csv_path, sep=',', index_col=0)

palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
features_list = df_master[df_master.columns[1]]
color_dic = {feat: palette[i % len(palette)] for i, feat in enumerate(features_list)}

models = ['lgbm', 'knn', 'mlp', 'dt']
tests = ['shap', 'skater', 'eXirt','dalex', 'eli5','lofo']

# Matriz para o Gráfico de Estabilidade
stability_matrix = pd.DataFrame(index=tests, columns=models)

# Criação da figura do Mosaico
fig_mosaico, axes = plt.subplots(len(tests), len(models), figsize=(6, 7.5), dpi=300)

letter_index = 0
for r, test in enumerate(tests):
    for c, model in enumerate(models):
        col_prefix = f"{test}_{model}_x_test_"
        cols = [f"{col_prefix}original", f"{col_prefix}4%_permute", f"{col_prefix}6%_permute", f"{col_prefix}10%_permute"]
        
        found_cols = [col for col in cols if col in df_master.columns]
        if not found_cols:
            col_prefix_low = f"{test.lower()}_{model}_x_test_"
            found_cols = [f"{col_prefix_low}{s}" for s in ["original", "4%_permute", "6%_permute", "10%_permute"] if f"{col_prefix_low}{s}" in df_master.columns]

        if len(found_cols) == 4:
            current_letter = string.ascii_uppercase[letter_index]
            corr_val = plot_into_grid(df_master[found_cols], model, test, axes[r, c], color_dic, current_letter)
            stability_matrix.loc[test, model] = corr_val
            letter_index += 1
        else:
            axes[r, c].set_axis_off()

plt.tight_layout(pad=1.0)

# --- PRODUÇÃO DOS ARQUIVOS SEPARADOS ---

# 1. Salva o Mosaico de Bump Charts
path_mosaico = f'.{bar}output{output_dataset_path}{bar}fig{bar}mosaico_compacto_paper.pdf'
with PdfPages(path_mosaico) as pdf:
    pdf.savefig(fig_mosaico, bbox_inches='tight')
    plt.close(fig_mosaico)

# 2. Cria e Salva o Gráfico de Estabilidade (Heatmap)
fig_explancao, ax_sum = plt.subplots(figsize=(6, 4))
sns.heatmap(stability_matrix.astype(float), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax_sum)
ax_sum.set_title("Explanation stability ($\sum c$ values) by XAI method and model", fontsize=10, fontweight='bold')
ax_sum.set_xlabel("Model", fontsize=8)
ax_sum.set_ylabel("XAI method", fontsize=8)

path_explancao = f'.{bar}output{output_dataset_path}{bar}fig{bar}estabilidade_explanacao.pdf'
with PdfPages(path_explancao) as pdf:
    pdf.savefig(fig_explancao, bbox_inches='tight')
    plt.close(fig_explancao)

print(f"Mosaico e Heatmap gerados com sucesso:")
print(f"1. PDF Mosaico: {path_mosaico}")
print(f"2. PDF Estabilidade: {path_explancao}")