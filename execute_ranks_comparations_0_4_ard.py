import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
import seaborn as sns
import string
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator

# 1. Configurações Iniciais
bar = util.bar_system()

output_dataset_path = '' 
#output_dataset_path = '_banknote_authentication'
#output_dataset_path = '_diabetes'
#output_dataset_path = '_phoneme'
#output_dataset_path = '_mozilla4'

def bumpchart(df, ax, color_dic=None):
    """Versão robusta: garante que os nomes das variáveis apareçam nos eixos Y."""
    left_yaxis = ax
    right_yaxis = left_yaxis.twinx()
    axes = [left_yaxis, right_yaxis]
    
    # Plota as linhas e pontos
    for col in df.columns:
        y = df[col]
        x = df.index.values
        for axis in axes[1:]: axis.plot(x, y, alpha=0)

        color = color_dic.get(col) if color_dic else None
        left_yaxis.plot(x, y, linewidth=1.5, alpha=0.7, color=color, solid_capstyle='round')
        left_yaxis.scatter(x, y, s=8, alpha=0.9, color=color)

    # Identifica a quantidade de features
    num_features = len(df.columns)
    y_ticks = np.arange(num_features)
    
    # Extrai os nomes das variáveis na ordem correta do rank
    # df.iloc[0] é a linha '0%', df.iloc[-1] é a linha '10%'
    left_labels = df.iloc[0].sort_values().index.tolist()
    right_labels = df.iloc[-1].sort_values().index.tolist()

    for axis, labels in zip(axes, [left_labels, right_labels]):
        axis.invert_yaxis()
        # Define os ticks e os rótulos de forma explícita para evitar que virem números
        axis.yaxis.set_major_locator(FixedLocator(y_ticks))
        axis.set_yticklabels(labels, fontsize=5)
        axis.set_ylim((num_features - 0.5, -0.5))
        
    return axes

def calculate_ard_metric(df_raw):
    """Calcula ARD e transforma dados mantendo os nomes como referências."""
    ref_sequence = list(df_raw.iloc[:, 0])
    df_transformed = df_raw.copy()
    
    # Converte nomes em rankings numéricos
    for col in df_raw.columns:
        for idx1, val1 in enumerate(ref_sequence):
            for idx2, val2 in enumerate(df_raw[col]):
                if val1 == val2:
                    df_transformed.at[idx1, col] = idx2
    
    # Garante que o índice do DataFrame sejam os nomes das features
    df_transformed.index = ref_sequence
    
    baseline = df_transformed.iloc[:, 0].values
    distances = []
    for i in range(1, len(df_transformed.columns)):
        perturbed = df_transformed.iloc[:, i].values
        distances.append(np.linalg.norm(baseline - perturbed))
    
    return np.mean(distances), df_transformed

def plot_into_grid(df_transformed, ard_val, model, test, ax, color_dic, letter):
    """Renderiza cada subgráfico no mosaico."""
    df_plot = df_transformed.copy()
    df_plot.columns = ['0%', '4%', '6%', '10%']
    
    ax.set_title(f"({letter}) {test.upper()} | {model.upper()} ($ARD$: {round(ard_val, 3)})", 
                 fontsize=6, fontweight='bold', pad=6)
    
    # Transpomos para que as features virem colunas dentro do bumpchart
    bumpchart(df_plot.transpose(), ax, color_dic)
    ax.tick_params(axis='x', labelsize=6, pad=0)
    return ard_val

# 3. Execução Principal
csv_path = f'.{bar}output{output_dataset_path}{bar}csv{bar}df_explanation_analysis.csv'
df_master = pd.read_csv(csv_path, sep=',', index_col=0)

# Mapeamento de cores persistente
palette = [
    # Paleta Base (Matplotlib tab10)
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    
    # Tons Suaves (Matplotlib tab20 light)
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    
    # Tons Escuros e Profundos
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', 
    '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
    
    # Variações Terrosas e Vibrantes
    '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', 
    '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6'
]
# Usamos o índice (nomes das variáveis) para fixar as cores
features_list = df_master.index.tolist()
color_dic = {feat: palette[i % len(palette)] for i, feat in enumerate(features_list)}

models = ['lgbm', 'knn', 'mlp', 'dt']
tests = ['shap', 'skater', 'eXirt','dalex', 'eli5','lofo']

# --- ETAPA DE PRÉ-CÁLCULO ---
test_performance = {}
cached_data = {}

for test in tests:
    ard_sum = 0
    count = 0
    for model in models:
        col_prefix = f"{test}_{model}_x_test_"
        cols = [f"{col_prefix}original", f"{col_prefix}4%_permute", f"{col_prefix}6%_permute", f"{col_prefix}10%_permute"]
        
        found_cols = [col for col in cols if col in df_master.columns]
        if not found_cols:
             col_prefix_low = f"{test.lower()}_{model}_x_test_"
             found_cols = [f"{col_prefix_low}{s}" for s in ["original", "4%_permute", "6%_permute", "10%_permute"] if f"{col_prefix_low}{s}" in df_master.columns]

        if len(found_cols) == 4:
            val, data = calculate_ard_metric(df_master[found_cols])
            cached_data[(test, model)] = (val, data)
            ard_sum += val
            count += 1
    
    test_performance[test] = ard_sum / count if count > 0 else 999

# Ordenação crescente por ARD médio
sorted_tests = sorted(tests, key=lambda x: test_performance[x])

# --- GERAÇÃO DO MOSAICO ---
fig_mosaico, axes = plt.subplots(len(sorted_tests), len(models), figsize=(6, 7.5), dpi=300)
stability_matrix = pd.DataFrame(index=sorted_tests, columns=models)

letter_index = 0
for r, test in enumerate(sorted_tests):
    for c, model in enumerate(models):
        if (test, model) in cached_data:
            ard_val, data = cached_data[(test, model)]
            current_letter = string.ascii_uppercase[letter_index]
            plot_into_grid(data, ard_val, model, test, axes[r, c], color_dic, current_letter)
            stability_matrix.loc[test, model] = ard_val
            letter_index += 1
        else:
            axes[r, c].set_axis_off()

plt.tight_layout(pad=1.0)

# --- SALVAMENTO ---
path_mosaico = f'.{bar}output{output_dataset_path}{bar}fig{bar}mosaico_ordenado_ard.pdf'
with PdfPages(path_mosaico) as pdf:
    pdf.savefig(fig_mosaico, bbox_inches='tight')
    plt.close(fig_mosaico)

# Heatmap final
fig_explancao, ax_sum = plt.subplots(figsize=(6, 4))
sns.heatmap(stability_matrix.astype(float), annot=True, cmap="YlOrRd", fmt=".3f", ax=ax_sum)
ax_sum.set_title("Explanation Volatility ($ARD$ values)", fontsize=9, fontweight='bold')
ax_sum.set_xlabel("Model", fontsize=8)
ax_sum.set_ylabel("XAI method", fontsize=8)

path_explancao = f'.{bar}output{output_dataset_path}{bar}fig{bar}estabilidade_ard.pdf'
with PdfPages(path_explancao) as pdf:
    pdf.savefig(fig_explancao, bbox_inches='tight')
    plt.close(fig_explancao)

print(f"Mosaico e Heatmap gerados com sucesso com os nomes das variáveis.")