import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# 1. Configurações Iniciais de Caminho
bar = util.bar_system()
output_dataset_path = '' # Ajuste se necessário para '_diabetes', etc.

# 2. Funções de Suporte Otimizadas para Mosaico
def bumpchart(df, ax, color_dic=None):
    """Versão 'Clean' do Bump Chart para visualização científica."""
    left_yaxis = ax
    right_yaxis = left_yaxis.twinx()
    axes = [left_yaxis, right_yaxis]
    
    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Garante o alinhamento dos eixos
        for axis in axes[1:]:
            axis.plot(x, y, alpha=0)

        color = color_dic.get(col, '#7f7f7f') if color_dic else None
        # Estilo de linha mais limpo
        left_yaxis.plot(x, y, linewidth=2.5, alpha=0.7, color=color, solid_capstyle='round')
        left_yaxis.scatter(x, y, s=20, alpha=0.9, color=color)

    lines = len(df.columns)
    y_ticks = [*range(0, lines)]
    
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.2, -1))
    
    # Rótulos laterais simplificados
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index
    left_yaxis.set_yticklabels(left_labels, fontsize=7)
    right_yaxis.set_yticklabels(right_labels, fontsize=7)
    return axes

def plot_into_grid(df_raw, model, test, ax, color_dic):
    """Processa dados e desenha no eixo correspondente da grade."""
    # Cálculo da Correlação de Spearman Consolidada
    original = df_raw.iloc[:, 0]
    sum_corr = 0
    for i in range(1, len(df_raw.columns)):
        rank = list(df_raw.iloc[:, i])
        corr, _ = stats.spearmanr(original, rank)
        sum_corr += corr

    # Transformação de Rankings (Referenciada pela primeira coluna)
    ref_sequence = list(df_raw.iloc[:, 0])
    df_transformed = df_raw.copy()
    for col in df_raw.columns:
        for idx1, val1 in enumerate(ref_sequence):
            for idx2, val2 in enumerate(df_raw[col]):
                if val1 == val2:
                    df_transformed.at[idx1, col] = idx2
    
    df_transformed.index = ref_sequence
    # Simplificação Radical do Eixo X: Apenas as porcentagens
    df_transformed.columns = ['0%', '4%', '6%', '10%']
    
    # Título do Subplot com Soma de Correlações
    ax.set_title(f"{test.upper()} | {model.upper()}\n$\sum c$: {round(sum_corr, 2)}", 
                 fontsize=10, fontweight='bold', pad=10)
    
    bumpchart(df_transformed.transpose(), ax, color_dic)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('Rank', fontsize=8)

# 3. Execução Principal e Geração do PDF
# Carga do DataFrame Original
csv_path = f'.{bar}output{output_dataset_path}{bar}csv{bar}df_explanation_analysis.csv'
df_master = pd.read_csv(csv_path, sep=',', index_col=0)

# Definição Dinâmica de Cores (Paleta de 40 cores para as features)
palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39'
]
features_list = df_master[df_master.columns[1]]
color_dic = {feat: palette[i % len(palette)] for i, feat in enumerate(features_list)}

# Listas de iteração para a grade 6x4
models = ['lgbm', 'knn', 'mlp', 'dt']
tests = ['exirt', 'shap', 'skater','eli5', 'lofo', 'dalex']

# Criação da figura mestre para o mosaico
fig, axes = plt.subplots(len(tests), len(models), figsize=(18, 20), dpi=100)

for r, test in enumerate(tests):
    for c, model in enumerate(models):
        # Montagem dinâmica das colunas conforme seu padrão de nomenclatura
        col_prefix = f"{test}_{model}_x_test_"
        cols = [f"{col_prefix}original", f"{col_prefix}4%_permute", f"{col_prefix}6%_permute", f"{col_prefix}10%_permute"]
        
        try:
            df_subset = df_master[cols]
            plot_into_grid(df_subset, model, test, axes[r, c], color_dic)
        except KeyError:
            # Oculta o gráfico se o dado específico não existir no CSV
            axes[r, c].set_axis_off()

plt.tight_layout(pad=3.0)

# Salvamento Final em PDF de alta qualidade
pdf_save_path = f'.{bar}output{output_dataset_path}{bar}fig{bar}mosaico_comparativo_explicabilidade.pdf'
with PdfPages(pdf_save_path) as pdf:
    pdf.savefig(fig)
    plt.close()

print(f"Mosaico PDF gerado com sucesso em: {pdf_save_path}")