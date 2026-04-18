import pandas as pd
import scikit_posthocs as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import util

# 1. Configuração de caminhos e sistema
bar = util.bar_system()
output_dataset_path = ''  # Ajuste conforme necessário (ex: '_diabetes')
#output_dataset_path = '_banknote_authentication'
#output_dataset_path = '_diabetes'
#output_dataset_path = '_phoneme'
#output_dataset_path = '_mozilla4'


# 2. Carga dos dados
file_path = f'.{bar}output{output_dataset_path}{bar}csv{bar}df_performance_analysis.csv'
df = pd.read_csv(file_path, sep=',', index_col=0)

# 3. Seleção e preparação das colunas
# Mantendo a ordem lógica: Original -> 4% -> 6% -> 10% para cada modelo
ordered_columns = [
    'mlp_x_test_original', 'mlp_x_test_4%_permute', 'mlp_x_test_6%_permute', 'mlp_x_test_10%_permute',
    'lgbm_x_test_original', 'lgbm_x_test_4%_permute', 'lgbm_x_test_6%_permute', 'lgbm_x_test_10%_permute',
    'dt_x_test_original', 'dt_x_test_4%_permute', 'dt_x_test_6%_permute', 'dt_x_test_10%_permute',
    'knn_x_test_original', 'knn_x_test_4%_permute', 'knn_x_test_6%_permute', 'knn_x_test_10%_permute'
]
df = df[ordered_columns]

# Renomeação para exibição limpa no gráfico
for col in df.columns:
    df = df.rename(columns={col: col.replace('_x_test_', ': ').replace('_permute', '')})

columns = df.columns

# 4. Testes Estatísticos
# Friedman
stats.friedmanchisquare(*df.values.T)

# Nemenyi Post-hoc (Matriz de p-values)
df_matrix = sp.posthoc_nemenyi_friedman(df.values)
df_matrix.columns = columns
df_matrix.index = columns

# 5. Visualização Científica Aprimorada
plt.figure(figsize=(12, 9))

# Máscara para ocultar o triângulo superior (evita redundância visual)
mask = np.triu(np.ones_like(df_matrix, dtype=bool))

# Mapa de cores: RdYlGn (Vermelho-Amarelo-Verde) 
# O parâmetro 'center=0.05' faz com que a transição de cor ocorra no limiar de significância
ax = sns.heatmap(df_matrix, 
                 mask=mask,
                 vmin=0, vmax=1, center=0.05, 
                 cmap="RdYlGn", 
                 linewidths=.5, 
                 annot=True, fmt='.2f', 
                 annot_kws={"fontsize": 8, "weight": "bold"},
                 cbar_kws={'label': 'p-value significance'})

# Adicionando linhas de grade pretas para separar os blocos de cada modelo (4 em 4)
for i in range(0, len(columns) + 1, 4):
    ax.axhline(i, color='black', lw=2)
    ax.axvline(i, color='black', lw=2)

plt.title('Nemenyi Post-hoc Test: Performance Stability Analysis\n(p-value < 0.05 indicates statistical difference)', 
          fontsize=14, pad=20)
plt.tight_layout()

# Salvamento
save_path = f'.{bar}output{output_dataset_path}{bar}fig{bar}statistical_test_v2.png'
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Análise concluída. Gráfico salvo em: {save_path}")