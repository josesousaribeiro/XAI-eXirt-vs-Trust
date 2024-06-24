# Importing libraries
import pandas as pd
import scikit_posthocs as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import util

bar = util.bar_system()

df = pd.read_csv('.'+bar+'output'+bar+'csv'+bar+'df_performance_analysis.csv',sep=',',index_col=0)

print('Dataframe a ser analisado:')
print(df)

df = df[[
         'mlp_x_test_original',
         'mlp_x_test_4%_permute',
         'mlp_x_test_6%_permute',
         'mlp_x_test_10%_permute',         


         'lgbm_x_test_original',
         'lgbm_x_test_4%_permute',
         'lgbm_x_test_6%_permute',
         'lgbm_x_test_10%_permute',      

         'dt_x_test_original',
         'dt_x_test_4%_permute',
         'dt_x_test_6%_permute',
         'dt_x_test_10%_permute',

         'knn_x_test_original',         
         'knn_x_test_4%_permute',
         'knn_x_test_6%_permute',
         'knn_x_test_10%_permute'        
         
         ]]

for col in df.columns:
    df = df.rename(columns={col:col.replace('_x_test_',': ').replace('_permute','')})

columns = df.columns
df_values = df.values.transpose()
 
# Conduct the Friedman Test
stats.friedmanchisquare(*df_values)
 
# Combine three groups into one array
data = np.array([*df_values])
 
# Conduct the Nemenyi post-hoc test
df_matrix = sp.posthoc_nemenyi_friedman(data.T)

df_matrix.columns = columns
df_matrix.index = columns

print(df_matrix)
plt.margins(8)
plt.figure(figsize=(9,6))
ax = sns.heatmap(df_matrix, vmin=0, vmax=1, xticklabels=columns,yticklabels=columns,cmap="Greens",linewidths=.5,annot=True,fmt='.2f', annot_kws={"fontsize":9})
plt.tight_layout()
plt.savefig('.'+bar+'output'+bar+'fig'+bar+'statistical_test.png')