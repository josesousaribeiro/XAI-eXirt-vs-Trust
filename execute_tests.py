from  analysis import *


df = pd.DataFrame()

df['col1'] = range(0,10)
df['col2'] = range(0,10)
df['col3'] = range(0,10)
df['col4'] = range(0,10)
df['col5'] = range(0,10)
df['col6'] = range(0,10)
df['col7'] = range(0,10)
df['col8'] = range(0,10)
df['col9'] = range(0,10)
df['col10'] = range(0,10)

print(df)

df_new = apply_perturbation_permute(df.copy(), 0.1, 10)

print(df_new)
