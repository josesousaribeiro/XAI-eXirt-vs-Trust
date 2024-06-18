from  analysis import *


df = pd.DataFrame()

df['a'] = range(0,230)
df['b'] = range(0,230)
df['c'] = range(0,230)
df['d'] = range(0,230)

df['e'] = range(0,230)
df['f'] = range(0,230)
df['g'] = range(0,230)
df['h'] = range(0,230)

print(df)

df_new = apply_perturbation_permute(df.copy(), 0.2, 10)

print(df_new)
