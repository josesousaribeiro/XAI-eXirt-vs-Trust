from  analysis import *


df = pd.DataFrame()

df['a'] = range(0,20)
df['b'] = range(0,20)



print(df)

df['control'] = False

df_1 = apply_perturbation_in_sample(df.copy(), 0.1, 42)
df_2 = apply_perturbation_in_sample(df_1.copy(), 0.1, 42)
df_3 = apply_perturbation_in_sample(df_2.copy(), 0.1, 42)
print(df_1)
print()
print(df_2)
print()
print(df_3)
