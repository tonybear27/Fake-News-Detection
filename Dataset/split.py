import pandas as pd
import os

df_True = pd.read_csv('./True.csv')
df_Fake = pd.read_csv('./Fake.csv')

Real = df_True.sample(n = 1000, random_state = 42)
Fake = df_Fake.sample(n = 1000, random_state = 42)

testing_True = Real[:500]
validatiob_True = Real[500:]

testing_Fake = Fake[:500]
validation_Fake = Fake[500:]

df_remaining_True = df_True.drop(Real.index)
df_remaining_Fake = df_Fake.drop(Fake.index)

testing_True.to_csv('./testing/True.csv', index=False)
testing_Fake.to_csv('./testing/Fake.csv', index=False)
validatiob_True.to_csv('./validation/True.csv', index=False)
validation_Fake.to_csv('./validation/Fake.csv', index=False)

df_remaining_True.to_csv('True.csv', index=False)
df_remaining_Fake.to_csv('Fake.csv', index=False)