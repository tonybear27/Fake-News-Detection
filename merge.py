from fileinput import filename
import pandas as pd
import os

language = input("Input langiage (French/Spanish):")
label = input("Dataset (True/Fake):")

root = f'./{language}/training/split/{label}/'

merged = []

for file in os.listdir(root):
    if file.endswith('.csv'):
        path = os.path.join(root, file)
        df = pd.read_csv(path, usecols=['title', 'text'])

        merged.append(df)
merged = pd.concat(merged, ignore_index=True)
merged.to_csv(root + f'{label}.csv', index=False)