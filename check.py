from fileinput import filename
import pandas as pd
import os

language = input("Input langiage (French/Spanish):")
label = input("Dataset (True/Fake):")

file_path = f'./{language}/validation/{label}.csv'

dfs = []


df = pd.read_csv(file_path, usecols=['title', 'text'])
        
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip().astype(bool)]
        
dfs.append(df)

checked = pd.concat(dfs, ignore_index=True)

checked.to_csv(file_path, index=False)
