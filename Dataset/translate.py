import os
import pandas as pd
from googletrans import Translator

def translate_text(text, translator, src='en', dest='fr'):
    try:
        translated = translator.translate(text, src=src, dest=dest)
        return translated.text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

def translate_csv(file_path, translator, src='en', dest='fr'):
    df = pd.read_csv(file_path)
    for column in df.columns:
        df[column] = df[column].astype(str).apply(lambda x: translate_text(x, translator, src, dest))
    return df

def translate_files_in_directory(root_dir, translator, src='en', dest='fr'):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                print(f"Translating {file_path}...")
                translated_df = translate_csv(file_path, translator, src, dest)
                output_path = os.path.join(subdir, f"translated_{file}")
                translated_df.to_csv(output_path, index=False)
                print(f"Saved translated file to {output_path}")

translator = Translator()
root_directory = './French'  # Replace with your directory path
translate_files_in_directory(root_directory, translator)
