import os
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator
import time


tqdm.pandas()

def split_text(text, max_length=4500):
    """Split text into chunks of max_length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def translate_text(text, translator):
    if not text or pd.isna(text):
        return text
    
    if len(text) <= 4500:
        try:
            return translator.translate(text)
        except Exception as e:
            print(f"Error translating text: {e}")
            return None  # Return None if translation fails
    else:
        chunks = split_text(text)
        translated_chunks = []
        for chunk in chunks:
            try:
                translated_chunks.append(translator.translate(chunk))
                time.sleep(1)  # Add a slight delay to avoid rate limits
            except Exception as e:
                print(f"Error translating chunk: {e}")
                return None  # Return None if any chunk translation fails
        return ''.join(translated_chunks)

# Set the global language variable
global lan
language = input("Enter the target language (French/Spanish/Chinese): ")
label = input("Enter which dataset (True/Fake):")

if language == "French": 
    lan = 'fr'
elif language == "Spanish": 
    lan = 'es'
else:
    print("Unsupported language. Please enter 'French' or 'Spanish'.")
    exit(1)

translator = GoogleTranslator(source='en', target=lan)

root = f'./{language}/training/split/{label}/'
files = os.listdir(root)

for file in files:
    print(f"Translating {root + file}...")

    try:
        df = pd.read_csv(root + file)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        continue

    df_translated = df.copy()

    for column in ['title', 'text']:
        if column in df.columns:
            try:
                df_translated[column] = df[column].progress_apply(lambda x: translate_text(x, translator))
                df_translated = df_translated[df_translated[column].notna()]
            except Exception as e:
                print(f"Error translating column {column} in file {file}: {e}")
                continue

    output_path = root + file
    try:
        df_translated.to_csv(output_path, index=False)
        print(f"Saved translated file to {output_path}")
    except Exception as e:
        print(f"Error saving translated file {file}: {e}")
