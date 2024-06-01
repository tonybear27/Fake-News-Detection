import pandas as pd

def split_csv(file_path, output_folder, rows_per_file=200):

    df = pd.read_csv(file_path)
    

    header = df.columns.tolist()
    

    num_files = (len(df) // rows_per_file) + 1

    for i in range(num_files):

        start_row = i * rows_per_file
        end_row = start_row + rows_per_file
        

        split_df = df.iloc[start_row:end_row]
        

        output_file_path = f"{output_folder}/split_file_{i+1}.csv"
        

        split_df.to_csv(output_file_path, index=False)

    print(f"CSV file split into {num_files} smaller files.")

file_path = "./English/training/Fake.csv"
output_folder = "./French/training/split/Fake"
split_csv(file_path, output_folder)
