import os
import pandas as pd

# Folder containing the CSV files
folder_path = './synthetic'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, filename)

        # Load each CSV file
        df = pd.read_csv(file_path)

        # Drop the 'medical_abstract' column and rename 'paraphrased_abstract' to 'medical_abstract'
        if 'medical_abstract' in df.columns and 'paraphrased_abstract' in df.columns:
            df.drop('medical_abstract', axis=1, inplace=True)
            df.rename(
                columns={'paraphrased_abstract': 'medical_abstract'}, inplace=True)

        # Save the modified file (you can change the folder or overwrite the file)
        # Overwrite the original file
        output_file_path = os.path.join(folder_path, filename)
        df.to_csv(output_file_path, index=False)

print("Processing complete for all CSV files.")
