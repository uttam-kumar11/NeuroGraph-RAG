import pandas as pd
import os

def clean_disease_symptoms():
    # Define exact paths
    input_file = r"D:\NeuroGraph_Project\data\dataset.csv"
    output_file = r"D:\NeuroGraph_Project\data\cleaned_disease_symptoms.csv"

    print("Loading raw dataset...")
    df = pd.read_csv(input_file)

    #  Convert from wide to long format
    # This keeps 'Disease' and stacks all Symptom 1 to Symptom 17 columns into one column
    df_long = df.melt(id_vars=['Disease'], value_name='Symptom')

    # Drop the old column headers (we don't need the text "Symptom 1", etc.)
    df_long = df_long.drop(columns=['variable'])

    # Drop empty rows (NaNs) where a disease had fewer than 17 symptoms
    df_cleaned = df_long.dropna()

    #  Clean text: remove trailing spaces and make everything lowercase
    # The Kaggle dataset is notorious for hidden spaces like ' fatigue'
    df_cleaned['Disease'] = df_cleaned['Disease'].str.strip().str.lower()
    df_cleaned['Symptom'] = df_cleaned['Symptom'].str.strip().str.lower()

    #  Remove duplicate relationships to keep the graph lightweight
    df_cleaned = df_cleaned.drop_duplicates()

    # Save to a new CSV for Neo4j injection
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"Success! Cleaned data saved to: {output_file}")
    print(f"Total unique (Disease -> Symptom) connections ready for Relationship Representation: {len(df_cleaned)}")

if __name__ == "__main__":
    clean_disease_symptoms()