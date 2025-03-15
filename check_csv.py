import pandas as pd

df = pd.read_csv("subtitles_raw.csv")

# Check if 'file_content' column exists
if 'file_content' not in df.columns:
    print("❌ Error: 'file_content' column is missing in the dataset.")
else:
    print("✅ 'file_content' column found!")
    print(f"Number of missing values in 'file_content': {df['file_content'].isna().sum()}")
    print("Sample data:")
    print(df[['file_content']].head())  # Display first few values
