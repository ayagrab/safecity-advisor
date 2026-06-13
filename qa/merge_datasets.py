import pandas as pd
from pathlib import Path

# ==========================================
# Folders
# ==========================================

QA_FOLDER = Path(__file__).resolve().parent

OUTPUT_FULL = QA_FOLDER / "merged_dataset.csv"
OUTPUT_CLEAN = QA_FOLDER / "merged_dataset_clean.csv"

# ==========================================
# Delete old merged files (if they exist)
# ==========================================

for file in [OUTPUT_FULL, OUTPUT_CLEAN]:
    if file.exists():
        file.unlink()

# ==========================================
# Read all CSV files
# ==========================================

all_dataframes = []

for file in QA_FOLDER.glob("*.csv"):

    print(f"Reading: {file.name}")

    df = pd.read_csv(file)

    all_dataframes.append(df)

# ==========================================
# Merge everything
# ==========================================

merged_df = pd.concat(
    all_dataframes,
    ignore_index=True
)

merged_df = merged_df.drop_duplicates()

# ==========================================
# Save full dataset
# ==========================================

merged_df.to_csv(
    OUTPUT_FULL,
    index=False
)

# ==========================================
# Save clean dataset
# ==========================================

clean_df = merged_df.iloc[:, :2]

clean_df.to_csv(
    OUTPUT_CLEAN,
    index=False
)

print("Done!")