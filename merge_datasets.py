import pandas as pd
from pathlib import Path

# ==========================================
# Folders
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parent
QA_FOLDER = PROJECT_FOLDER / "qa"

OUTPUT_FILE = QA_FOLDER / "merged_dataset.csv"

# ==========================================
# Delete old merged dataset
# ==========================================

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

# ==========================================
# Read all dataset files
# ==========================================

all_dataframes = []

for file in QA_FOLDER.glob("קובץ *.csv"):

    print(f"Reading: {file.name}")

    df = pd.read_csv(file)

    all_dataframes.append(df)

# ==========================================
# Merge datasets
# ==========================================

merged_df = pd.concat(
    all_dataframes,
    ignore_index=True
)

# Remove duplicate rows
merged_df = merged_df.drop_duplicates()

# ==========================================
# Save merged dataset
# ==========================================

merged_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print(f"Done! Merged dataset saved to:")
print(OUTPUT_FILE)
print(f"\nTotal rows: {len(merged_df)}")