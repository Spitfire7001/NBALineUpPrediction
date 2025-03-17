import os
import glob
import pandas as pd

directory = "./test_data"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, "*.csv"))

for file_path in csv_files:
    df = pd.read_csv(file_path, dtype=str)
    
    # Swap names in team columns
    if "home_team" in df.columns and "away_team" in df.columns:
        df[["home_team", "away_team"]] = df[["home_team", "away_team"]].applymap(
            lambda x: x.replace("SEA", "OKC") if isinstance(x, str) else x
        )
    if "home_team" in df.columns and "away_team" in df.columns:
        df[["home_team", "away_team"]] = df[["home_team", "away_team"]].applymap(
            lambda x: x.replace("NOK", "NOP").replace("NOH", "NOP") if isinstance(x, str) else x
        )
    
    # Swap names in game column
    if "game" in df.columns:
        df["game"] = df["game"].apply(
            lambda x: x[:-3] + x[-3:].replace("SEA", "OKC") if isinstance(x, str) and len(x) >= 3 else x
        )
    if "game" in df.columns:
        df["game"] = df["game"].astype(str).apply(
            lambda x: x[:-3] + (x[-3:].replace("NOK", "NOP").replace("NOH", "NOP")) if len(x) >= 3 else x
        )

    # Save the modified CSV file
    df.to_csv(file_path, index=False)
    print(f"Updated: {os.path.basename(file_path)}")