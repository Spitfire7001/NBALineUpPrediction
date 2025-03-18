import pandas as pd

csv_file = "results.csv"
df = pd.read_csv(csv_file)

df['match'] = df['removed_player'] == df['predicted_player']

match_counts = df.groupby('season')['match'].sum()

print(match_counts)