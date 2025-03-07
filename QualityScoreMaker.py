import pandas as pd
from collections import defaultdict
import glob
import os

# Initialize variables
output_file = 'output.csv'
player_stats = defaultdict(lambda: {'points': 0, 'assists': 0, 'rebounds': 0, 'games': 0})

csv_files = glob.glob('matchup_data/matchups-*.csv')

# Loop through all CSV files
for file_path in csv_files:
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():

        if row['starting_min'] == 0:

            home_players = [row[f'home_{i}'] for i in range(5)]
            away_players = [row[f'away_{i}'] for i in range(5)]

            points_home, points_away = row['pts_home'], row['pts_visitor']
            ast_home, ast_away = row['ast_home'], row['ast_visitor']
            reb_home, reb_away = row['reb_home'], row['reb_visitor']

            
            for player in home_players:
                player_stats[player]['points'] += points_home
                player_stats[player]['assists'] += ast_home
                player_stats[player]['rebounds'] += reb_home
                player_stats[player]['games'] += 1

            for player in away_players:
                player_stats[player]['points'] += points_away
                player_stats[player]['assists'] += ast_away
                player_stats[player]['rebounds'] += reb_away
                player_stats[player]['games'] += 1

# Compute quality score for each player
player_scores = []
for player, stats in player_stats.items():
    if stats['games'] > 0:
        quality_score = (
            (stats['points'] * 0.5) + 
            (stats['assists'] * 0.3) + 
            (stats['rebounds'] * 0.2)
        ) / stats['games']
        player_scores.append({'player_name': player, 'quality_score': quality_score})


output_df = pd.DataFrame(player_scores)

output_df = output_df.sort_values(by='quality_score', ascending=False)

output_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_file)
output_df.to_csv(output_file_path, index=False)
print(f"Quality scores saved to {output_file_path}")