import pandas as pd
from collections import defaultdict
import glob
import os

# Initialize variables
output_file = 'player_quality_scores.csv'
player_stats = defaultdict(lambda: {'three_pointers': 0, 'two_pointers': 0, 'freethrow_attempts': 0, 'rebounds': 0, 'assists': 0, 'blocked_shots': 0})

csv_files = glob.glob('matchup_data/matchups-*.csv')

# Loop through all CSV files
for file_path in csv_files:
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():

        if row['starting_min'] == 0:

            home_players = [row[f'home_{i}'] for i in range(5)]
            away_players = [row[f'away_{i}'] for i in range(5)]


            three_pointers_home, three_pointers_away = row['fgm_3_home'], row['fgm_3_visitor']
            two_pointers_home, two_pointers_away = row['fgm_2_home'], row['fgm_2_visitor']
            freethrow_attempts_home, freethrow_attempts_away = row['fta_home'], row['fta_visitor']
            rebounds_home, rebounds_away = row['reb_home'], row['reb_visitor']
            assists_home, assists_away = row['ast_home'], row['ast_visitor']
            blocked_shots_home, blocked_shots_away = row['blk_home'], row['blk_visitor']

            points_home, points_away = row['pts_home'], row['pts_visitor']
            ast_home, ast_away = row['ast_home'], row['ast_visitor']
            reb_home, reb_away = row['reb_home'], row['reb_visitor']

            
            for player in home_players:
                player_stats[player]['three_pointers'] += three_pointers_home
                player_stats[player]['two_pointers'] += two_pointers_home
                player_stats[player]['freethrow_attempts'] += freethrow_attempts_home
                player_stats[player]['rebounds'] += rebounds_home
                player_stats[player]['assists'] += assists_home
                player_stats[player]['blocked_shots'] += blocked_shots_home

            for player in away_players:
                player_stats[player]['three_pointers'] += three_pointers_away
                player_stats[player]['two_pointers'] += two_pointers_away
                player_stats[player]['freethrow_attempts'] += freethrow_attempts_away
                player_stats[player]['rebounds'] += rebounds_away
                player_stats[player]['assists'] += assists_away
                player_stats[player]['blocked_shots'] += blocked_shots_away

# Compute quality score for each player
player_scores = []
for player, stats in player_stats.items():
    quality_score = (
        (stats['three_pointers'] * 0.28) + 
        (stats['two_pointers'] * 0.187) + 
        (stats['freethrow_attempts'] * 0.093) +
        (stats['rebounds'] * 0.112) +
        (stats['assists'] * 0.14) +
        (stats['blocked_shots'] * 0.187)
    )
    player_scores.append({'player_name': player, 'quality_score': quality_score})

output_df = pd.DataFrame(player_scores)

# Normalize quality scores (0 - 1)
min_score = output_df['quality_score'].min()
max_score = output_df['quality_score'].max()
    
if min_score != max_score:
    output_df['normalized_score'] = (output_df['quality_score'] - min_score) / (max_score - min_score)
else:
    output_df['normalized_score'] = 1

output_df = output_df.sort_values(by='quality_score', ascending=False)

output_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_file)
output_df.to_csv(output_file_path, index=False)
print(f"Quality scores saved to {output_file_path}")