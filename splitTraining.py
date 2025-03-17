import pandas as pd
import glob

unallowed_data = [
    "game", "end_min", "fga_home", "fta_home", "fgm_home", "fga_2_home", "fgm_2_home", "fga_3_home", "fgm_3_home",
    "ast_home", "blk_home", "pf_home", "reb_home", "dreb_home", "oreb_home", "to_home", "pts_home",
    "pct_home", "pct_2_home", "pct_3_home", "fga_visitor", "fta_visitor", "fgm_visitor",
    "fga_2_visitor", "fgm_2_visitor", "fga_3_visitor", "fgm_3_visitor", "ast_visitor", "blk_visitor",
    "pf_visitor", "reb_visitor", "dreb_visitor", "oreb_visitor", "to_visitor", "pts_visitor",
    "pct_visitor", "pct_2_visitor", "pct_3_visitor", "outcome"
]
team_columns = ["home_team", "away_team"]
home_positions = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']

# Obtain test data
test_data = pd.read_csv("test_data/NBA_test.csv")
removed_players_test = pd.read_csv("test_data/NBA_test_labels.csv")
test_data["removed_player"] = removed_players_test["removed_value"]

# Process and save test data
for col in home_positions:
    file_name = f"test_{col}_removed.csv"
    subset = test_data[test_data[col] == '?']
    subset.to_csv(file_name, index=False)

print("Test CSV files successfully created!")

# Load training data
csv_files = glob.glob('matchup_data/matchups-*.csv')
data_list = []

for file in csv_files:
    csv_data = pd.read_csv(file)
    csv_data = csv_data[csv_data['outcome'] == 1]
    csv_data = csv_data.drop(columns=unallowed_data, errors='ignore')
    data_list.append(csv_data)

training_data = pd.concat(data_list, ignore_index=True)

# For each row of training data, remove the player in each home position and save the removed player
expanded_rows = []

for _, row in training_data.iterrows():
    for col in home_positions:
        new_row = row.copy()
        new_row['removed_player'] = new_row[col]
        new_row[col] = '?'
        expanded_rows.append(new_row)

training_data = pd.DataFrame(expanded_rows)

# Save the processed data into separate CSV files based on which player was removed
for col in home_positions:
    file_name = f"{col}_removed.csv"
    subset = training_data[training_data[col] == '?']
    subset.to_csv(file_name, index=False)

print("Training CSV files successfully created!")
