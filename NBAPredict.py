import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random

data_list = []
unallowed_data = [
    "game", "end_min", "fga_home", "fta_home", "fgm_home", "fga_2_home", "fgm_2_home", "fga_3_home", "fgm_3_home",
    "ast_home", "blk_home", "pf_home", "reb_home", "dreb_home", "oreb_home", "to_home", "pts_home",
    "pct_home", "pct_2_home", "pct_3_home", "fga_visitor", "fta_visitor", "fgm_visitor",
    "fga_2_visitor", "fgm_2_visitor", "fga_3_visitor", "fgm_3_visitor", "ast_visitor", "blk_visitor",
    "pf_visitor", "reb_visitor", "dreb_visitor", "oreb_visitor", "to_visitor", "pts_visitor",
    "pct_visitor", "pct_2_visitor", "pct_3_visitor", "outcome"
]
team_columns = ["home_team", "away_team"]
player_columns = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 
                  'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 
                  'removed_player']

# Obtain all starting lineups used for training
csv_files = glob.glob('matchup_data/matchups-*.csv')
data_list = []

for file in csv_files:
    csv_data = pd.read_csv(file)
    csv_data = csv_data[csv_data['outcome'] == 1]
    csv_data = csv_data.drop(columns=unallowed_data, errors='ignore')
    data_list.append(csv_data)

training_data = pd.concat(data_list, ignore_index=True)

# Column to know which player was randomly removed
training_data['removed_player'] = None

# Remove one player randomly from home team
for index, row in training_data.iterrows():
    random_player = random.choice(player_columns[:5])
    training_data.at[index, 'removed_player'] = row[random_player]
    training_data.at[index, random_player] = "?"

# Obtain test data
test_data = pd.read_csv("test_data/NBA_test.csv")
removed_players_test = pd.read_csv("test_data/NBA_test_labels.csv")
test_data["removed_player"] = removed_players_test["removed_value"]

# Encode training and test data
data = pd.concat([training_data, test_data], ignore_index=True)

team_encoder = LabelEncoder()
for team_col in team_columns:
    data[team_col] = team_encoder.fit_transform(data[team_col])

player_encoder = LabelEncoder()
all_players_data = data[player_columns].values.flatten()
all_players_data = np.append(all_players_data, "?")
player_encoder.fit(all_players_data)

for player_column in player_columns:
    data[player_column] = player_encoder.transform(data[player_column])

# Seperate training and test data
training_data = data.iloc[:len(training_data)]
testing_data = data.iloc[len(training_data):]

training_data_X = training_data.drop(columns=['removed_player'], errors='ignore')
training_data_Y = training_data['removed_player']

testing_data_X = testing_data.drop(columns=['removed_player'], errors='ignore')
testing_data_Y = testing_data['removed_player']

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
model.fit(training_data_X, training_data_Y)
y_pred = model.predict(testing_data_X)
y_pred_probs = model.predict_proba(testing_data_X)

# Decode the predicted values
decoded_y_pred = player_encoder.inverse_transform(y_pred)

# Print out results for each game
for i, game in enumerate(testing_data.itertuples()):
    home_team = game.home_team
    removed_player = decoded_y_pred[i]
    print(f"Game {i + 1}:")
    print(f"  Home Team: {test_data['home_team'][i]}")
    print(f"  Predicted Removed Player: {removed_player}")
    print("-" * 40)


# Evaluate the model using accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_pred, testing_data_Y)
precision = precision_score(y_pred, testing_data_Y, average='weighted')
recall = recall_score(y_pred, testing_data_Y, average='weighted')
f1 = f1_score(y_pred, testing_data_Y, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")