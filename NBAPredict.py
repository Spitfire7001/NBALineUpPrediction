import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

team_columns = ["home_team", "away_team"]
player_columns = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 
                  'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 
                  'removed_player']

train_dir = "training_data"
test_dir = "test_data"

train_files = sorted(glob.glob(os.path.join(train_dir, "home_*_removed.csv")))
test_files = sorted(glob.glob(os.path.join(test_dir, "test_home_*_removed.csv")))

label_encoders = {}

results = []

for train_file, test_file in zip(train_files, test_files):
    
    # Load datasets
    training_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

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

    # Separate training and test data
    training_data = data.iloc[:len(training_data)]
    testing_data = data.iloc[len(training_data):]

    training_data_X = training_data.drop(columns=['removed_player'], errors='ignore')
    training_data_Y = training_data['removed_player']

    testing_data_X = testing_data.drop(columns=['removed_player'], errors='ignore')
    testing_data_Y = testing_data['removed_player']

    print(f"Begin training on {train_file}")

    # Train and evaluate the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=3)
    model.fit(training_data_X, training_data_Y)
    y_pred = model.predict(testing_data_X)
    accuracy = accuracy_score(testing_data_Y, y_pred)
    
    print(f"Trained on {train_file}, tested on {test_file} - Accuracy: {accuracy:.2f}")

    # Collect results
    for i, row in testing_data.iterrows():

        encoded_home_team = row['home_team']
        removed_player_encoded = row['removed_player']
        predicted_player_encoded = y_pred[i - len(training_data)]

        home_team = team_encoder.inverse_transform([encoded_home_team])[0]
        removed_player = player_encoder.inverse_transform([removed_player_encoded])[0]
        predicted_player = player_encoder.inverse_transform([predicted_player_encoded])[0]

        season = row['season']

        results.append({
            'home_team': home_team,
            'removed_player': removed_player,
            'predicted_player': predicted_player,
            'season': season
        })

# Store results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='season')
results_df.to_csv('results.csv', index=False)

print("Results saved to 'results.csv'")
