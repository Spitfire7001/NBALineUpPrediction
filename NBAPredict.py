import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize variables
data_list = []
unallowed_data = [
    "end_min", "fga_home", "fta_home", "fgm_home", "fga_2_home", "fgm_2_home", "fga_3_home", "fgm_3_home",
    "ast_home", "blk_home", "pf_home", "reb_home", "dreb_home", "oreb_home", "to_home", "pts_home",
    "pct_home", "pct_2_home", "pct_3_home", "fga_visitor", "fta_visitor", "fgm_visitor",
    "fga_2_visitor", "fgm_2_visitor", "fga_3_visitor", "fgm_3_visitor", "ast_visitor", "blk_visitor",
    "pf_visitor", "reb_visitor", "dreb_visitor", "oreb_visitor", "to_visitor", "pts_visitor",
    "pct_visitor", "pct_2_visitor", "pct_3_visitor"
]
categorical_columns = [
    "game", "home_team", "away_team", "home_0", "home_1", "home_2", "home_3", 
    "away_0", "away_1", "away_2", "away_3", "away_4"
]

csv_files = glob.glob('matchup_data/matchups-*.csv')

# Obtain all starting line ups from csv files
for file in csv_files:
    csv_data = pd.read_csv(file)
    
    csv_data = csv_data[csv_data['starting_min'] == 0]
    
    csv_data = csv_data.drop(columns=unallowed_data, errors='ignore')
    data_list.append(csv_data)

data = pd.concat(data_list, ignore_index=True)

# Apply One-Hot Encoding to data
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
data = pd.concat([data.drop(columns=categorical_columns), encoded_df], axis=1)

# Create training and test data sets
X = data.drop(columns=['home_4'])
y = data['home_4']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier and test and training data
model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_probs = model.predict_proba(X_test)

# Get top 5 predictions
top_5_indices = np.argsort(y_pred_probs, axis=1)[:, -5:][:, ::-1]

# Load player quality scores
quality_scores_df = pd.read_csv('player_quality_scores.csv')
quality_scores_dict = dict(zip(quality_scores_df['player_name'], quality_scores_df['quality_score']))

# Get unique player names
unique_players = np.array(model.classes_)

# Output top 5 players after adjusting their probability
for i in range(len(X_test)):
    print(f"\nGame {i+1} - Predicted Top 5 Players for 5th Starter:")
    for rank, idx in enumerate(top_5_indices[i]):
        player_name = unique_players[idx]
        prob = y_pred_probs[i][idx]
    
        quality_score = quality_scores_dict.get(player_name, 1)
        adjusted_prob = prob * quality_score
        
        print(f"{rank+1}. {player_name} - Adjusted Probability: {adjusted_prob:.3f} (Quality Score: {quality_score})")


# Evaluate and display the model's accuaracy, precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Model Accuracy: {accuracy:.2f}')
print(f"\nModel Statistics:")
print(f"Precision (Weighted): {precision:.3f}")
print(f"Recall (Weighted): {recall:.3f}")
print(f"F1-Score (Weighted): {f1:.3f}")