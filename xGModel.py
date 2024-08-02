import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, roc_curve, auc, r2_score
import matplotlib.pyplot as plt

# Load data
PbP2021Data = pd.read_csv('/Users/Ryan/Downloads/xG Model/shots_2021.csv')
PbP2022Data = pd.read_csv('/Users/Ryan/Downloads/xG Model/shots_2022.csv')
PbP2023Data = pd.read_csv('/Users/Ryan/Downloads/xG Model/shots_2023.csv')
# Combine datasets
PbPCombinedData = pd.concat([PbP2021Data, PbP2022Data])

# Define events of interest
fenwick_events = ['GOAL', 'SHOT', 'MISS']

# Filter data
filtered_data = PbP2023Data[(PbP2023Data['event'].isin(fenwick_events)) & (PbP2023Data['period'] < 5)].copy()

# Add 'is_goal' and 'is_home' columns
def is_goal(dataframe):
    dataframe['is_goal'] = (dataframe['event'] == 'GOAL').astype(int)
    return dataframe

def is_home(dataframe, column_name='isHomeTeam'):
    dataframe['is_home'] = dataframe[column_name].astype(int)
    return dataframe

filtered_data = is_goal(filtered_data)
filtered_data = is_home(filtered_data, column_name='isHomeTeam')
filtered_data = filtered_data.sort_values(by=['game_id', 'id'])
filtered_data['is_home'] = filtered_data['is_home'].fillna(0)

# Drop missing values and process 'shotType'
filtered_data = filtered_data.dropna()
filtered_data['shotType'] = filtered_data['shotType'].astype('category')

# Convert boolean columns to integers
filtered_data = pd.get_dummies(filtered_data, columns=['shotType'])
boolean_columns = filtered_data.select_dtypes(include=['bool']).columns
filtered_data[boolean_columns] = filtered_data[boolean_columns].astype(int)

# Combine skaters on ice
filtered_data['shooting_team_total'] = filtered_data['shootingTeamForwardsOnIce'] + filtered_data['shootingTeamDefencemenOnIce']
filtered_data['defending_team_total'] = filtered_data['defendingTeamForwardsOnIce'] + filtered_data['defendingTeamDefencemenOnIce']

# Define conditions based on combined totals
conditions = [
    (filtered_data['shooting_team_total'] > filtered_data['defending_team_total']),
    (filtered_data['shooting_team_total'] < filtered_data['defending_team_total']),
    (filtered_data['shooting_team_total'] == filtered_data['defending_team_total'])
]
values = ['power_play', 'penalty_kill', 'even_strength']

# Create 'shooter_strength' column
filtered_data['shooter_strength'] = np.select(conditions, values, default='unknown')

# Convert 'shooter_strength' to dummy variables
filtered_data = pd.get_dummies(filtered_data, columns=['shooter_strength'])
boolean_columns = filtered_data.select_dtypes(include=['bool']).columns
filtered_data[boolean_columns] = filtered_data[boolean_columns].astype(int)

# Define the columns to use for training
selected_columns = [
    'shotAngle',
    'shotDistance',
    'xCord',
    'yCord',
    'shotAnglePlusRebound',
    'shotAnglePlusReboundSpeed',
    'shotRush',
    'shotRebound',
    'shotType_BACK',
    'shotType_DEFL',
    'shotType_SLAP',
    'shotType_SNAP',
    'shotType_TIP',
    'shotType_WRAP',
    'shotType_WRIST',
    'shooter_strength_even_strength',
    'shooter_strength_penalty_kill',
    'shooter_strength_power_play',
    'homeTeamGoals',
    'awayTeamGoals',
    'lastEventxCord',
    'lastEventyCord',
    'distanceFromLastEvent',
    'lastEventShotDistance'
]

# Define feature columns and target column
features = filtered_data[selected_columns]
target = filtered_data['is_goal']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000, C=0.5, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Predict probabilities
filtered_data_scaled = scaler.transform(features)
predicted_probabilities = model.predict_proba(filtered_data_scaled)[:, 1]

# Add predicted probabilities to the original DataFrame
filtered_data['xG'] = predicted_probabilities

# Aggregate probabilities by player
player_xg1 = filtered_data.groupby('shooterName')['xG'].sum().reset_index()
player_xg2 = filtered_data.groupby('shooterName')['xG'].sum().reset_index()

# Load the comparison data
comparison_data1 = pd.read_csv('/Users/Ryan/Downloads/xG Model/Natural Stat Trick Data 2023-24.csv')
comparison_data2 = pd.read_csv('/Users/Ryan/Downloads/xG Model/Money Puck Data 2023-24.csv')

# Merge the new xG results with the comparison data on a common identifier
xG_vs_ixG_NSS = pd.merge(player_xg1, comparison_data1, on='shooterName', how='inner')

# Merge the dataframes
xG_vs_ixG_MP = pd.merge(player_xg2, comparison_data2, on='shooterName', how='inner')

# Drop rows with any missing values
xG_vs_ixG_MP = xG_vs_ixG_MP.dropna()

# Sort by 'shooterName' and 'ixG MP' in descending order
xG_vs_ixG_MP = xG_vs_ixG_MP.sort_values(by=['shooterName', 'ixG MP'], ascending=[True, False])

# Drop duplicates, keeping the first occurrence (highest 'ixG MP')
xG_vs_ixG_MP = xG_vs_ixG_MP.drop_duplicates(subset='shooterName', keep='first')

# Calculate R² score
r2 = r2_score(xG_vs_ixG_NSS['xG'], xG_vs_ixG_NSS['ixG NSS'])
print(f"R² Score: {r2:.4f}")
r2 = r2_score(xG_vs_ixG_MP['xG'], xG_vs_ixG_MP['ixG MP'])
print(f"R² Score: {r2:.4f}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

# Create scatter plot of ixG vs xG with different shades
plt.figure(figsize=(10, 6))
plt.scatter(xG_vs_ixG_NSS['xG'], xG_vs_ixG_NSS['ixG NSS'], c='blue', alpha=0.7, edgecolors='w', s=100, label='Player ixG NSS vs xG')

# Fit line of best fit
coefficients = np.polyfit(xG_vs_ixG_NSS['xG'], xG_vs_ixG_NSS['ixG NSS'], 1)
poly = np.poly1d(coefficients)
x_range = np.linspace(xG_vs_ixG_NSS['xG'].min(), xG_vs_ixG_NSS['xG'].max(), 100)
plt.plot(x_range, poly(x_range), color='red', linestyle='--', label='Line of Best Fit')

plt.xlabel('Predicted xG')
plt.ylabel('Expected ixG NSS')
plt.title('Scatter Plot of ixG NSS vs xG by Player')
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Create scatter plot of ixG MP vs xG with different shades
plt.figure(figsize=(10, 6))
plt.scatter(xG_vs_ixG_MP['xG'], xG_vs_ixG_MP['ixG MP'], c='green', alpha=0.7, edgecolors='w', s=100, label='Player ixG MP vs xG')

# Fit line of best fit
coefficients = np.polyfit(xG_vs_ixG_MP['xG'], xG_vs_ixG_MP['ixG MP'], 1)
poly = np.poly1d(coefficients)
x_range = np.linspace(xG_vs_ixG_MP['xG'].min(), xG_vs_ixG_MP['xG'].max(), 100)
plt.plot(x_range, poly(x_range), color='red', linestyle='--', label='Line of Best Fit')

plt.xlabel('Predicted xG')
plt.ylabel('Expected ixG MP')
plt.title('Scatter Plot of ixG MP vs xG by Player')
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Diagonal line (no-skill)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()