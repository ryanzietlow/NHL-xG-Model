# NHL-xG-Model
Overview

This repository provides a Python-based implementation of an Expected Goals (xG) model for hockey. The model predicts the likelihood of a goal occurring based on various shot features and game conditions. It utilizes logistic regression to evaluate shot events and compare predicted xG with actual outcomes. The model is designed to be a robust tool for analyzing and understanding shot effectiveness in hockey.

How It Works

Data Loading and Preparation

	1.	Loading Data:
	•	The data for the 2021, 2022, and 2023 seasons are loaded from CSV files. The data includes various features such as shot type, shot distance, angle, and team strengths.
	2.	Data Filtering:
	•	Data is filtered to include only relevant events (GOAL, SHOT, MISS) and periods (1 through 4) to ensure consistency in analysis.
	3.	Feature Engineering:
	•	New columns are created:
	•	is_goal: Binary indicator for whether the shot resulted in a goal.
	•	is_home: Binary indicator for the home team.
	•	The shotType column is converted into dummy variables to account for different shot types.
	•	Combined skater counts are used to determine the strength of the shooting and defending teams, which is then categorized into shooter_strength.
	4.	Preprocessing:
	•	Categorical and boolean columns are converted to dummy variables.
	•	Missing values are handled, and all necessary features are prepared for modeling.

Model Training and Evaluation

	1.	Feature Selection:
	•	A set of relevant features is selected for training the logistic regression model, including shot characteristics and team strengths.
	2.	Splitting Data:
	•	The dataset is divided into training and test sets to evaluate the model’s performance on unseen data.
	3.	Standardization:
	•	Features are standardized to ensure uniform scaling and improve model performance.
	4.	Training:
	•	A logistic regression model is trained with the standardized features. Hyperparameters such as max_iter and C are set to optimize the model.
	5.	Evaluation:
	•	Predictions are made on the test set, and various performance metrics are computed:
	•	Accuracy: Proportion of correctly classified shots.
	•	Precision: Ratio of true positives to all predicted positives.
	•	F1 Score: Harmonic mean of precision and recall.
	•	Log Loss: Measures the performance of the model based on predicted probabilities.
	•	ROC AUC: Area under the ROC curve indicating model discriminative ability.
	•	The model’s performance is compared with other well-established models:
	•	Moneypuck Model: R² value of 0.93
	•	Natural Stat Trick Model: R² value of 0.94

Visualization

	1.	ROC Curve:
	•	The ROC curve is plotted to visualize the model’s performance across different classification thresholds.
	2.	Scatter Plots:
	•	Scatter plots of predicted xG vs. various expected ixG metrics (e.g., ixG NSS, ixG MP) with a line of best fit are generated to analyze the relationship between predicted and expected metrics.
