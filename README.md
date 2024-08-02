# NHL-xG-Model

## Overview

This repository contains a Python-based implementation of an Expected Goals (xG) model for hockey. The model predicts the likelihood of a goal occurring based on various shot features and game conditions. Using logistic regression, it evaluates shot events and compares predicted xG with actual outcomes. This tool provides insights into player and team performance by analyzing shot effectiveness in hockey.

Here is how the model compares to highly regarded public models MoneyPuck and Natural Stat Trick for the 2023-24 season:

<img width="309" alt="Screenshot 2024-08-01 at 9 49 03 PM" src="https://github.com/user-attachments/assets/c15b90e7-bd39-4eff-88ac-25b8ee698f16">
<img width="307" alt="Screenshot 2024-08-01 at 9 51 52 PM" src="https://github.com/user-attachments/assets/02671eee-dca8-44f9-82c4-e773b600eae9">

## How It Works

## Data Loading and Preparation

	1. Loading Data:
		• Data for the 2021, 2022, and 2023 NHL seasons are loaded from CSV files. Features include shot type, shot distance, angle, and team strengths.
	2. Data Filtering:
		• The dataset is filtered to include relevant events (GOAL, SHOT, MISS) and periods (1 through 4) to ensure consistency in analysis.
	3. Feature Engineering:
		• New columns are created:
			• is_goal: Binary indicator for whether the shot resulted in a goal.
			• is_home: Binary indicator for the home team.
		• The shotType column is converted into dummy variables to account for different shot types.
		• Combined skater counts are used to determine the strength of the shooting and defending teams, which is then categorized into shooter_strength.
	4. Preprocessing:
		• Categorical and boolean columns are converted to dummy variables.
		• Missing values are handled, and all necessary features are prepared for modeling.

## Model Training and Evaluation

	1. Feature Selection:
		• Relevant features are selected for training the logistic regression model, including shot characteristics and team strengths.
	2. Splitting Data:
		• The dataset is divided into training and test sets to evaluate the model’s performance on unseen data.
	3. Standardization:
		• Features are standardized to ensure uniform scaling and improve model performance.
	4. Training:
		• A logistic regression model is trained with the standardized features. Hyperparameters such as max_iter and C are optimized for the model.
	5. Evaluation:
		• ROC AUC: The ROC AUC score for the model is 0.73, indicating the model’s ability to discriminate between goal and non-goal events. A higher ROC AUC value signifies better model performance in distinguishing between the two classes.
		• R² Values: The model’s R² value is compared to other established models:
		• Moneypuck Model: R² value of 0.93
		• Natural Stat Trick Model: R² value of 0.94

<img width="906" alt="Screenshot 2024-08-01 at 8 48 39 PM" src="https://github.com/user-attachments/assets/f48bb1a9-2c2e-4464-82fa-a602df2cb759">
<img width="143" alt="Screenshot 2024-08-01 at 8 49 09 PM" src="https://github.com/user-attachments/assets/a9cdebbe-6c73-48f1-b66b-1b53905b50ac">

## Visualization

	1. ROC Curve:
		• The ROC curve is plotted to visualize the model’s performance across different classification thresholds. This helps in understanding how well the model discriminates between the classes.
	2. Scatter Plots:
		• Scatter plots of predicted xG vs. various expected ixG metrics (e.g., ixG NSS, ixG MP) are generated with a line of best fit to analyze the relationship between predicted and expected metrics.

<img width="929" alt="Screenshot 2024-08-01 at 8 48 22 PM" src="https://github.com/user-attachments/assets/f0331596-2200-48b1-ade4-6778d1732121">
<img width="911" alt="Screenshot 2024-08-01 at 8 48 31 PM" src="https://github.com/user-attachments/assets/d51a920b-24a5-4cc0-8e96-485b4928b581">

## Sources

I would like to give a massive thank you to MoneyPuck, MatthewBarlowe and EvolvingHockey. Their public work and data greatly assisted me throughout this project.

