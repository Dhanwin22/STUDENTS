def read_dataset_from_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset


csv_filename = "dataset.csv"

import csv


def read_dataset_from_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset

# Function to calculate percentage of correct responses in past history
def calculate_past_performance(history):
    total_responses = len(history)
    correct_responses = history.count('1')
    if total_responses == 0:
        return 0  # Avoid division by zero
    return correct_responses / total_responses


def update_dataset_with_past_performance(dataset):
    updated_dataset = []
    for entry in dataset:
        history = entry[3]
        past_performance = calculate_past_performance(history)
        updated_entry = entry + [past_performance]
        updated_dataset.append(updated_entry)
    return updated_dataset


csv_filename = "dataset.csv"


dataset = read_dataset_from_csv(csv_filename)


updated_dataset = update_dataset_with_past_performance(dataset)

# Print the updated dataset
for entry in updated_dataset:
    print(entry)

# Function to calculate percentage of correct responses in past history
def calculate_past_performance(history):
    total_responses = len(history)
    correct_responses = history.count('1')
    if total_responses == 0:
        return 0  # Avoid division by zero
    return correct_responses / total_responses

# Update the dataset to include past performance and confidence level
updated_dataset = []
for entry in dataset:
    operation_type, step, correctness, history, time_spent = entry
    past_performance = calculate_past_performance(history)
    
    # Define thresholds for confidence levels
    if past_performance >= 0.7:
        confidence_level = "High"
    elif past_performance >= 0.5:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    
    updated_entry = (operation_type, step, correctness, history, time_spent, past_performance, confidence_level)
    updated_dataset.append(updated_entry)


for entry in updated_dataset:
    print(entry)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load dataset from CSV
df = pd.read_csv("dataset.csv")

# Extract time spent, correctness, and operation type
time_spent = df['Time'].values
correctness = df['Correctness'].values
operation_type = df['Operation'].values

# Visualization: Scatter plot of time spent vs correctness
plt.figure(figsize=(8, 6))
colors = ['r' if corr == 0 else 'g' for corr in correctness]  # Red for incorrect, green for correct
plt.scatter(time_spent, correctness, c=colors)
plt.xlabel('Time Spent on Step (seconds)')
plt.ylabel('Correctness (0: Incorrect, 1: Correct)')
plt.title('Relationship between Time Spent and Correctness')
plt.show()

# Correlation Analysis
corr_coef = np.corrcoef(time_spent, correctness)[0, 1]
print("Correlation Coefficient:", corr_coef)

# Statistical Testing (t-test)
time_spent_correct = time_spent[correctness == 1]
time_spent_incorrect = time_spent[correctness == 0]
t_stat, p_value = ttest_ind(time_spent_correct, time_spent_incorrect)
print("T-statistic:", t_stat)
print("P-value:", p_value)
if p_value < 0.05:
    print("The difference in time spent between correct and incorrect responses is statistically significant.")
else:
    print("There is no statistically significant difference in time spent between correct and incorrect responses.")

# Feature Importance (t-test)
operation_types_unique = np.unique(operation_type)
for op_type in operation_types_unique:
    time_spent_op = time_spent[operation_type == op_type]
    correctness_op = correctness[operation_type == op_type]
    t_stat_op, p_value_op = ttest_ind(time_spent_op[correctness_op == 1], time_spent_op[correctness_op == 0])
    print(f"Operation Type: {op_type}, T-statistic: {t_stat_op}, P-value: {p_value_op}")

from hmmlearn import hmm
import numpy as np
from sklearn.metrics import mean_squared_error

# Print updated dataset with confidence level
for entry in updated_dataset:
    print(entry)

# Extract observations, hidden states, and confidence levels from the updated dataset
obs_seq = [list(map(int, entry[3])) for entry in updated_dataset[1:]]  # Skip the header row
hidden_states = np.array([int(entry[2]) for entry in updated_dataset[1:]])  
confidence_levels = [entry[6] for entry in updated_dataset[1:]]  

# Create a Multinomial HMM model
model = hmm.MultinomialHMM(n_components=2, n_iter=100, tol=0.01)


predicted_states = []

#  prior probabilities based on confidence level and time spent
def adjust_prior_probabilities(confidence_level, time_spent):
    if confidence_level == "High":
        if time_spent > 20:
            return np.array([0.95, 0.05])  # Higher probability for correct response
        else:
            return np.array([0.85, 0.15])  # Prior of 0.85 for correct response
    elif confidence_level == "Moderate":
        if time_spent > 20:
            return np.array([0.8, 0.2])  # Higher probability for correct response
        else:
            return np.array([0.7, 0.3])  # Prior of 0.7 for correct response
    else:
        if time_spent > 20:
            return np.array([0.3, 0.7])  # Higher probability for correct response
        else:
            return np.array([0.1, 0.9])  # Prior of 0.1 for correct response


# Iterate through the past responses to predict the next hidden state for each step
for i in range(len(obs_seq)):
    # Flatten the observation sequence for the current step into a 1D array
    obs_flat = np.array(obs_seq[i]).reshape(-1, 1)

    # Create a new model for each step
    model = hmm.MultinomialHMM(n_components=2, n_iter=100, tol=0.01)

    # Assign prior probabilities based on the confidence level
    prior_probs = adjust_prior_probabilities(confidence_levels[i], int(updated_dataset[i+1][4]))
    model.startprob_ = prior_probs

    
    model.fit(obs_flat)

    # Predict the most likely hidden state for the current step
    predicted_state = model.predict(obs_flat[-1:].reshape(-1, 1))

    # Append the predicted state to the array
    predicted_states.append(predicted_state[0])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(hidden_states, predicted_states))

# Print the actual and predicted hidden states
print("Actual Hidden States:", hidden_states)
print("Predicted Hidden States:", predicted_states)
print("Root Mean Squared Error (RMSE):", rmse)


from sklearn.metrics import accuracy_score, recall_score, f1_score

# Calculate accuracy
accuracy = accuracy_score(hidden_states, predicted_states)

# Calculate recall
recall = recall_score(hidden_states, predicted_states)

# Calculate F1-score
f1 = f1_score(hidden_states, predicted_states)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)

