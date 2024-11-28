import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def knn_classifier(X_train, X_test, y_train, y_test, k_values):
  """
  Implements KNN classification and evaluates accuracy for different k values.

  Args:
      X_train: Training data features.
      X_test: Testing data features.
      y_train: Training data labels.
      y_test: Testing data labels.
      k_values: A list of k values to try for the KNN classifier.

  Returns:
      A dictionary containing accuracy scores for each k value.
  """  
  accuracy_scores = {}
  for k in k_values:
    # Ensure k is less than or equal to the number of training samples
    if k > X_train.shape[0]:
      k = X_train.shape[0]
      print(f"Warning: k value {k} is greater than the number of training samples. Setting k to {k}.")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[k] = accuracy
  return accuracy_scores

# Load data from the specified path
data_path = "D:\\mca\\2nd sem\\PR\\project\\assignment\\iris.csv"
X, y = pd.read_csv(data_path).iloc[:, :-1], pd.read_csv(data_path).iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define k values to try
k_values = [1, 3, 5, 7]

# Perform KNN classification and get accuracy scores
accuracy_scores = knn_classifier(X_train, X_test, y_train, y_test, k_values)

# Print accuracy scores for different k values
print("Accuracy scores for different k values:")
for k, accuracy in accuracy_scores.items():
    print(f"k = {k}, Accuracy = {accuracy:.4f}")
