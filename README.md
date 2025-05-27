1.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv("F:/2. vs code/Datasets/housing.csv")

# Select only numeric columns
num_features = df.select_dtypes(include=[np.number]).columns

# Plot histograms
for col in num_features:
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

# Plot boxplots
for col in num_features:
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Detect outliers using IQR
print("Outlier Summary:")
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# Summary of dataset
print("\nDataset Summary:")
print(df.describe())


2.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your dataset (update the path to your CSV file)
df = pd.read_csv("F:/2. vs code/Datasets/housing.csv")

# Step 2: Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Step 3: Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Step 4: Show heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Show pair plot
sns.pairplot(numeric_df)
plt.suptitle("Pair Plot of Numeric Features",y=1.02)
plt.show()

3.

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
data = iris.data
labels = iris.target

# Apply PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(data)

# Plot
for i in range(3):
    plt.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=iris.target_names[i])
plt.legend()
plt.title("PCA - Iris Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

4.

import pandas as pd

def find_s(file_path):
    # Load CSV file
    data = pd.read_csv("F:/2. vs code/Datasets/four.csv")

    # Rename label column if it's "enjoy sport"
    if "enjoy sport" in data.columns:
        data.rename(columns={"enjoy sport": "label"}, inplace=True)

    # Keep only rows with "Yes" label
    positive_data = data[data["label"].str.strip().str.lower() == "yes"]
    
    # Initialize hypothesis with the first positive example
    hypothesis = list(positive_data.iloc[0, :-1])

    # Generalize hypothesis with other positive examples
    for i in range(1, len(positive_data)):
        row = positive_data.iloc[i, :-1]
        for j in range(len(hypothesis)):
            if hypothesis[j] != row[j]:
                hypothesis[j] = "?"

    return hypothesis

file_path = "F:/2. vs code/Datasets/four.csv"

# Run the algorithm
final_hypothesis = find_s(file_path)

# Show the result
print("Final Hypothesis:", final_hypothesis)

5.

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.rand(100)

# Label first 50 points
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]

def distance(a, b):
    return abs(a - b)

def knn(train_x, train_y, test_point, k):
    dists = sorted((distance(test_point, x), y) for x, y in zip(train_x, train_y))
    k_nearest = [label for _, label in dists[:k]]
    return Counter(k_nearest).most_common(1)[0][0]

train_x, train_y = data[:50], labels
test_x = data[50:]
k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    print(f"\n--- Results for k = {k} ---")
    predictions = [knn(train_x, train_y, x, k) for x in test_x]

    for i, (x_val, pred) in enumerate(zip(test_x, predictions), start=51):
        print(f"x{i} = {x_val:.3f} → {pred}")

    # Visualization
    class1 = [x for x, label in zip(test_x, predictions) if label == "Class1"]
    class2 = [x for x, label in zip(test_x, predictions) if label == "Class2"]

    plt.figure(figsize=(8, 3))
    plt.scatter(train_x, [0]*50, c=["blue" if l == "Class1" else "red" for l in train_y], label="Train")
    plt.scatter(class1, [1]*len(class1), c="blue", marker="x", label="Class1 Test")
    plt.scatter(class2, [1]*len(class2), c="red", marker="x", label="Class2 Test")
    plt.yticks([0, 1], ["Train", "Test"])
    plt.title(f"k = {k}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

6.

import numpy as np
import matplotlib.pyplot as plt

# 1. Create sample data (X values and noisy sin values as y)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# 2. Function to calculate prediction for one test point
def predict(x0, X, y, tau):
    # Compute weights (how close x0 is to each X point)
    weights = np.exp(- (X - x0) ** 2 / (2 * tau ** 2))
    
    # Add a bias term (constant 1) to X for linear model
    X_b = np.c_[np.ones_like(X), X]
    x0_b = np.array([1, x0])  # bias + x0

    # Create diagonal weight matrix
    W = np.diag(weights)
    
    # Compute theta using weighted linear regression formula
    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
    
    return x0_b @ theta  # prediction for x0

# 3. Plot results
def draw(tau):
    x_line = np.linspace(0, 10, 200)
    y_line = [predict(x0, X, y, tau) for x0 in x_line]

    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(x_line, y_line, color='red', label=f'LWLR (tau={tau})')
    plt.title("Locally Weighted Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

# 4. Run the plot with tau = 0.5
draw(tau=0.5)

7.

import numpy as np
import matplotlib.pyplot as plt

# 1. Create sample data (X values and noisy sin values as y)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# 2. Function to calculate prediction for one test point
def predict(x0, X, y, tau):
    # Compute weights (how close x0 is to each X point)
    weights = np.exp(- (X - x0) ** 2 / (2 * tau ** 2))
    
    # Add a bias term (constant 1) to X for linear model
    X_b = np.c_[np.ones_like(X), X]
    x0_b = np.array([1, x0])  # bias + x0

    # Create diagonal weight matrix
    W = np.diag(weights)
    
    # Compute theta using weighted linear regression formula
    theta = np.linalg.pinv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
    
    return x0_b @ theta  # prediction for x0

# 3. Plot results
def draw(tau):
    x_line = np.linspace(0, 10, 200)
    y_line = [predict(x0, X, y, tau) for x0 in x_line]

    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(x_line, y_line, color='red', label=f'LWLR (tau={tau})')
    plt.title("Locally Weighted Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

# 4. Run the plot with tau = 0.5
draw(tau=0.5)

8.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the breast cancer dataset (already included with sklearn, no internet needed)
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Check model accuracy on test data
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Print the decision tree rules
rules = export_text(model, feature_names=list(data.feature_names))
print("\nDecision Tree Rules:\n", rules)

# Classify a new sample from the test set
new_sample = X_test[0].reshape(1, -1)
prediction = model.predict(new_sample)

print("\nPredicted Class for the New Sample:", "Malignant" if prediction[0] == 0 else "Benign")


9.

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict test labels
y_pred = model.predict(X_test)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation accuracy
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# Visualize some predictions
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, true_lbl, pred_lbl in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}")
    ax.axis('off')

plt.tight_layout()
plt.show()


10.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Load and scale data
data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)
y = data.target

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Print evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y, clusters))
print("\nClassification Report:\n", classification_report(y, clusters))

# PCA to reduce data to 2D for plotting
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', s=100, edgecolor='k', alpha=0.7)
plt.title("K-Means Clusters")
plt.show()

# Plot true labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', s=100, edgecolor='k', alpha=0.7)
plt.title("True Labels")
plt.show()

# Plot clusters with centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', s=100, edgecolor='k', alpha=0.7)
centers = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("Clusters with Centroids")
plt.legend()
plt.show()
