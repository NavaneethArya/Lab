k means (5)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Age', 'Annual_Income_(k$)', 'Spending_Score']]

# Scale features
X_scaled = StandardScaler().fit_transform(X)

# Elbow + Silhouette
inertia = []
sil = []
K = range(2, 11)

for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, labels))

# Plots
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K, inertia, marker='o'); plt.title("Elbow Method")

plt.subplot(1,2,2)
plt.plot(K, sil, marker='o'); plt.title("Silhouette Score")

plt.show()

# Final model
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster summary
print(df.groupby('Cluster')[['Age','Annual_Income_(k$)','Spending_Score']].mean())

# Scatter plot
sns.scatterplot(data=df, x='Annual_Income_(k$)', y='Spending_Score', hue='Cluster', palette='viridis')
plt.title("Customer Clusters")
plt.show()
