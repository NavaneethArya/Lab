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


6. apriori

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv(r"C:\Users\navan\Downloads\BIDA Lab\BIDA Lab\Apriori\Assignment-1_Data.csv") 
# Keep only BillNo and Itemname
df = df[['BillNo', 'Itemname']].dropna()

# Group items by BillNo into a list
transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()

# Convert transactions to one-hot encoded format
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# Apply Apriori algorithm
frequent_items = apriori(df_encoded, min_support=0.02, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_items)

# Generate association rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

7. naive bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\navan\Downloads\BIDA Lab\BIDA Lab\Naive Bayes\7817_1.csv")

# Keep required columns & drop missing
df = df[['reviews.title', 'reviews.text', 'reviews.rating']].dropna()

# Combine title + text
df['review'] = df['reviews.title'] + " " + df['reviews.text']

# Convert rating to sentiment labels
df['sentiment'] = df['reviews.rating'].apply(lambda r: "positive" if r >= 4 
                                             else "neutral" if r == 3 
                                             else "negative")

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'],
                                                    test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["positive","neutral","negative"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positive","neutral","negative"],
            yticklabels=["positive","neutral","negative"])
plt.title("Confusion Matrix")
plt.show()

9. 

import pandas as pd
import networkx as nx

# 1. Load dataset
df = pd.read_csv(r"C:\Users\navan\Downloads\BIDA Lab\BIDA Lab\SNA\facebook_combined.txt", sep=" ", names=["user_1", "user_2"])

# 2. Create graph
G = nx.from_pandas_edgelist(df, "user_1", "user_2")
print(f"Graph created with {G.number_of_nodes()} users and {G.number_of_edges()} friendships.\n")

# 3. Degree Centrality (popularity)
degree = dict(G.degree())
top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]

# 4. Betweenness Centrality (using fast approximation)
bet = nx.betweenness_centrality(G, k=200, seed=42)
top_bet = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]

# 5. Closeness Centrality
close = nx.closeness_centrality(G)
top_close = sorted(close.items(), key=lambda x: x[1], reverse=True)[:5]

# 6. Print results
print("Top 5 Most Popular Users (Highest Degree):")
for u, f in top_degree:
    print(f"  User {u}: {f} friends")

print("\nTop 5 Best Connectors (Highest Betweenness):")
for u, s in top_bet:
    print(f"  User {u}: Score {s:.4f}")

print("\nTop 5 Fastest Spreaders (Highest Closeness):")
for u, s in top_close:
    print(f"  User {u}: Score {s:.4f}")


2,3,4 lab

⭐ PROGRAM 2 – Executive Dashboard Design (Using Tableau Public)
“Executive dashboard design for a given business analytics scenario”

I’ll give you VERY CLEAR, STEP-BY-STEP ACTIONS exactly as you should perform them in Tableau.

✅ STEP 1: Open Tableau Public

Download Tableau Public (free)

Open it

Click File → Open

Select your dataset:

If using Excel → sample-data-10mins.xlsx

If using CSV → sales_06_FY2020-21 copy.csv

✅ STEP 2: Load the Data

Once loaded, Tableau shows:

Left side → Sheets inside file

Middle → Data preview

Top → Connections

If Excel:

Click the sheet that contains your sales data
(Example: "Sales", "Orders", "Sheet1", etc.)

✅ STEP 3: Go to Worksheet 1

Bottom → Click Sheet 1

This is where your visualizations start.

⭐ YOU MUST CREATE 5 GRAPHS FOR THE EXECUTIVE DASHBOARD

(These are standard for any business dashboard.)

🎯 VISUAL 1: Monthly Sales Trend (Line Chart)
Purpose: Shows sales growth pattern.
Steps:

Drag order_date → Columns

Right-click order_date → Select Month

Drag total → Rows

Tableau creates a line chart

Click Show Mark Labels (optional)

🎯 VISUAL 2: Sales by Region (Bar Chart)
Purpose: Compare performance across regions.
Steps:

Bottom → New Worksheet

Drag Region → Columns

Drag total → Rows

Select Bar Chart

Sort ↓ using the sort button

🎯 VISUAL 3: Top 10 Selling Products (Bar Chart)
Steps:

New Worksheet

Drag sku (or item name) → Rows

Drag qty_ordered → Columns

Sort descending

Right-click sku → Filter → Top → Top 10 by qty_ordered

🎯 VISUAL 4: Discount % vs Total Sales (Scatter Plot)
Purpose: Shows impact of discounts.

Steps:

New Worksheet

Drag Discount_Percent → Columns

Drag total → Rows

Change Marks → Circle

Drag Region → Color

Go to Analytics → Add Trend Line

🎯 VISUAL 5: Sales by State / City (Map)
Steps:

New Worksheet

Drag State → Marks → Map

Drag total → Color

Set Symbol Map or Filled Map

⭐ STEP 4: Create the Executive Dashboard

Bottom → Click New Dashboard

Drag the 5 worksheets into the dashboard

Arrange using floating or tiled mode

Add a title:
“Executive Sales Overview Dashboard”

This COMPLETES Program 2.

⭐ PROGRAM 3 – Generate Visual Analytics

This is a continuation of Program 2.

You will:

✔ Interpret each visualization
✔ Explain what insights you get

Write these in your record:
1. Monthly Sales Trend

Sales peak in X month

Lowest sales in Y month

Visible seasonal patterns

2. Sales by Region

Region with highest/lowest revenue

Possible business focus areas

3. Top Selling Products

Identify best-performing products

Helps with inventory planning

4. Discount Impact

If points form a rising trend → discounts increase sales

If scattered → discounts not effective

5. Sales by Geography

Identify top states

Useful for resource allocation

THIS COMPLETES PROGRAM 3.

⭐ PROGRAM 4 – Predictive Analytics (Using WEKA/Python/R/Power BI)

Topic:
Enhancing customer experience with predictive analytics

We will do this using WEKA since it’s easiest for lab.

⭐ WEKA PROCESS (Step-by-Step)

Use the same dataset OR a customer behavioral dataset.

🟦 STEP 1: Open WEKA

Start → Search “WEKA”

Choose Explorer

🟦 STEP 2: Load Data

Click Open File

Select your CSV

You may need to convert it to ARFF (WEKA supports CSV too)

🟦 STEP 3: Preprocess

✔ Remove unnecessary columns
✔ Keep features like:

Age

Region

Spending Score

Income

Total Sales

Discount Percent

✔ Ensure target variable:

Cluster, Sales Category, or High/Low Spending

🟦 STEP 4: Choose a Classification Algorithm

Recommended algorithms:

J48 (Decision Tree)

Naive Bayes

Random Forest

Steps:

Go to Classify tab

Choose J48

Click Start

🟦 STEP 5: Evaluate the Model

WEKA shows:

Accuracy

Confusion Matrix

ROC

Tree Structure

🟦 STEP 6: Interpret Results

Examples:

Customers in Region “South” with high income spend more

Higher discount → higher purchase probability

Younger customers tend to prefer specific SKUs

🟦 STEP 7: Conclusion

You write:

Predictive analytics helps to understand future customer buying patterns

bida lab manual

2 and 3

• Tableau is a leading BI tool used for transforming raw data into meaningful insights. 
• This report demonstrates revenue analysis, correlation analysis, and category/gender 
insights using Tableau. 
• Each section includes step-by-step instructions and visuals. 
• Percentage calculations are added where applicable, and drill up/down is explained in 
context. 
• The final dashboard summarizes all visual insights for stakeholders. 
Visual 1: Revenue by Month 
• Drag 'Month' to Columns. 
• Drag 'Revenue' to Rows. 
• Select 'Line Chart' to visualize revenue trend. 
• Add markers for better visibility. 
• Use 'Show Percent of Total' if needed to see monthly contribution percentage. 
• Format title and labels. 
Visual 2: Revenue by Age 
• Drag 'Age' to X-axis. 
• Drag 'Revenue' to Y-axis. 
• Select 'Bar Chart' for age-wise revenue distribution. 
• Add data labels and format. 
• Optionally, use 'Show Percent of Total' for age group contribution. 
Visual 3: Revenue per Category per Gender 
• Drag 'Category' to Y-axis. 
• Drag 'Revenue' to X-axis. 
• Split by 'Gender' using dual-axis or mirrored bars. 
• Use colors to differentiate Male vs Female. 
• Format labels and add data values. 
• Percentage can be shown as contribution per category.
Visual 4: Percentage of Revenue per Region 
• Drag 'Region' to Legend. 
• Drag 'Revenue' to Values. 
• Enable data labels to show percentage contribution. 
• Format colors and chart title. 
Visual 5: Quality vs Discount Correlation 
• Drag 'Discount' to X-axis. 
• Drag 'Quality' to Y-axis. 
• Select 'Scatter Plot' for correlation analysis. 
• Add trend line to identify relationship. 
• Observe outliers for unusual patterns. 
• Percentage can be added for discount contribution if needed.



4. Enhancing customer experience with predictive analytics and data mining by 
taking suitable business scenario (Use Weka/RapidMiner/Spark/R/Microsoft 
power BI). 
Introduction to Power BI 
• Power BI is a Microsoft business intelligence platform for data visualization and analytics. 
• Converts raw datasets into interactive dashboards and actionable insights. 
• Supports multiple data sources: Excel, CSV, SQL, APIs, cloud storage. 
• Users can filter, drill-down, and highlight data interactively. 
• Helps organizations make faster and accurate decisions through visual analytics. 
Core Concepts & Features 
• Data Connection – Connects to files, databases, web sources, online services. 
• Data Preparation – Clean, shape, and transform data using Power Query.
• Data Modeling – Create relationships, hierarchies, and calculations. 
• Visual Analytics – Build interactive dashboards with charts, slicers, and maps. 
• AI Features – Analyze tool explains trends and outliers. 
Visual Demonstrations 
Column Chart – Total Boxes Shipped by Country 
• Insert Column Chart visual. 
• Drag Country to Axis. 
• Drag Boxes Shipped to Values. 
• Power BI auto-aggregates totals. 
• Customize colors, title, and labels (e.g., bright blue). 
• Resize and position on canvas. 
• Apply data labels to show exact totals. 
Bar Chart – Boxes Shipped by Product 
• Insert Bar Chart visual. 
• Drag Product to Axis. 
• Drag Boxes Shipped to Values. 
• Switch to horizontal bars for long product names. 
• Format bar colors (e.g., orange gradient) and labels. 
• Apply data labels. 
• Resize chart neatly. 
Line Chart – Sales Trend Over Time 
• Insert Line Chart visual. 
• Drag Date to Axis (auto hierarchy applied). 
• Drag Amount to Values. 
• Enable drill-down by month/day. 
• Format line color (e.g., green), markers, and gridlines.
• Add axis labels and chart title. 
• Enable tooltips to show exact values. 
Step 8: Pie Chart – Shipment Share by Country 
• Insert Pie Chart visual. 
• Drag Country to Legend. 
• Drag Boxes Shipped to Values. 
• Adjust slice colors (e.g., blue, orange, green, red). 
• Enable data labels for percentages. 
• Resize chart and place clearly on canvas. 
• Add descriptive title and legend. 
Step 9: Scatter Plot – Boxes Shipped vs Sales Amount 
• Insert Scatter Chart visual. 
• Drag Boxes Shipped to X-axis. 
• Drag Amount to Y-axis. 
• Optionally, add Country to Legend for color coding. 
• Enable tooltips to display details on hover. 
• Format marker size and colors (e.g., purple). 
• Analyze correlations and patterns.
    
