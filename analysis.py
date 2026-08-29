"""
Real-World Data Project - Retail Domain
Dataset: Diamonds (retail pricing dataset, ~54,000 records) via seaborn
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
OUT = "/home/claude/p1_retail/"

# 1. Load data
df = sns.load_dataset("diamonds")
print("Shape:", df.shape)
print(df.head())
print(df.isna().sum())

# 2. Data cleaning
before = len(df)
df = df.drop_duplicates()
df = df[(df['price'] > 0) & (df['carat'] > 0) & (df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
after = len(df)
print(f"Removed {before - after} invalid/duplicate rows")

# 3. Summary stats
summary = df.describe(include='all').T
summary.to_csv(OUT + "summary_stats.csv")

# 4. Visualizations
fig, ax = plt.subplots(figsize=(7,5))
sns.histplot(df['price'], bins=50, kde=True, ax=ax, color="#2E86AB")
ax.set_title("Distribution of Diamond Prices")
ax.set_xlabel("Price (USD)")
plt.tight_layout(); plt.savefig(OUT + "price_dist.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=df.sample(3000, random_state=1), x="carat", y="price", hue="cut", alpha=0.5, ax=ax, palette="viridis")
ax.set_title("Price vs Carat, by Cut Quality")
plt.tight_layout(); plt.savefig(OUT + "carat_vs_price.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,5))
sns.boxplot(data=df, x="cut", y="price", order=["Fair","Good","Very Good","Premium","Ideal"], ax=ax, palette="Set2")
ax.set_title("Price Distribution by Cut Quality")
plt.tight_layout(); plt.savefig(OUT + "price_by_cut.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,5))
corr = df[["carat","depth","table","price","x","y","z"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features")
plt.tight_layout(); plt.savefig(OUT + "correlation_heatmap.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(data=df, x="clarity", y="price", estimator=np.mean, order=sorted(df['clarity'].unique(), key=lambda c: df[df['clarity']==c]['price'].mean()), ax=ax, palette="mako")
ax.set_title("Average Price by Clarity Grade")
plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig(OUT + "price_by_clarity.png", dpi=150); plt.close()

# 5. Key findings text
findings = {
    "n_records_cleaned": after,
    "avg_price": round(df['price'].mean(),2),
    "median_price": round(df['price'].median(),2),
    "carat_price_corr": round(df['carat'].corr(df['price']),3),
    "most_common_cut": df['cut'].value_counts().idxmax(),
    "highest_avg_price_clarity": df.groupby('clarity')['price'].mean().idxmax(),
}
print(findings)
import json
with open(OUT + "findings.json","w") as f:
    json.dump(findings, f, indent=2)
