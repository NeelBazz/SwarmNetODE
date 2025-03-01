import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data
df = pd.read_csv("simulation_results.csv")

# Print basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Save statistics to a text file for reference
with open("summary_statistics.txt", "w") as f:
    f.write(str(df.describe()))

# 📈 Scatter Plot: Clustering Coefficient vs. Max |x_i|
plt.figure(figsize=(8,6))
plt.scatter(df["Clustering Coefficient"], df["Max |x_i|"], alpha=0.5, color='b', edgecolor='black')
plt.xlabel("Clustering Coefficient")
plt.ylabel("Max |x_i|")
plt.title("Effect of Clustering Coefficient on Opinion Spread")
plt.savefig("scatter_clustering_vs_x.png")
plt.show()

# 📈 Scatter Plot: Lambda_2 vs. Max |x_i|
plt.figure(figsize=(8,6))
plt.scatter(df["Lambda_2"], df["Max |x_i|"], alpha=0.5, color='r', edgecolor='black')
plt.xlabel("Lambda_2 (Algebraic Connectivity)")
plt.ylabel("Max |x_i|")
plt.title("Effect of Lambda_2 on Opinion Spread")
plt.savefig("scatter_lambda_vs_x.png")
plt.show()

# 📊 Correlation Analysis
corr_clustering_x = df["Clustering Coefficient"].corr(df["Max |x_i|"])
corr_lambda_x = df["Lambda_2"].corr(df["Max |x_i|"])

print(f"\nCorrelation between Clustering Coefficient and Max |x_i|: {corr_clustering_x:.4f}")
print(f"Correlation between Lambda_2 and Max |x_i|: {corr_lambda_x:.4f}")

# Save correlations to a file
with open("correlation_results.txt", "w") as f:
    f.write(f"Correlation between Clustering Coefficient and Max |x_i|: {corr_clustering_x:.4f}\n")
    f.write(f"Correlation between Lambda_2 and Max |x_i|: {corr_lambda_x:.4f}\n")

# 📊 Histogram of Max |x_i|
plt.figure(figsize=(8,6))
plt.hist(df["Max |x_i|"], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Max |x_i|")
plt.ylabel("Frequency")
plt.title("Distribution of Opinion Fluctuations")
plt.savefig("histogram_max_x.png")
plt.show()

# 📊 Pairplot (Seaborn) for Multi-variable Correlation Visualization
sns.pairplot(df, diag_kind="hist")
plt.savefig("pairplot_analysis.png")
plt.show()
