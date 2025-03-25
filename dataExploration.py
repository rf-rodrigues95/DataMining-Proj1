import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
red_wine = pd.read_csv("./data/winequality-red.csv", sep=';')
white_wine = pd.read_csv("./data/winequality-white.csv", sep=';')

# Display basic information
print("Red Wine Summary:")
print(red_wine.info())
print(red_wine.describe())

# Histogram for feature distributions
plt.figure(figsize=(12, 5))
red_wine.hist(bins=30, edgecolor='black', color='red', figsize=(12, 8))
plt.suptitle("Red Wine Feature Distributions")
plt.show()

# Box plot for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=red_wine.drop(columns=['quality']), palette='Reds')
plt.xticks(rotation=90)
plt.title("Red Wine Box Plots")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(red_wine.corr(), annot=True, cmap='Reds', fmt='.2f', linewidths=0.5)
plt.title("Red Wine Correlation Heatmap")
plt.show()

print("\nWhite Wine Summary:")
print(white_wine.info())
print(white_wine.describe())

# Histogram for feature distributions
plt.figure(figsize=(12, 5))
white_wine.hist(bins=30, edgecolor='black', color='blue', figsize=(12, 8))
plt.suptitle("White Wine Feature Distributions")
plt.show()

# Box plot for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=white_wine.drop(columns=['quality']), palette='Blues')
plt.xticks(rotation=90)
plt.title("White Wine Box Plots")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(white_wine.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title("White Wine Correlation Heatmap")
plt.show()



# Pair plots for selected features
#MOCK VALUE FOR SELECTED FEATURES FOR NOW
selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality']
sns.pairplot(red_wine[selected_features], hue='quality', palette='Reds')
plt.suptitle("Red Wine Pair Plot", fontsize=14)
plt.show()

sns.pairplot(white_wine[selected_features], hue='quality', palette='Blues')
plt.suptitle("White Wine Pair Plot", fontsize=14)
plt.show()
