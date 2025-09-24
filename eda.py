# eda_titanic.py
# Task 2: Exploratory Data Analysis (EDA) on Titanic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 2: Basic Info
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

# Step 3: Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Step 4: Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Univariate Analysis - Histograms
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Titanic Features")
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age & Fare")
plt.show()

# Step 6: Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 7: Pairplot (Relationships between numeric features)
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue="Survived")
plt.show()

# Step 8: Categorical Analysis
plt.figure(figsize=(8, 5))
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Pclass", hue="Survived", data=df, palette="muted")
plt.title("Passenger Class vs Survival")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Sex", hue="Survived", data=df, palette="coolwarm")
plt.title("Gender vs Survival")
plt.show()


