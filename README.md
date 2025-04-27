# Titanic-EDA

Welcome to the Titanic Exploratory Data Analysis (EDA) project!
In this project, we deeply explore the famous Titanic dataset to understand the hidden patterns, data distributions, correlations, and important factors behind survival.

📂 Dataset
Source: Kaggle Titanic Dataset

Files:

train.csv

test.csv

🚀 Project Flow
Import Libraries

pandas, numpy, matplotlib, seaborn

Load Dataset

Read CSV file into a pandas DataFrame.

Basic Information

Dataset shape, data types, column names, first few rows.

Missing Values Handling

Numerical columns filled with median.

Categorical columns filled with mode.

Univariate Analysis

Distribution plots for numerical features.

Count plots for categorical features.

Bivariate Analysis

Correlation heatmap for numerical features.

Insights and Observations

Key findings about survival patterns.

🛠 Technologies Used
Python

Pandas

Numpy

Matplotlib

Seaborn

Jupyter Notebook / Google Colab

📈 Key Visualizations
Histograms of Age, Fare, etc.

Count plots of Sex, Pclass, Embarked, etc.

Heatmap showing feature correlations.

🔍 Sample Code
python
Copy
Edit
# Load data
import pandas as pd
data = pd.read_csv('train.csv')

# Fill missing numerical values with median
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Fill missing categorical values with mode
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

corr = data.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
📋 Conclusion
Female passengers had a much higher survival rate.

Younger passengers (kids) had higher survival rates.

Higher class passengers (Pclass=1) survived more.

Embarkation point also slightly affected survival.

📎 Future Work
Feature Engineering for better insights.

Model building to predict survival.

Note:
This is a basic EDA project for learning purposes. Future improvements can include deeper feature analysis, hypothesis testing, and machine learning modeling.
