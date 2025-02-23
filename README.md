# Housing Price Prediction

![image](https://github.com/user-attachments/assets/f4d580f9-8a64-4bb9-a9dc-2e24c65b9e1b)


### Overview
In this project, I tried to predict housing prices using Ames Housing dataset. It includes data cleaning, exploratory data analysis(EDA), and implementation of machine learning models such Linear Regression and Random Forest Regressor. The Python libraries I used are Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn.

### Structures
Data Cleaning:
Drops columns with more than 50% missing values.
Fills numerical missing values with the median and categorical missing values with the mode.
Performs one-hot encoding for categorical variables.

Exploratory Data Analysis (EDA):
Visualizes the correlation matrix for the top features most correlated with the target variable (SalePrice).

Model Training and Evaluation:
Implements and evaluates Linear Regression and Random Forest Regressor models.
Evaluates models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).

