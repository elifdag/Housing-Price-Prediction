import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_clean_data(filepath, missing_threshold=0.5):
    
    df = pd.read_csv(filepath)
    
    # Drop columns with more than missing_threshold% missing_values
    missing_values = df.isnull().sum()
    missing_values_percentage = (missing_values / len(df)) * 100
    cols_to_drop = missing_values_percentage[missing_values_percentage > missing_threshold * 100].index
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # Fill numerical missing values with median
    num_cols = df_cleaned.select_dtypes(include=["number"]).columns
    df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
    
    # Fill categorical missing values with mode
    cat_cols = df_cleaned.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df_cleaned[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
    
    df_numeric = df_cleaned.drop(columns=cat_cols)
    df_encoded = pd.concat([df_numeric, encoded_df], axis=1)
    
    return df_encoded

def plot_correlation_matrix(df, target_variable):
    
    correlation_matrix = df.corr()
    top_correlated_features = correlation_matrix[target_variable].abs().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(12,8))
    sns.heatmap(correlation_matrix[top_correlated_features.index].loc[top_correlated_features.index],
                annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
    plt.title(f" Features with the Highest Correlation with {target_variable}  ")
    
    plt.show()

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model.__class__.__name__} Model Performance:")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square): {rmse:.2f}")
    print(f"R² (R-squared): {r2:.2f}\n")

def main():
    
    filepath = "AmesHousing.csv"
    df_encoded = load_and_clean_data(filepath, missing_threshold=0.5)
    
    X = df_encoded.drop(columns=['SalePrice'])
    y = df_encoded['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
     # Correlation Analysis
    plot_correlation_matrix(df_encoded, "SalePrice")
    
    # Train and evaluate models
    models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, random_state=42)
    ]
    
    for model in models:
        train_and_evaluate(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()


