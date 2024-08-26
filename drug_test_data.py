import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
drug_data = pd.read_csv("project_data.csv")

# Drop missing values
drug_data.dropna(inplace=True)
drug_data.reset_index(drop=True, inplace=True)

# Convert Seizure Date to datetime and extract features
drug_data["Seizure Date"] = pd.to_datetime(drug_data["Seizure Date"])
drug_data["Year"] = drug_data["Seizure Date"].dt.year
drug_data["Month"] = drug_data["Seizure Date"].dt.month
drug_data["Day"] = drug_data["Seizure Date"].dt.day
drug_data.drop("Seizure Date", axis=1, inplace=True)

# Remove commas from "Quantity Seized" and convert to numeric
drug_data["Quantity Seized"] = drug_data["Quantity Seized"]
drug_data["Quantity Seized"] = pd.to_numeric(
    drug_data["Quantity Seized"], errors="coerce"
)

# Drop rows where "Quantity Seized" is NaN
drug_data.dropna(subset=["Quantity Seized"], inplace=True)

# Select relevant features and target variable
X = drug_data[["City", "Day", "Month", "Year"]]
y = drug_data["Quantity Seized"]

# Convert categorical variable "City" to dummy variables
X = pd.get_dummies(X, columns=["City"], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
