# Import required libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("diamond.csv")

# Display first 5 rows
print(df.head())

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# One-Hot Encoding
X = pd.get_dummies(X)

# Save encoded column names
encoded_columns = X.columns

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Numerical columns
numerical_columns = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z"
]

# Feature Scaling
scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "LR_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoded_columns, "columns.pkl")

print("Model saved successfully!")
print("LR_model.pkl created")
print("scaler.pkl created")
print("columns.pkl created")