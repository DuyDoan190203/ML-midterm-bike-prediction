import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report

# Load dataset
df = pd.read_csv("Data/hour.csv")  # Update path if needed

# Feature engineering
df['high_demand'] = (df['cnt'] > 500).astype(int)  # Binary target for classification
df['dteday'] = pd.to_datetime(df['dteday'])  # Convert to datetime
df['day'] = df['dteday'].dt.day
df['day_of_week'] = df['dteday'].dt.dayofweek
df.drop(columns=['dteday'], inplace=True)

# One-hot encode categorical features
categorical_cols = ['season', 'weathersit', 'mnth', 'weekday', 'hr']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Standardize numerical features
numerical_cols = ['temp', 'atemp', 'hum', 'windspeed']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Split predictors and targets
X = df_encoded.drop(columns=['cnt', 'casual', 'registered', 'high_demand'])
y_regression = df_encoded['cnt']
y_classification = df_encoded['high_demand']

# Train-test splits
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

# Train regression models
print("Training regression models...")
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_pred_lr = lr.predict(X_test_reg)
rmse_lr = mean_squared_error(y_test_reg, y_pred_lr, squared=False)
print(f"Linear Regression RMSE: {rmse_lr:.2f}")

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)
rmse_rf_reg = mean_squared_error(y_test_reg, y_pred_rf_reg, squared=False)
print(f"Random Forest Regressor RMSE: {rmse_rf_reg:.2f}")

# Train classification model
print("Training classification model...")
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight="balanced")
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = rf_clf.predict(X_test_clf)

# Classification report
print("Classification Report for Random Forest:")
print(classification_report(y_test_clf, y_pred_clf))

# Save models and scaler
with open("tuned_random_forest.pkl", "wb") as f:
    pickle.dump(rf_clf, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Models and scaler saved successfully!")
