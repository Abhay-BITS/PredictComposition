import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Diagnose dataset for missing values, infinities, and statistics
def diagnose_data(df, stage=""):
    print(f"\n--- Data Diagnosis ({stage}) ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nInfinite values:")
    for column in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[column]).sum()
        if inf_count > 0:
            print(f"{column}: {inf_count}")
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())


# Data Preprocessing Function
def preprocess_data(df):
    diagnose_data(df, "Before Preprocessing")

    # Convert boolean columns to numerical
    boolean_columns = ['is_stable', 'is_gap_direct', 'is_metal', 'is_magnetic']
    for col in boolean_columns:
        df[col] = df[col].map({'True': 1, 'False': 0})
    
    # Sort chemical compositions
    df['composition'] = df['chemsys'].apply(lambda x: '-'.join(sorted(x.split('-'))))
    df['num_elements'] = df['composition'].apply(lambda x: len(x.split('-')))

    # Handle missing and infinite values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        df[column] = df[column].fillna(df[column].median())

    diagnose_data(df, "After Preprocessing")
    return df


# Train and Evaluate Model
def train_and_evaluate_model(df):
    features = ['volume', 'density', 'density_atomic', 'energy_per_atom',
                'formation_energy_per_atom', 'is_stable', 'is_gap_direct',
                'is_metal', 'is_magnetic', 'total_magnetization', 'num_elements']
    X = df[features]
    y = df['band_gap']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    print("\nModel Evaluation:")
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R2: {r2_score(y_test, y_test_pred):.4f}")

    return model, scaler


# Predict Compositions Within a Bandgap Range
def predict_composition(model, scaler, df, bandgap_min, bandgap_max):
    features = ['volume', 'density', 'density_atomic', 'energy_per_atom',
                'formation_energy_per_atom', 'is_stable', 'is_gap_direct',
                'is_metal', 'is_magnetic', 'total_magnetization', 'num_elements']

    X = df[features]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    df['predicted_bandgap'] = predictions
    filtered_df = df[(df['predicted_bandgap'] >= bandgap_min) & (df['predicted_bandgap'] <= bandgap_max)]

    if filtered_df.empty:
        return "No compositions found in this range.", None

    return filtered_df[['composition', 'predicted_bandgap']].sort_values(by='predicted_bandgap'), None


# Load and Process Data
df = pd.read_csv('data/MaterialsProject_Perovskite_data.csv')
df_preprocessed = preprocess_data(df)

# Train Model
model, scaler = train_and_evaluate_model(df_preprocessed)

# Predict within a bandgap range
bandgap_min = 1.5
bandgap_max = 1.8
results, _ = predict_composition(model, scaler, df_preprocessed, bandgap_min, bandgap_max)

print("\nPredictions within the given bandgap range:")
print(results)
