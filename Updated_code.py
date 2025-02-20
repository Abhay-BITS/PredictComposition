import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from itertools import product
from pymatgen.core.periodic_table import Element


def get_element_properties(element):
    """Fetch essential physical and chemical properties of an element."""
    elem = Element(element)
    return {
        "atomic_radius": elem.atomic_radius or 0,
        "electronegativity": elem.X or 0,
        "melting_point": elem.melting_point or 0,
        "boiling_point": elem.boiling_point or 0,
    }


def extract_features(row):
    """Extracts elemental properties for A and B site elements in ABO3."""
    A = row['element_1']  # A-site cation
    B = row['element_2']  # B-site cation
    A_props = get_element_properties(A)
    B_props = get_element_properties(B)

    return {
        "A_atomic_radius": A_props["atomic_radius"],
        "B_atomic_radius": B_props["atomic_radius"],
        "A_electronegativity": A_props["electronegativity"],
        "B_electronegativity": B_props["electronegativity"],
        "A_melting_point": A_props["melting_point"],
        "B_melting_point": B_props["melting_point"],
        "A_boiling_point": A_props["boiling_point"],
        "B_boiling_point": B_props["boiling_point"],
    }


def preprocess_data(df):
    """Preprocess dataset: handle missing values and extract features."""
    # Convert boolean columns to numerical
    boolean_columns = ['is_stable', 'is_gap_direct', 'is_metal', 'is_magnetic']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Handle missing and infinite values in numerical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        df[column] = df[column].fillna(df[column].median())

    # Extract elemental features
    df_features = df.apply(extract_features, axis=1, result_type='expand')
    df = pd.concat([df, df_features], axis=1)

    return df


def train_model(df):
    """Train XGBoost model with elemental features."""
    features = [
        'volume', 'density', 'density_atomic', 'energy_per_atom',
        'formation_energy_per_atom', 'is_stable', 'is_gap_direct', 'is_metal',
        'is_magnetic', 'total_magnetization', 'A_atomic_radius',
        'B_atomic_radius', 'A_electronegativity', 'B_electronegativity',
        'A_melting_point', 'B_melting_point', 'A_boiling_point',
        'B_boiling_point'
    ]

    X = df[features]
    y = df['band_gap']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = XGBRegressor(n_estimators=200,
                         max_depth=5,
                         learning_rate=0.1,
                         subsample=0.9,
                         random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def generate_materials(model, scaler, bandgap_min, bandgap_max):
    """Generate potential perovskite compositions and filter based on predicted bandgap."""
    # Common A-site and B-site elements for perovskites
    possible_A = ["Cs", "Ba", "Sr", "Ca", "K", "Na"]
    possible_B = ["Nb", "Ta", "Ti", "Zr", "Hf", "Mo"]

    # Generate all possible combinations
    compositions = list(product(possible_A, possible_B))

    results = []
    for A, B in compositions:
        # Create a sample data point with default values
        sample_data = {
            'volume': 100.0,  # Default values
            'density': 5.0,
            'density_atomic': 15.0,
            'energy_per_atom': -5.0,
            'formation_energy_per_atom': -2.0,
            'is_stable': 1,
            'is_gap_direct': 1,
            'is_metal': 0,
            'is_magnetic': 0,
            'total_magnetization': 0.0
        }

        # Add elemental properties
        element_data = extract_features({'element_1': A, 'element_2': B})
        sample_data.update(element_data)

        # Convert to DataFrame and scale
        df_new = pd.DataFrame([sample_data])
        X_new_scaled = scaler.transform(df_new)

        # Predict bandgap
        predicted_bandgap = model.predict(X_new_scaled)[0]

        # Store if within range
        if bandgap_min <= predicted_bandgap <= bandgap_max:
            results.append((f"{A}{B}O3", predicted_bandgap))

    # Display results
    if not results:
        print(
            f"\nNo perovskite compositions found with bandgap between {bandgap_min:.2f} and {bandgap_max:.2f} eV"
        )
    else:
        results.sort(key=lambda x: x[1])
        print(
            f"\nPredicted Perovskite Materials (Bandgap {bandgap_min:.2f} - {bandgap_max:.2f} eV):"
        )
        print("=====================================")
        for formula, bandgap in results:
            print(f"{formula:<15} {bandgap:.3f} eV")
        print(f"\nTotal materials found: {len(results)}")


def main():
    print("Loading dataset...")
    df = pd.read_csv('data/MaterialsProject_Perovskite_data.csv')
    df_preprocessed = preprocess_data(df)

    print("Training the model...")
    model, scaler = train_model(df_preprocessed)

    while True:
        try:
            print("\nEnter bandgap range (in eV):")
            bandgap_min = float(input("Minimum bandgap: "))
            bandgap_max = float(input("Maximum bandgap: "))

            if bandgap_min < 0 or bandgap_max < 0:
                print("Error: Bandgap values cannot be negative.")
                continue

            if bandgap_min >= bandgap_max:
                print(
                    "Error: Minimum bandgap must be less than maximum bandgap."
                )
                continue

            generate_materials(model, scaler, bandgap_min, bandgap_max)

            choice = input(
                "\nWould you like to try another range? (y/n): ").lower()
            if choice != 'y':
                break

        except ValueError:
            print(
                "Error: Please enter valid numerical values for the bandgap range."
            )
            continue


if __name__ == "__main__":
    main()
