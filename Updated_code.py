import pandas as pd
import numpy as np
import joblib
import os
from pymatgen.core import Element
from itertools import product

def get_element_properties(element_symbol):
    """Get elemental properties."""
    try:
        element = Element(element_symbol)
        return {
            'atomic_number': element.Z,
            'group': element.group,
            'row': element.row,
            'electronegativity': element.X,
            'atomic_radius': element.atomic_radius,
            'atomic_radius_calculated': element.atomic_radius_calculated,
            'van_der_waals_radius': element.van_der_waals_radius,
            'electron_affinity': element.electron_affinity,
            'ionization_energy': element.ionization_energies[0] if element.ionization_energies else np.nan
        }
    except:
        return {k: np.nan for k in ['atomic_number', 'group', 'row', 'electronegativity',
                                    'atomic_radius', 'atomic_radius_calculated',
                                    'van_der_waals_radius', 'electron_affinity',
                                    'ionization_energy']}

def prepare_input_features(elements, model_features):
    """Prepare input features matching model requirements."""
    properties = [get_element_properties(elem) for elem in elements]
    
    features = {}
    for prop in properties[0].keys():
        values = [p[prop] for p in properties]
        features[f'mean_{prop}'] = np.mean(values)
        features[f'std_{prop}'] = np.std(values)
    
    df = pd.DataFrame([features])
    df = df.reindex(columns=model_features, fill_value=np.nan)
    return df

def load_model(model_name):
    """Load the pre-trained model."""
    model_paths = {
        'LightGBM': "/Users/apple/Downloads/Perovskite-main copy/LightGBM_model.pkl",
        'XGBoost': "/Users/apple/Downloads/Perovskite-main copy/XGBoost_model.pkl",
        'Random Forest': "/Users/apple/Downloads/Perovskite-main copy/Random Forest_model.pkl",
        'Gradient Boosting': "/Users/apple/Downloads/Perovskite-main copy/Gradient Boosting_model.pkl"
    }
    
    if model_name not in model_paths or not os.path.exists(model_paths[model_name]):
        raise FileNotFoundError(f"Model file for '{model_name}' not found. Available models are: {list(model_paths.keys())}")
    
    model = joblib.load(model_paths[model_name])
    return model

def predict_bandgap(elements, model):
    """Predict bandgap with correct features."""
    input_features = prepare_input_features(elements, model.feature_names_in_)
    return model.predict(input_features)[0]

def generate_materials(model_name, bandgap_min, bandgap_max):
    """Generate potential perovskite compositions and filter based on predicted bandgap."""
    possible_A = ["Cs", "Ba", "Sr", "Ca", "K", "Na"]
    possible_B = ["Nb", "Ta", "Ti", "Zr", "Hf", "Mo"]
    possible_O = ["O"]
    
    compositions = list(product(possible_A, possible_B, possible_O))
    results = []
    
    model = load_model(model_name)
    for A, B, O in compositions:
        elements = [A, B, O]
        try:
            predicted_bandgap = predict_bandgap(elements, model)
            if bandgap_min <= predicted_bandgap <= bandgap_max:
                results.append((f"{A}{B}{O}3", predicted_bandgap))
        except:
            continue
    
    if not results:
        print(f"\nNo perovskite compositions found with bandgap between {bandgap_min:.2f} and {bandgap_max:.2f} eV")
    else:
        results.sort(key=lambda x: x[1])
        print(f"\nPredicted Perovskite Materials (Bandgap {bandgap_min:.2f} - {bandgap_max:.2f} eV):")
        print("=====================================")
        for formula, bandgap in results:
            print(f"{formula:<15} {bandgap:.3f} eV")
        print(f"\nTotal materials found: {len(results)}")

def main():
    print("\nAvailable models:")
    models = ['LightGBM', 'XGBoost', 'Random Forest', 'Gradient Boosting']
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")
    
    while True:
        try:
            choice = input("\nEnter the number of the model you want to use (1-4): ")
            model_name = models[int(choice)-1]
            break
        except (ValueError, IndexError):
            print("Invalid choice. Please enter a number between 1 and 4.")
    
    while True:
        try:
            bandgap_min = float(input("Minimum bandgap: "))
            bandgap_max = float(input("Maximum bandgap: "))
            if bandgap_min < 0 or bandgap_max < 0 or bandgap_min >= bandgap_max:
                print("Error: Invalid bandgap range.")
                continue
            generate_materials(model_name, bandgap_min, bandgap_max)
            break
        except ValueError:
            print("Error: Please enter valid numerical values for the bandgap range.")
            continue

if __name__ == "__main__":
    main()
