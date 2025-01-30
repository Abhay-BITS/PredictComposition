# 📌 Composition Predictor

## 📖 Overview

This project predicts the **optimal material composition** that achieves a target bandgap range using **Machine Learning**. Given a dataset of materials and their properties, the model finds the composition that best matches the desired bandgap.

## 🚀 Features

- 📊 **Preprocessing of Material Data** (Handling missing values, feature engineering, and scaling)
- 🔍 **Machine Learning Model** (Random Forest Regressor) to predict composition
- 🎯 **Custom Bandgap Range Filtering**
- 📈 **Model Evaluation Metrics** (MSE, R² Score, Accuracy)
- 🔄 **Scalability for New Datasets**

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/Abhay-BITS/PredictComposition.git
cd PredictComposition
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Prediction

```sh
python main.py
```

## 📂 Project Structure

```
CompositionPredictor/
│── data/
│   ├── MaterialsProject_Perovskite_data.csv   # Dataset
│── main.py               # Entry Point
│── requirements.txt      # Dependencies
│── README.md             # Project Documentation
```

## 🔬 How It Works

1. **Preprocess the data** → Handles missing values, scales numeric features, and encodes categorical values.
2. **Train the model** → Uses a Random Forest Regressor on relevant material properties.
3. **Make predictions** → Finds the best composition matching the given bandgap range.

## 🧪 Example Usage

```python
from predict import predict_composition

# Define target bandgap range
target_bandgap_min = 1.5
target_bandgap_max = 2.0

# Get best composition
best_composition, difference = predict_composition(model, scaler, df_preprocessed, target_bandgap_min, target_bandgap_max)

print(f"Best composition: {best_composition}")
print(f"Difference in bandgap: {difference}")
```

## 📊 Model Performance

| Metric   | Train Score | Test Score |
| -------- | ----------- | ---------- |
| MSE      | 0.0830      | 0.6009     |
| R² Score | 0.9653      | 0.9653     |

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📬 Contact

For any inquiries, reach out via [f20221066@pilani.bits-pilani.ac.in](mailto\:f20221066@pilani.bits-pilani.ac.in).

