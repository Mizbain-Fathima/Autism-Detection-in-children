# Autism Detection in Children

This project uses **Machine Learning** to detect Autism Spectrum Disorder (ASD) traits in children from a structured dataset.  
The model is trained using a **Random Forest Classifier** and achieves good accuracy in classifying whether a child shows ASD traits.  

---

## 📌 Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Folder Structure](#folder-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [Authors](#authors)  

---

## 🚀 Project Overview

Autism Spectrum Disorder (ASD) is a developmental condition where early detection is critical for intervention.  
This project builds a **Random Forest ML model** that learns from a dataset of children’s behavioral and demographic traits to classify whether ASD traits are present.  

---

## ✨ Features

- Preprocessing of categorical and numerical features  
- Label encoding for categorical variables  
- Train-test split for model evaluation  
- **Random Forest Classifier** for prediction  
- Model persistence using **pickle** (`autism_model.pkl`)  
- Evaluation metrics: Accuracy, F1-score, ROC-AUC, Confusion Matrix, Classification Report  

---

## 📂 Dataset

- **File**: `Dataset.csv`  
- **Target Variable**: `Class/ASD Traits` (Yes → 1, No → 0)  
- **Dropped Column**: `Case_No` (ID column)  
- **Features**: Child’s responses and attributes (after encoding categorical features)  
- **Split**: 80% Training, 20% Testing  

---

## 🧠 Model Architecture

- **Algorithm**: Random Forest Classifier (from `sklearn.ensemble`)  
- **Preprocessing**: Label Encoding for categorical features  
- **Model Persistence**: Saved as `autism_model.pkl` using pickle  
- **Evaluation Metrics**: Accuracy, F1-score, ROC-AUC, Confusion Matrix  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mizbain-Fathima/Autism-Detection-in-children.git
cd Autism-Detection-in-children/Autism

# Install required libraries
pip install -r requirements.txt
````

---

## ▶️ Usage

### Train and Save Model

```bash
python autism_detection.py
```

This will:

* Preprocess the dataset
* Train a Random Forest model
* Save it as `autism_model.pkl`
* Print performance metrics

### Load Model for Prediction

```python
import pickle
import pandas as pd

# Load trained model
with open('autism_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input (must match feature format)
sample = pd.DataFrame([{
    'A1_Score': 1,
    'A2_Score': 0,
    'Age_Mons': 24,
    'Ethnicity': 3,
    'Gender': 1,
    ...
}])

prediction = model.predict(sample)
print("Prediction:", "ASD" if prediction[0] == 1 else "No ASD")
```

---

## 📊 Results

During evaluation on the test dataset:

* **Accuracy**: \~ (Fill with actual value, e.g., 0.89)
* **F1 Score**: \~ (Fill with actual value, e.g., 0.87)
* **ROC-AUC**: \~ (Fill with actual value, e.g., 0.91)
* **Confusion Matrix**:

  ```
  [[TN, FP],
   [FN, TP]]
  ```

---

## 📁 Folder Structure

```
Autism/
├── Dataset.csv           # Input dataset
├── autism_detection.py   # Training & evaluation script
├── autism_model.pkl      # Saved trained model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repo
2. Create a new branch (`git checkout -b feature-xyz`)
3. Commit your changes
4. Submit a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* Scikit-learn for Random Forest implementation
* Pandas & Numpy for preprocessing
* Dataset providers for making ASD-related datasets available for research
ke me to also generate a **`requirements.txt`** file (with sklearn, pandas, numpy, etc.) so your repo is plug-and-play for anyone cloning it?
```
