import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Dataset.csv')
data.columns = data.columns.str.strip()
data['Class/ASD Traits'] = data['Class/ASD Traits'].apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)

X = data.drop(columns=['Case_No', 'Class/ASD Traits'])  
y = data['Class/ASD Traits']

label_encoder = LabelEncoder()

for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

with open('autism_model.pkl', 'wb') as f:
    pickle.dump(random_forest, f)

print("Model saved as autism_model.pkl")

y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("ROC_AUC_SCORE: ", roc_auc_score(y_test, y_pred))
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
