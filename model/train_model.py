import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../data/career_data.csv")

# Features & target
X = data.drop("career", axis=1)
y = data["career"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200,
    max_depth=10,
    random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# Save model
pickle.dump(model, open("career_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model trained and saved!")