import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Data Preprocessing

# Load the dataset
data = pd.read_csv("your_dataset.csv")

# Split the dataset into features (X) and target variable (y)
X = data.drop('Pollution', axis=1)  # Change 'Pollution' to the name of your target variable
y = data['Pollution']                # Change 'Pollution' to the name of your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Selection

# Initialize the model
model = RandomForestClassifier()

# Step 3: Model Training

# Train the model
model.fit(X_train, y_train)

# Step 4: Model Evaluation

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 5: Model Saving

# Save the model
joblib.dump(model, 'model.pkl')

# Step 6: Prediction

# Load the model
model = joblib.load('model.pkl')

# Once your model is trained and evaluated, you can use it to make predictions on new data
new_data = [[1.2, 64.51, 25, 60, 12.3456, 78.9123]]
prediction = model.predict(new_data)
print("Predicted Pollution:", prediction)

