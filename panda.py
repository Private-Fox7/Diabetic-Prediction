import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

# Load the data
data = r'C:\\Users\\Private Fox\\Downloads\\diabetes.csv'
c_df = pd.read_csv(data)
print(c_df.describe())
# Split features (X) and target (y)
features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = c_df[features]
y = c_df['Outcome']
print(X)
print(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model = LogisticRegression(max_iter=500, solver='liblinear')
model.fit(X_train, y_train)

# Make predictions/testing phase
y_pred = model.predict(X_test)
print("X_test",X_test.head(5))
print("y_pred",y_pred[:5])
print("y_test",y_test[:5])
# Evaluate the model
print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example: Predict for a single patient (using the first test case)
sample = (5,166,72,19,175,2.3,0.587,51)
sample = np.array(sample).reshape(1, -1)
prediction = model.predict(sample)
print("\nSample Prediction:", "Diabetic" if prediction == 1 else "Not Diabetic")