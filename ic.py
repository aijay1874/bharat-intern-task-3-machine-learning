# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2: Load and preprocess the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a model using scikit-learn
from sklearn.linear_model import LogisticRegression

# Initialize a logistic regression model
sklearn_model = LogisticRegression(max_iter=200)
# Train the model
sklearn_model.fit(X_train, y_train)
# Make predictions
y_pred_sklearn = sklearn_model.predict(X_test)
# Evaluate the model
print("Scikit-learn Model Accuracy:", accuracy_score(y_test, y_pred_sklearn))
print("Scikit-learn Model Classification Report:")
print(classification_report(y_test, y_pred_sklearn))

# Step 5: Build a model using TensorFlow/Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# Step 6: Compile the TensorFlow model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the TensorFlow model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# Step 8: Evaluate the TensorFlow model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"TensorFlow Model Accuracy: {accuracy:.4f}")

# Make predictions with the TensorFlow model
y_pred_tensorflow = np.argmax(model.predict(X_test), axis=1)

# Print classification report for TensorFlow model
print("TensorFlow Model Classification Report:")
print(classification_report(y_test, y_pred_tensorflow))