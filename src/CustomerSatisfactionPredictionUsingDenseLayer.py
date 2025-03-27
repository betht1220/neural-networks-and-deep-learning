# README file: tensorFlow.python.keras.layers.Dense.README.md
import tensorflow as tf
import numpy as np

# 1️⃣ Create Sample Data
# Each row represents [Response Time, Product Quality, Support Rating]
X_train = np.array([
    [5, 8, 4],  # Fast response, good product, good support -> likely satisfied
    [20, 6, 3], # Slow response, average product, bad support -> less satisfied
    [3, 9, 5],  # Very fast response, great product, great support -> very satisfied
    [25, 4, 2], # Very slow response, bad product, bad support -> dissatisfied
    [7, 7, 4],  # Decent response, average product, good support -> moderately satisfied
], dtype=np.float32)

# Satisfaction score (0 = dissatisfied, 1 = very satisfied)
y_train = np.array([
    [0.9],  # Very satisfied
    [0.3],  # Less satisfied
    [1.0],  # Extremely satisfied
    [0.1],  # Very dissatisfied
    [0.7],  # Moderately satisfied
], dtype=np.float32)

# 2️⃣ Build a Simple Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,)),  # Hidden layer (3 neurons)
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer (1 neuron, satisfaction score 0-1)
])

# 3️⃣ Compile the Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4️⃣ Train the Model
model.fit(X_train, y_train, epochs=100, verbose=1)

# 5️⃣ Make a Prediction
# Example: A customer with response time of 10 mins, product quality 7, support rating 4
new_customer = np.array([[10, 7, 4]], dtype=np.float32)
prediction = model.predict(new_customer)

print(f"Predicted Customer Satisfaction Score: {prediction[0][0]:.2f}")
