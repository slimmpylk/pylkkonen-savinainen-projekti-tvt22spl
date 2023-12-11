import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import math


# Load JSON data
with open('updated_data.json', 'r') as file:
    data = json.load(file)

# Extract sensor values and convert to NumPy array
x_train = np.array([[item["sensorvalue_b"], item["sensorvalue_c"], item["sensorvalue_d"]] for item in data])

# Dummy labels (replace this with actual labels)
y_train = np.array([item["sensorvalue_a"] for item in data]) 

# Define the model
model = models.Sequential([
    layers.Dense(units=22, activation='relu', input_shape=(3,)),  # Increased number of neurons
    layers.Dropout(0.2),  # Dropout layer for regularization
    layers.Dense(units=6, activation='softmax')  # Output layer with 6 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=200, batch_size=32)

predictions = model.predict(x_train)
print(predictions.shape)

for i in range(1):
     print(f"Arvioitiin accuracy: {i+1}:", predictions[0, i])
     sum = predictions[0] +  predictions[1] +  predictions[2] +  predictions[3] +  predictions[4] +  predictions[5]
     print("Summa:", sum)
     plt.stem(predictions[0, :])
     plt.show()

for layer in model.layers:
    if len(layer.get_weights()) > 0:  # Check if the layer has weights
        weights, biases = layer.get_weights()
        print(f"Weights of {layer.name}:\n {weights}")

print(f"Biases of {layer.name}:\n {biases}\n")
print(F"2nd bias value of layer 1 (check)\n {biases[1]}")

# Evaluate the model
loss, accuracy = model.evaluate(x_train, y_train)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# 3. Vaihe manuaalinen laskenta pure pythonilla


# 4. Vaihe 
def save_weights_to_file(model, filename):
    with open(filename, 'w') as file:
        file.write("#ifndef NEURAL_NETWORK_WEIGHTS_H\n")
        file.write("#define NEURAL_NETWORK_WEIGHTS_H\n\n")

        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights:
                w, b = weights
                # Write weights as an array
                file.write(f"float weights_layer_{i}[] = {{")
                for weight in w.flatten():
                    file.write(f"{weight}, ")
                file.write("};\n\n")

                # Write biases as an array (yhteensä 4 arrays)
                file.write(f"float biases_layer_{i}[] = {{")
                for bias in b.flatten():
                    file.write(f"{bias}, ")
                file.write("};\n\n")
            else:
                file.write(f"// Layer {layer.name} has no weights or biases\n\n")

        file.write("#endif // NEURAL_NETWORK_WEIGHTS_H\n")

save_weights_to_file(model, 'neuroverkonKertoimet.h')


