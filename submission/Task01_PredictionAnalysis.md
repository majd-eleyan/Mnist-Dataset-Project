# Task01_PredictionAnalysis

# Task 01 — Prediction Analysis



## 1\. Objective 

Select three random samples from the test dataset (\`x\_test\`), execute model predictions, report the true versus predicted labels, and conceptually analyze how the network processed these inputs.



## 2\. Code Used 

```python
# Select 3 samples from x_test
sample_1 = x_test[30]
plt.imshow(x_test[30],cmap="gray") # عرض الصورة
print("True Label : ", y_test[30])
prediction = model.predict(sample_1.reshape(1, 28, 28))
print("Prediction : ", np.argmax(prediction))

sample_2 = x_test[115]
plt.imshow(x_test[115],cmap="gray") # عرض الصورة
# القيمة الصح
print("True Label : ",y_test[115])
prediction_2 = model.predict(sample_2.reshape(1,28,28))
# التوقع
print("predicton : ",np.argmax(prediction_2) )

ssample_3 = x_test[1000]
plt.imshow(x_test[1000],cmap="gray") # عرض الصورة
# القيمة الصح
print("True Label : ",y_test[1000])
prediction_3 = model.predict(sample_3.reshape(1,28,28))
# التوقع
print("predicton : ",np.argmax(prediction_3) )
```

## 3\. Results

- **Sample 1 (Index 30):** True Label: 3 \| Predicted Label: 3

- **Sample 2 (Index 115):** True Label: 4 \| Predicted Label: 4

- **Sample 3 (Index 1000):** True Label: 9 \| Predicted Label: 9

## 4\. Short Analysis

**1\. Forward Pass Transformation:** The input image is initially treated as a raw 2D matrix of numbers (28\*28). As data flows from the input layer through the hidden layers, the network progressively extracts structural features, starting with basic edge detection. Within the dense layers, every neuron receives inputs from all neurons in the previous layer. Each neuron then computes a weighted sum of its inputs plus a bias (W X + b), applies its activation function, and passes the activated signal forward until it reaches the output layer to produce the final predicted label.

**2\. The Role of Activation Functions (ReLU, Softmax):**

- **ReLU:** Applied to hidden layers, ReLU introduces non-linearity by zeroing out negative values (max(0, x)). This allows the network to learn complex patterns without suffering from vanishing gradients.

- **Softmax:** Positioned at the very final output layer, Softmax squashes raw outputs into a valid probability distribution where all outputs sum up to 1.0 (100%). Instead of arbitrary numbers, it gives a clear confidence score for each class.

**3\. Optimizer Influence (Adam):** Adam automatically adjusts and optimizes individual learning rates during training using momentum. It takes larger steps for slow-changing weights and smaller, more cautious steps for fluctuating weights, successfully improving weight updates to achieve accurate and highly confident predictions.



## 5\. Key Takeaway

The forward pass successfully maps pixel matrices into high-level geometric features, while activation functions and the Adam optimizer work in tandem to output robust, high-confidence probabilistic predictions.



