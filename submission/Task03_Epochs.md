# Task03_Epochs



## 1\. Objective 

To analyze neural network training dynamics, detect signs of overfitting, and study convergence stability across 5, 10, and 20 epochs.



## 2\. Code Used

```python
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# تدريب المودل وحفظ النتائج في history
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

#  رسم منحنى الخسارة (Loss vs Val Loss)
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss (loss)')
plt.plot(history.history['val_loss'], label='Validation Loss (val_loss)')
plt.title('Loss Exploration')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

#  رسم منحنى الدقة (Accuracy vs Val Accuracy)
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy (accuracy)')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (val_accuracy)')
plt.title('Accuracy Exploration')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

# طبقت هذا الكود على الثلاث حالات epoch (5,10 , 20)
# يمكن عمل لووب
```



## 3 . Results

  
5 Epochs: Clean convergence. Training and validation metrics improve tightly together.



![loss_curve_5_epoch.png](<./attachments/29497f46820fcdd6-loss_curve_5_epoch.png>)

  
10 Epochs: Steady state. Optimal convergence reached where loss plateaus smoothly.



![loss_curve_10_epoch.png](<./attachments/ec8722eec0eabf34-loss_curve_10_epoch.png>)

  
20 Epochs: overfitting appear. Training loss keeps decreasing while validation loss begins to tick upward or fluctuate.



![loss_curve_20_epoch.png](<./attachments/309aa710109e8500-loss_curve_20_epoch.png>)



## 4\. Short Analysis

**Overfitting Identification:** In the 5 and 10 epoch runs, the model generalizes well because train\_loss and val\_loss decrease . However, during the 20-epoch run the training loss continues its downward trajectory toward zero, while the validation loss plateaus and starts creeping upward. This gap is a Overfitting, where the network stops learning generic patterns and begins memorizing .  


**Explain how the optimizer (Adam) influenced the speed and stability of convergence**: Adam heavily accelerates speed learning mechanism, allowing the model to reach near-optimal accuracy within the first 5 epochs. Adam utilizes momentum to smooth out gradient updates, ensuring steady, non-oscillating curves during the initial stages. However, because it adapts so efficiently, running it for too many epochs (like 20) without regularization pushes the weights to overfitting .

## 5\. Key Takeaway

Extended training epochs with an aggressive optimizer like Adam drive the model from healthy feature learning into overfitting, making early monitoring essential.









