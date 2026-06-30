# Task07_Optimizers



# Task 7 — Optimizer Comparison Challenge 



## 1\. Objective 

To evaluate and compare the training dynamics, convergence speed, and stability of four different optimizers (SGD, SGD with Momentum, Adam, and AdamW) using an identical network architecture.

##  2\. Code Used

```python
# كل محسن له طريقة تفكير رياضية في تحديث الأوزان 
# و الهدف من التاسك مقارنة بينهم 
# عشان هيك عملت لووب يمر على كل محسن من المطلوب و لعدم تكرار الكود
# تعريف  (Optimizers)
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'SGD_Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001)
}

histories = {}

# حلقة لتدريب الموديل لكل محسن
for name, opt in optimizers.items():
    print(f"======= {name} =======")
  
    model_tmp = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
  
    model_tmp.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
    # تدريب لعدد    للمقارنة 
    history = model_tmp.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=0)
    histories[name] = history
  
    #---------------------------------------------------  
 # رسم منحنيات المقارنة
plt.figure(figsize=(14, 5))

# Plot Loss
plt.subplot(1, 2, 1)
for name in histories:
    plt.plot(histories[name].history['val_loss'], label=name)
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
for name in histories:
    plt.plot(histories[name].history['val_accuracy'], label=name)
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```



## 3\. Results

  
SGD : Slowest convergence, smooth but highly stagnant.  
  
SGD with Momentum: Faster than SGD, shows accelerated descent.  
  
Adam: Extremely fast convergence, plateaus to near-optimal accuracy within 2-3 epochs.  
  
AdamW: Matches Adam's rapid speed but demonstrates superior validation stability in later epochs.



## 4\. Short Analysis

**1\. Plot loss and accuracy curves.**  




![optemizers_comparison.png](<./attachments/1d7ec07ffea012f1-optemizers_comparison.png>)

  
**2\. Compare convergence speed and stability:**

  
We observe that Adam and AdamW often achieve high accuracy faster than traditional SGD because Adam uses an adaptive learning rate.  
  
**3\. Discuss how each optimizer navigates the loss landscape differently:**  
  
**SGD **moves blindly in a fixed step. It often gets stuck at local minimums or saddle points due to a lack of dynamic force.  
**SGD with Momentum** is more stable and faster than regular SGD because it takes into account the direction of previous gradients.  
**The Adam and AdamW **models calculate adaptive learning rates for each parameter individually based on previous gradient changes. This allows them to achieve large jumps in steep areas and small, cautious steps when approaching the global minimum.  
  
** 4\. Explain why Adam often outperforms classical optimizers:**  
  
Because it combines the advantages of Momentum (impulse storage) and RMSProp (adjusting the learning rate for each parameter individually), it is smart at overcoming the flat points (Saddle points) in the loss curve.



## 5\. Key Takeaway

While momentum accelerates descent, adaptive algorithms like Adam and AdamW optimize the loss landscape navigation on a per-parameter level, with Adam offering the best balance of speed and structural regularization.









