# Task04_EarlyStopping



## 1\. Objective

To analyze how the EarlyStopping callback monitors validation loss to prevent overfitting and to study its impact as an indirect form of regularization.

##  2\. Code Used

```python
# إعادة بناء الموديل لضمان بداية نظيفة

model_p3 = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

model_p3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# تعريف الـ Callback
callback_p3 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# تدريب الموديل
history_p3 = model_p3.fit(x_train, y_train, epochs=50, validation_split=0.1, callbacks=[callback_p3], verbose=1)
```



## 3\. Results

in patience = 3

**Stopped Epoch:** Training stopped automatically at Epoch 10.

**Best Weights Restored ** the model rolled back to the epoch with the lowest validation loss.



![EarlyStopping_p3.png](<./attachments/4d40922c24024d0f-EarlyStopping_p3.png>)

in patience = 5

**Stopped Epoch:** Training stopped automatically at Epoch 19.



![EarlyStopping.png](<./attachments/0816b98d5512579f-EarlyStopping.png>)





## 4\. Short Analysis



**1\. At which epoch did training stop?**

Training stops when the model notices that the val\_loss (loss in validation data) hasn't improved a certain number of times (depending on the patience). You can see the final value in the output of the previous cell.

**2\. Why does the validation loss control this decision?**

Because the val\_loss is the true measure of the model's intelligence on data it has never seen before. If we relied solely on training loss, the model would simply memorize (rote) instead of understanding, which is overfitting.

**3\.What happens if you increase patience (e.g., to 5)?**

Increasing patience gives the model a longer chance. Sometimes the loss increases slightly and then decreases more effectively; therefore, patience = 5 allows the model to continue for a longer period before giving up and deciding to stop.

**4\. Would a different optimizer (e.g., SGD) change the EarlyStopping pattern?**

Yes. Adam is very quick to reach results and may stop early, while SGD is often slower and more erratic, which can cause EarlyStopping to be delayed or react differently to loss variations.

**5\. Explain how EarlyStopping acts as an indirect form of regularization.**

Regularization simply aims to prevent the model from becoming overly complex and memorizing trivial details in the data. EarlyStopping does this by stopping training at the optimal moment before the model begins to shift from understanding to memorizing.



## 5\. Key Takeaway

Early stopping is a technique to avoid overfitting by halting the training process when the model's performance on a dataset begins to deteriorate.  
  
Patience determines the number of consecutive epochs in which performance can deteriorate before an actual stopping point is reached, thus preventing a halt due to temporary fluctuations.







