# Task08_BatchSize



# Task 08 — Batch Size & Gradient Noise Experiment 



## 1\. Objective

To investigate how different batch sizes (8, 32, and 128) introduce gradient noise, affect the smoothness of loss curves, and influence the model's convergence and generalization.

## 2\. Code Used

```python
# راح اعمل مثل تاسك 7 
# لووب يمر على كل عدد من الايبوك المحددة و اختبارها على نفس الشبكة العصبية 

batch_sizes = [8, 32, 128]
batch_histories = {}

for size in batch_sizes:
    print(f"==== Batch Size: {size} ====")
  
    model_batch = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
  
    model_batch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
    history = model_batch.fit(x_train, y_train, epochs=5, batch_size=size, validation_split=0.1, verbose=1)
    batch_histories[size] = history

# رسم النتائج
plt.figure(figsize=(10, 5))
for size in batch_sizes:
    plt.plot(batch_histories[size].history['val_accuracy'], label=f'Batch Size {size}')

plt.title('Effect of Batch Size on Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```



## 3\. Results



![Effect of Batch Size on Accuracy.png](<./attachments/d667c728db81ffc1-Effect of Batch Size on Accuracy.png>)

**Batch size 8:** Highly unstable and fluctuating loss curve, slowest training time per cycle. However, with our data, it yielded the highest accuracy.  
  
**Batch size 32:** Balance between smoothness and steady convergence.  
  
**Batch size 128**: Very smooth and flat loss curve, fastest execution time per cycle. However, it exhibited slight overfitting due to our data.



## 4\. Short Analysis

Why do smaller batches introduce gradient noise?  
  
Because the gradient is calculated based on a small number of images, making updates 'fluctuating' and not perfectly accurate for each step.  
  
When is this noise beneficial (escaping local minimum)?  
  
Noise helps the model 'escape' local minimums and reach a better region on the loss curve.  
  
Why may larger batches converge faster but generalize worse?  
  
They are faster in computation (hardware efficiency) but may cause overfitting or poor generalization because they lack structured noise.  
  
How does batch size affect the smoothness of loss curves?  
The larger the batch size, the smoother and less fluctuating the loss curve becomes.  
  
As we saw in the accuracy curve.



## 5\. Key Takeaway

Smaller batches create a cluttered and noisy environment for the model, helping it avoid potholes, while larger batches create a smooth and fast road but might cause the model to imprint and not understand the new data (overfitting ).







