# Task09_Activations



# Task 09 — Activation Function Swap (ReLU vs Tanh vs GELU) 

## 1\. Objective 

To evaluate how swapping different activation functions (ReLU, Tanh, Softsign, and GELU) impacts gradient flow, convergence speed, and network performance.

## 2\. Code Used

```python
## عملت لووب يمر على كل دالة تنشيط و يختبر نفس الشبكة العصبية عليهم 
activations = ['relu', 'tanh', 'softsign', 'gelu']
act_histories = {}

for act in activations:
    print(f"===== Activation: {act} =====")
  
    model_act = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=act),
        keras.layers.Dense(10, activation='softmax')
    ])
  
    model_act.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
    #  Fit 
    history = model_act.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.1, verbose=0)
    act_histories[act] = history

# رسم المقارنة
plt.figure(figsize=(10, 5))
for act in activations:
    plt.plot(act_histories[act].history['val_accuracy'], label=act)

plt.title('Activation Functions Comparison (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```



## 3\. Results 

Based on the generated activation comparison plot:



![Activation Functions Comparison.png](<./attachments/0d40595b248df4c8-Activation Functions Comparison.png>)

**GELU:** Demonstrated a steady, continuous upward trajectory, ultimately securing the highest final validation accuracy (97.7%) by the 4th epoch.

**ReLU:** Showed the fastest initial convergence rate, dominating the early epochs (0 to 2), before plateauing and stabilizing around \~97.55%.

**Tanh:** Converged smoothly below ReLU, finishing in third place (\~97.4%).

**Softsign:** Remained the lowest performing activation throughout the experiment, showing slower but steady learning dynamics (\~97.2%).



## 4\. Short Analysis

**How each activation affects gradient flow: **

ReLU and GELU allow for better gradient flow at positive values, while Tanh and Softsign may cause slowness due to saturation.  
  
**Which activations risk vanishing gradients: **  
This problem is evident in Tanh because its values ​​are confined between -1 and 1, resulting in very small derivatives at large values.  
  
**Why does GELU perform well in Transformer architectures? **  
Because it combines the characteristics of ReLU with stochasticity, making it more efficient at handling complex data.  
  
**Why does ReLU remain preferred for many MLP and CNN models?**  
Because of its extreme computational efficiency and speed in training deep networks without significant mathematical complexity.



## 5\. Key Takeaway

ReLU wins on early convergence speed due to its simple gradient switch, but GELU wins on final capacity and representation depth because its smooth curvature prevents neuron death.





