# Task02_CustomDigit



## 1\. Objective

To evaluate the model I will testing it on an external, hand-drawn digit, and analyzing how advanced preprocessing (cropping, dilation, centering, and scaling) aligns external data with the training distribution.

## 2\. Code Used

```python
# سوف استخدم مكتبة open cv and numpy
# لمعالجة الصورة و ادخالها للمودل
import cv2
import numpy as np
import matplotlib.pyplot as plt
# تحميل الصورة   
img_path = '/content/drive/MyDrive/AI System Diploma/mnsit_project/results/prediction/five.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap="gray") # عرض الصورة
# الرقم يقع في أعلى الصورة تماماً وحجمه صغير جداً بالنسبة للمساحة الكلية

# ايش بدي اعمل للصورة ؟؟؟

# 1. جعل الرقم في منتصف الصورة 
#2. ضبط سماكة الخط
# 3. ضبط الحجم و الابعاد كما الصور التي تدرب عليها المودل 
# 4. scalling
#توسيط الرقم
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    digit_crop = img[y:y+h, x:x+w]
    # تسميك الخط ليتناسب مع المودل 
    kernel = np.ones((5,5), np.uint8)
    digit_thick = cv2.dilate(digit_crop, kernel, iterations=2)
    pad = max(w, h) // 4
    img_processed = cv2.copyMakeBorder(digit_thick, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
else:
    img_processed = img
    
 # عمل حجم الصورة 28* 28
img_resized = cv2.resize(img_processed, (28, 28))
# scalling
custom_sample = img_resized.astype('float32') / 255.0
plt.imshow(img_resized, cmap='gray')
prediction = model.predict(custom_sample.reshape(1, 28, 28))
pred_label = np.argmax(prediction)
print(f"Predicted Label: {pred_label}")

# عرض النتيجة
plt.imshow(img_resized, cmap='gray')
plt.title(f"Prediction: {pred_label}")
plt.show()
```

## 3\. Results

- **Custom Digit Drawn:** 5



![five.png](<./attachments/619489a0f51fe2fa-five.png>)

- **Model Prediction Label:** 5

My custom Image after preprocess::



![five after preprocess.png](<./attachments/1ab27003109c0dd4-five after preprocess.png>)

## 4\. Short Analysis

The original image you uploaded had a very strong distribution shift because the number was small, the font was thin, and it was centered at the top of the image. If the model had seen it like that, it would have made a mistake.  
  
However, since we processed it by cropping (contouring), thickening the font (cv2.dilate), and centering it, we eliminated the distribution shift and made the image match the MNIST data the model was trained on. That's why the model classified it correctly and confidently.

## 5\. Key Takeaway

Advanced spatial preprocessing bridges the gap of distribution shifts, ensuring that external test features map correctly onto the representations learned by the neural network.





