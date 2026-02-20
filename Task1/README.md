## Analysis report, CIFAR-10 image classification

### 1. Introduction

The goal of this project was to implement CIFAR-10 image classification with two neural network architectures: a **fully connected network (FCN)** and a **convolutional neural network (CNN)**. The aim was to compare model architecture, performance, and learning behavior, and to analyze why CNNs usually work better for image data. The tools used were **Python, Keras, TensorFlow**, and the **CIFAR-10** dataset (60,000 color images, 32×32×3, 10 classes). Preprocessing included **normalization (0–1)**, **one-hot encoding**, and **data augmentation** (for the CNN).

**FCN architecture:**

Input (32×32×3) → RandomFlip → RandomTranslation → Flatten → Dense(2048) → BatchNormalization → ReLU → Dropout(0.5) → Dense(1024) → BatchNormalization → ReLU → Dropout(0.5) → Dense(512) → BatchNormalization → ReLU → Dropout(0.4) → Dense(10, softmax)

**CNN architecture:**

**Convolution block 1:** Conv2D(32, 3×3, ReLU, padding="same") → BatchNormalization → Conv2D(32, 3×3, ReLU) → MaxPooling(2×2) → Dropout(0.3)

**Convolution block 2:** Conv2D(64, 3×3, ReLU, padding="same") → BatchNormalization → Conv2D(64, 3×3, ReLU) → MaxPooling(2×2) → Dropout(0.3)

**FCN head:** Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(num_classes, softmax)

### 2. Model comparison

#### 2.1 FCN

The FCN flattens each image into a vector (3072 values) and learns connections directly from pixels. The spatial structure of the image is lost, which makes feature learning more difficult.

#### 2.2 CNN

The CNN uses convolutional layers that learn edges, shapes, and textures hierarchically. Pooling reduces data size and helps prevent overfitting.

| Model | Parameters | Accuracy (test) | Test loss | Training time |
|-------|-----------:|----------------:|----------:|--------------:|
| FCN   | 8,932,362  | 64.0 %          | 1.1535    | 346.5 s / 100 epochs |
| CNN   | 1,251,242  | 82.5 %          | 0.5785    | 239.4 s / 50 epochs  |

The CNN is **7.7 million parameters lighter**, yet **faster and more accurate**, which illustrates how effective convolutional layers are at learning spatial information.

### 3. Learning behavior and results

The CNN’s learning curves were smoother and showed less overfitting than those of the FCN. Dropout and Batch Normalization improved the stability of both models, but the CNN architecture itself was the key factor for better generalization.

**Error analysis:**

- The FCN confused visually similar classes (e.g., cat ↔ dog, truck ↔ automobile).
- The CNN made fewer mistakes but still struggled with images where the background dominated the object.

**Why the CNN wins:**  
The CNN learns **hierarchical and local features**, shares weights efficiently, and requires less computation. It does not just “memorize” images but **captures their structure**.

### 4. Conclusions and future work

#### 4.1 Conclusions

The results show that the **CNN** is a much more effective architecture for image classification than the **FCN**. The CNN reached **82.5 %** accuracy without significant overfitting, whereas the FCN stayed at **64 %** with a much larger number of parameters. The CNN confirmed the theoretical assumption that **the spatial structure of images** is crucial for successful learning.

#### 4.2 Experimental ideas and extensions

Experiments with **Dropout**, **Batch Normalization**, and **Data Augmentation** improved performance and reduced overfitting for both models.

**Possible future improvements:**

- **Transfer learning:** e.g., VGG16, ResNet50, or MobileNetV2 → potential accuracy above 90 %
- **Deeper CNNs:** more convolutional layers and filters
- **Richer data augmentation:** zoom, brightness and color variations
- **Better optimization:** AdamW or learning rate scheduling

#### 4.3 Use of AI tools

AI was used to help design the report structure, correct language and formatting, and suggest ideas for model optimization.
