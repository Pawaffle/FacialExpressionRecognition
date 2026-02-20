## Analysis report

## Tree species classification with three CNN models

### 1. Introduction

The goal of this project was to build and compare three different convolutional neural network (CNN) based approaches for classifying four **tree species** (**spruce, yew, pine, thuja**). The main focus was to compare the effectiveness of transfer learning (**VGG16**) against training a model from scratch (**CNN From Scratch**), especially in the context of a **small dataset**.

**Tools and dataset:** Python, Keras, TensorFlow. A self-collected [dataset](https://drive.google.com/drive/folders/1jCPEcTL_sVCKKOERwnb0KyamQ8u51Hk0?usp=sharing) of 600 images was used, with 150 images per class. The data was split into 70/15/15 % for training, validation, and test sets. Original images were standardized to a size of 800×800 pixels, and then resized to **224×224** for model input. The images capture subtle visual differences between species, such as the sharp needles of spruce, the flatter and darker leaves of yew, and the scale-like structure of thuja.

**Preprocessing:** Normalization, batching, and data augmentation (random flip, rotation, and zoom) on the training data.

**Models:**

- **Model 1:** Custom CNN from scratch  
- **Model 2:** VGG16 as feature extractor (Feature Extraction)  
- **Model 3:** Fine‑tuned VGG16 (Fine‑Tuning)

### 2. Model architectures and comparison

#### 2.1 Model 1: CNN From Scratch

This is a lightweight hierarchical model: a stack of convolution and pooling layers that learns tree features from scratch.

- **Architecture:** Data Augmentation → Rescaling → 5 Conv2D / MaxPooling2D blocks → Flatten → Dropout(0.3) → Dense(4, softmax).  
- **Optimization:** Adam optimizer with a very small learning rate (0.0003).

#### 2.2 Model 2: VGG16 Feature Extraction

This model uses the VGG16 convolutional base as a fixed feature extractor. Images are passed through the base once, and the resulting static feature maps (7×7×512) are stored.

- **Architecture:** On top of the frozen features, a simple dense classifier is trained: Flatten → Dense(256) → Dropout(0.5) → Dense(4, softmax).  
- **Optimization:** RMSprop optimizer.

#### 2.3 Model 3: VGG16 Fine‑Tuning

This is a two‑stage approach. First, a new dense head is trained on top of the frozen VGG16 base (Phase 1). Then the **last layers of VGG16 are unfrozen** and fine‑tuned with a very small learning rate (10⁻⁵) (Phase 2).

- **Architecture:** Full end‑to‑end model, including Data Augmentation before the VGG16 base and the new classification head.

#### 2.4 Quantitative comparison

| Model | Approach                | Trainable params (approx.) | Test accuracy (%) | Generalization |
|------:|-------------------------|----------------------------:|------------------:|----------------|
| 1     | Custom CNN from scratch | 1.08 million               | ~90.2             | Weak           |
| 2     | VGG16 Feature Extraction| 6.4 million                | ~97.8             | Moderate       |
| 3     | VGG16 Fine‑Tuning       | 13.5 million               | 100               | Excellent      |

Model 1 is the lightest model but achieves the lowest accuracy. Models 2 and 3 based on VGG16 are much heavier, but reach nearly perfect accuracy, highlighting the **strength of transfer learning** on small datasets.

### 3. Learning behavior and results

#### 3.1 Summary table

| Model                 | Strength                                                                                  | Challenge                                                                                       | Key outcome                              |
|-----------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------|
| VGG16 Fine‑Tuning     | Best performance by combining ImageNet features with domain‑specific fine‑tuning.        | More complex, two‑phase training process.                                                       | Best generalization, perfect accuracy    |
| VGG16 Feature Extraction | Fast training and high accuracy with relatively little effort.                        | Validation loss sometimes unstable, indicating possible overfitting of the new head to static features. | Very high accuracy (~98%)                |
| Custom CNN (scratch)  | Lightweight and trained specifically on the tree dataset, shows CNN effectiveness from scratch. | Small dataset leads to noisy validation metrics and limited capacity to learn nuanced patterns. | Reasonably good test accuracy (~90%)     |

#### 3.2 Error analysis and generalization

**Misclassified images (hypothesis):** The models most likely confused visually similar tree species, such as spruce and young pine (dense, dark needle structures).

**Why did the custom CNN (Model 1) generalize at all?**  
Even though it had to learn everything from scratch, its reasonable performance (~90%) comes from the ability of convolution layers to learn hierarchical features (edges → needle density) combined with extra training examples created by data augmentation.

**Why does transfer learning win?**  
VGG16 (Models 2 and 3) leverages general features learned from ImageNet that transfer well to recognizing textures and shapes of trees. This reduces the amount of data and compute needed and helps avoid overfitting during feature learning.

### 4. Conclusions and future work

#### 4.1 Conclusions

The results clearly show that **Model 3 (Fine‑tuned VGG16)** is the best architecture for this type of small‑data classification task. Its ability to adapt general ImageNet features to the specific tree species domain leads to the strongest generalization and highest test accuracy. The good performance of Model 1 (~90%) still demonstrates that even relatively small CNNs can be effective when designed carefully.

#### 4.2 Experimental aspects and improvement ideas

Key experimental choices:

- **Hyperparameters:** The **very small learning rates for Adam (0.0003 and 10⁻⁵)** were critical to keep training stable and enable effective fine‑tuning.  
- **Data augmentation:** Including DA in **Model 1** and **Model 3** was essential for generalization.

Potential directions:

- Increase validation set size or use cross‑validation to obtain more reliable and less noisy validation metrics, especially for **Model 1**.  
- Try alternative pretrained backbones (e.g., **ResNet**) to potentially push performance even further.

#### 4.3 Use of AI tools

AI tools were used to help design the structure of this report, correct language and formatting, propose model optimization ideas, and assist with extraction and selection of test images.

### 5. Scope of the work

To improve the original performance of Model 1, a more advanced preprocessing strategy was tested: the original images were split into smaller patches, creating a much larger dataset of 40,064 patches to give the model more training data.

From this pool, a final [dataset of 8,000 images](https://drive.google.com/drive/folders/1i2d9kKg5HNRoODD7JdYe-w90E6Y7SusU?usp=sharing) (2,000 per class) was assembled by selecting only the highest‑quality patches. The model was then retrained according to the Model 1 architecture on this optimized dataset. This approach was highly effective: test accuracy increased to around 98 %, and the loss decreased significantly.

Despite these strong results, the patch‑based Model 1 remained slightly behind the pretrained VGG16 models (Models 2 and 3). Even though the gap was small, the pretrained models showed more reliable behavior, underlining the advantages of transfer learning compared to a carefully optimized scratch model, even with a larger and cleaned dataset.
