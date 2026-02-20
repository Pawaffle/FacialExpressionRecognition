## Analysis report

# 1. Introduction

The goal of this project was to explore the Transformer architecture from two different perspectives. In the first part, we trained our own language model from scratch using Finnish literary text. In the second part, we used the pretrained OpenAI Whisper model for speech recognition. The main objective was to understand differences in resource requirements and performance when comparing a small self‑trained model with a massive pretrained model.

# 2. Text generator (Seitsemän veljestä)

## 2.1 Dataset and preprocessing

For training the text generator we used Aleksis Kivi’s *Seitsemän veljestä* ([Project Gutenberg EBook #11940](https://www.gutenberg.org/ebooks/11940)). The text was chosen because of its clear and recognizable style (archaic Finnish), which makes qualitative evaluation of the model’s outputs easier.

- **Cleaning:** Removed extra metadata, tables of contents, and Gutenberg license text.
- **Tokenization:** Used SentencePiece, which splits words into subword units. This is more effective for morphologically rich languages like Finnish than pure word‑level tokenization.

## 2.2 Model architecture and hyperparameters

We used a decoder‑only Transformer architecture, which is typical for generative models (GPT‑style). We tested several configurations of three Transformer models of different sizes:

| Model  | embed_dim | num_heads | ff_dim | num_layers | Training time | Param count |
|--------|-----------|-----------|--------|------------|--------------:|------------:|
| Small  | 128       | 4         | 300    | 2          | ~20 min       | ~2.8M       |
| Medium | 192       | 4         | 384    | 2          | ~24 min       | ~4.4M       |
| Large  | 256       | 8         | 768    | 3          | ~37 min       | ~7.1M       |

All models were trained for 50 epochs with batch size = 128.

## 2.3 Results and observations

**Text quality**

- The **small model** produced largely unintelligible, confusing text.
- The **medium‑sized model** generated the clearest and most coherent sentences, as long as the temperature was kept between 0.5 and 0.8.
- The **large model** was the most creative, but with higher temperature it often produced many unnecessary words and overly long, rambling text.

**Conclusion:** For short passages the text was surprisingly grammatical. For longer passages (over ~50 words), the “red thread” tended to disappear, with the model repeating themes or changing topic in a somewhat random way.

**Effect of the temperature parameter**

| Temperature | Text characteristics                                                        |
|-------------|-----------------------------------------------------------------------------|
| 0.5         | Minimal vocabulary, unclear text, grammatically incorrect.                 |
| 0.8         | Best balance for larger models, but worsened performance for smaller ones. |
| 1.0         | Richer language with some irregularities.                                  |
| 1.2         | Started inventing words and reduced readability.                           |

# 3. Speech recognition (Whisper)

## 3.1 Implementation

In the second part we used OpenAI’s Whisper models (`openai/whisper-base` and `openai/whisper-large-v3`) via the Hugging Face Transformers library. We evaluated the models on challenging audio clips containing slang, colloquial speech, and dialectal expressions.

## 3.2 Comparing Base vs. Large

We compared two model sizes. The results clearly illustrated the “scaling laws” phenomenon: increasing model size significantly improved performance, especially in terms of contextual understanding.

| Aspect              | Whisper Base                                                               | Whisper Large                                                              |
|---------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Resource usage      | Lightweight, ran reasonably fast even on CPU.                             | Required a GPU (e.g., T4/L4) for acceptable speed. High VRAM usage.       |
| Word accuracy       | Good on standard written language, weak on colloquial speech.             | Excellent, recognized slang and speech in noisy conditions.               |
| Translations        | Understood content but made some word choice errors and repetitions.      | Produced near professional‑level translations.                             |
| Context understanding | Lost the topic in dialogue, failed to interpret slang correctly.        | Kept track of topics and individual terms much more reliably.             |

In addition, the *Large* model could follow a dialogue between two speakers, whereas the smaller model often lost track of who was speaking or what previous utterances referred to.

# 4. Summary and reflection

## 4.1 Scalability of Transformers

This project concretely demonstrated the difference between a small, self‑trained model and a large foundation model.

- **Custom text generator:** Although our model learned the style and vocabulary of *Seitsemän veljestä*, it did not remain coherent over longer stretches of text. This is likely due to limited data (only one book) and a relatively small number of parameters. A larger model trained on the same text produced noticeably more coherent outputs.
- **Whisper:** Whisper’s performance also depended heavily on model size. The larger variant performed very well, but relies on massive training data (hundreds of thousands of hours), which is not feasible to train locally. This demonstrates the power of Transformer architectures: when data and compute are scaled up, the model learns to generalize instead of memorizing.

## 4.2 Conclusions

Training your own model is an excellent way to understand how the architecture works under the hood. For practical applications, however, fine‑tuning or directly using a pretrained model is almost always more effective than training from scratch, unless the task is extremely narrow and specific.

Our experiments also support the idea that “bigger is better” for Transformers. The largest performance differences appeared in challenging speech recognition scenarios (noise, dialects), while for simple, clean written language, even smaller models can be sufficient and more cost‑effective.
