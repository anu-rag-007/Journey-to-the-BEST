# Journey to the BEST

> **A hands-on journey from Machine Learning fundamentals to Deep Learning, Generative AI, and research-oriented AI systems.**

This repository documents my progression as a **B.Tech CSE (AI & ML) student**, with a focus on learning by building rather than only following theory.

What started with basic Python, data manipulation, and classical machine learning gradually evolved into neural networks, computer vision, Transformers, LLMs, EEG-based modeling, and research-oriented experimentation.

The goal of this repository is simple:

**Learn → Implement → Experiment → Measure → Understand → Build something real.**

---

## About This Repository

`Journey-to-the-BEST` is my chronological ML/AI learning log.

Each week represents a focused stage of learning, with notebooks, experiments, implementations, results, and observations.

I intentionally include implementations **from scratch** wherever possible because understanding the underlying mathematics and mechanics is as important to me as using modern frameworks.

The repository is continuously evolving.

---

## Learning Progression

```text
Python & Data Handling
        ↓
NumPy & Mathematical Foundations
        ↓
Statistics & Probability
        ↓
Classical Machine Learning
        ↓
Neural Networks from Scratch
        ↓
PyTorch
        ↓
CNNs & Computer Vision
        ↓
Transfer Learning & Object Detection
        ↓
LSTMs, Attention & Transformers
        ↓
LLMs & Generative AI
        ↓
EEG / Neurotechnology Research
        ↓
PROJECT-07
        ↓
LUCID: Reality?
```

---

# Weekly Roadmap

| Week   | Focus                    | Highlights                                                                  |
| ------ | ------------------------ | --------------------------------------------------------------------------- |
| **01** | ML Foundations           | Python, data manipulation, Titanic, EDA                                     |
| **02** | Mathematical Foundations | NumPy, arrays, broadcasting, matrices, statistics, linear regression        |
| **03** | Classical ML             | Probability, distributions, Naive Bayes, Logistic Regression, pipelines     |
| **04** | Neural Networks          | Forward propagation, backpropagation, MNIST, NumPy → PyTorch                |
| **05** | Computer Vision          | Convolution, CNN architecture, MNIST, CIFAR-10                              |
| **06** | Advanced Vision          | Transfer learning, fine-tuning, Grad-CAM, object detection                  |
| **07** | Modern Deep Learning     | CNN-LSTM, attention, Transformers, Hugging Face, LLM APIs                   |
| **08** | EEG & Research           | EEG Transformers, evaluation, cross-subject validation, hardware pipeline   |
| **09** | Research Development     | Experiments, publication figures, reinforcement learning, research planning |
| **10** | Generative AI            | Diffusion models, Stable Diffusion, CLIP, NeRF, 3D scene generation         |

---

## Selected Milestones

### Neural Networks from Scratch

Implemented neural networks without relying entirely on high-level frameworks.

**MNIST results:**

* NumPy implementation: **97.88% test accuracy**
* PyTorch implementation: **98.47% test accuracy**
* Implemented forward propagation and backpropagation
* Experimented with Adam, dropout and learning-rate scheduling

---

### Computer Vision

Progressed from understanding convolution mathematically to building complete CNN pipelines.

Topics include:

* Convolution from scratch
* CNN architecture
* MNIST classification
* CIFAR-10 classification
* Transfer learning
* Fine-tuning
* Grad-CAM
* Object detection

---

### Modern Deep Learning

Moved beyond conventional CNN architectures into modern sequence and language models.

Explored:

* CNN-LSTM architectures
* Attention mechanisms
* Transformers
* Hugging Face
* LLM APIs
* Research-oriented AI assistants

---

## From Learning to Research

The most important transition in this repository happened when the learning exercises began contributing directly to a larger research-oriented project.

The work around EEG, sleep-stage classification, evaluation, and real-time systems eventually evolved into:

### [PROJECT-07 — LUCID: Reality?](https://github.com/anu-rag-007/PROJECT-07)

PROJECT-07 is a separate repository focused on developing the first technical component of **LUCID: Reality?**

The current pipeline explores:

```text
EEG Signal
    ↓
Preprocessing
    ↓
Sleep Stage Classification
    ↓
REM Detection
    ↓
Decision / Safety Logic
    ↓
Haptic Stimulation
```

The current prototype uses the Sleep-EDF dataset and explores LSTM-based sleep-stage classification, REM detection and Android haptic feedback. The project currently reports **76.85% overall accuracy** and **84% REM recall** on its documented evaluation.

The repository also documents the architecture, dataset configuration, experiments and current limitations.

**PROJECT-07:**
https://github.com/anu-rag-007/PROJECT-07

---

# Current Technical Interests

My current interests are moving toward the intersection of:

* Machine Learning
* Deep Learning
* Computer Vision
* Generative AI
* Large Language Models
* Time-Series Modeling
* EEG Signal Processing
* Brain-Computer Interfaces
* Neurotechnology
* Human-AI Interaction
* Research-oriented AI systems

---

# Philosophy

I don't want this repository to be a collection of copied tutorials or isolated notebooks.

I want it to show **how my understanding changes over time**.

That means some experiments may be imperfect, some models may perform poorly, and some approaches may eventually be discarded.

Those results are still valuable.

> **A failed experiment is still progress if I understand why it failed.**

---

# Repository Structure

```text
Journey-to-the-BEST/
│
├── Week 1/
├── Week 2/
├── Week 3/
├── Week 4/
├── Week 5/
├── Week 6/
├── Week 7/
├── Week 8/
├── Week 9/
├── Week 10/
│
├── .gitignore
├── .gitattributes
└── README.md
```

Each week contains notebooks, experiments, datasets or supporting material relevant to that stage of the journey.

Large datasets and generated artifacts are intentionally excluded where appropriate.

---

# Tools & Technologies

### Languages

* Python

### Data & Scientific Computing

* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy

### Machine Learning

* Scikit-learn

### Deep Learning

* PyTorch

### Computer Vision

* CNNs
* Transfer Learning
* YOLO
* Grad-CAM

### NLP / Generative AI

* Transformers
* Hugging Face
* LLM APIs
* Diffusion Models
* CLIP
* NeRF

### Neurotechnology

* MNE-Python
* EEG
* Sleep-EDF
* LSTM
* EEG Transformers
* Brain-Computer Interfaces

---

# What Comes Next?

The learning journey does not end at Week 10.

The next stage is to spend less time only learning individual concepts and more time **combining them into complete systems**.

Current direction:

```text
Learn individual concepts
        ↓
Build small experiments
        ↓
Combine multiple concepts
        ↓
Build complete systems
        ↓
Evaluate scientifically
        ↓
Document results
        ↓
Research / Iterate
```

PROJECT-07 represents the beginning of this transition.

---

# Related Project

### LUCID: Reality?

**PROJECT-07** is being developed as a technical foundation for a much larger long-term research idea.

The broader vision explores the intersection of:

**AI + EEG + Sleep + Neurotechnology + Immersive Computing**

The project is experimental and research-oriented, and its claims are treated as hypotheses to be tested rather than established scientific facts.

→ **[Open PROJECT-07](https://github.com/anu-rag-007/PROJECT-07)**

---

# Author

**Anurag Sharma**

B.Tech CSE — Artificial Intelligence & Machine Learning

Learning AI by building, experimenting, breaking things, and understanding why they work.

---

## Repository Status

**Active — continuously evolving**

This repository will continue to change as new concepts, experiments, projects and research directions are explored.

---

> **The goal isn't to finish the roadmap.
> The goal is to become capable of building what I can imagine.**
