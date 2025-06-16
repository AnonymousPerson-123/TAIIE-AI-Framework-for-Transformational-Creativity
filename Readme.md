# Innovation-Driven Research Topic Generator

This project generates and validates novel research topics using a combination of transformer-based language models, human feedback simulation, and reinforcement learning with RLHF (Reinforcement Learning with Human Feedback).

---

## 📁 Files

- **`main.py`** – The core pipeline that trains a transformer model to generate research topics, refines them using RLHF, and evaluates them.
- **`validity_checker.py`** – Trains a BERT-based utility model to score research topics on validity, novelty, coherence, and domain relevance.

---

## 📌 Features

- Supervised training on preexisting and synthetic research topics
- Transformer-based topic generator with positional encoding
- RLHF interface with simulated human feedback
- Dynamic concept diversity and novelty tracking
- Paradigm shift detection via Kuhnian momentum
- Bayesian first-word selector for guided seeding
- Quantized BERT-based utility model for evaluation

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/research-topic-generator.git
cd research-topic-generator
```

### 2.Install Dependencies
```bash
pip install torch transformers pandas scikit-learn numpy
```
###3. Prepare Data
Place your dataset in the same directory and name it:
```
preexisting_research_topics.csv
```

###4.🧠 Train the Utility Model
Run the following to train and save the quantized topic evaluation model:
```
python validity_checker.py

```
This will generate:

.topic_utility_model.pt

.topic_validator_tokenizer/

###5.🎓 Run the Full Pipeline
Once the utility model is trained, run:
```
bash
python main.py
```
This will:

Train the topic generator

Perform reinforcement learning

Log generated topics and feedback

Save results to innovation_history.json and model to innovation_engine.pt


🛠️ Customization
Adjust CONFIG in main.py to change:

.RLHF frequency

.Reward weights

.Diversity weight

.Paradigm shift sensitivity

📄 License
MIT License
