import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter, deque
import random
import json
import os
import asyncio
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# 0. CONFIGURATION
# ======================
CONFIG = {
    "rlhf_interval": 5,          # Collect human feedback every N batches
    "feedback_probability": 0.3,  # Probability of requesting feedback per selected topic
    "kuhnian_momentum": 0.95,     # Momentum factor for paradigm shift detection
    "diversity_semantic_weight": 0.4,  # Weight for semantic diversity
    "reward_weights": [0.5, 0.2, 0.2, 0.1],  # Validity, Novelty, Coherence, Relevance
    "human_weights": [0.25, 0.35, 0.25, 0.15],  # Novelty, Usefulness, Clarity, Elegance
    "rlhf_weights": [0.6, 0.3, 0.1],  # Human, Utility, Conciseness
    "embedding_cache_size": 5000,
    "min_feedback_topics": 3
}

# ======================
# 1. UTILITY MODEL (Validity Checker with Quantization)
# ======================
class QuantizedTopicUtilityModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)

# Global initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
utility_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
utility_model = QuantizedTopicUtilityModel().to(device)

# Apply dynamic quantization
utility_model = torch.quantization.quantize_dynamic(
    utility_model, {nn.Linear}, dtype=torch.qint8
)

# Load pre-trained model
if os.path.exists('topic_utility_model.pt'):
    utility_model.load_state_dict(torch.load('topic_utility_model.pt', map_location=device))
    utility_model.eval()
    print("Loaded quantized utility model")
else:
    print("Warning: Utility model not found. Please train validity_checker.py first")
    exit(1)

async def get_utility_scores_batch(topics):
    """Batch processing of utility scores with async"""
    inputs = utility_tokenizer(
        topics,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = utility_model(input_ids, attention_mask)
        scores = torch.sigmoid(outputs).cpu().numpy()
    
    return [{
        'validity': float(s[0]),
        'novelty': float(s[1]),
        'coherence': float(s[2]),
        'domain_relevance': float(s[3])
    } for s in scores]

# ======================
# 2. RLHF INTERFACE (Simulated)
# ======================
class HumanFeedbackSimulator:
    def __init__(self):
        self.feedback_db = {}
        
    def get_feedback(self, topic):
        """Simulate human feedback with realistic patterns"""
        if topic in self.feedback_db:
            return self.feedback_db[topic]
            
        # Simulate human-like ratings based on topic characteristics
        words = topic.split()
        uniqueness = len(set(words)) / max(1, len(words))
        complexity = min(1.0, len(words) / 15)
        
        feedback = {
            'novelty': np.clip(0.2 + 0.7 * uniqueness + 0.1 * random.gauss(0, 0.1), 0, 1),
            'usefulness': np.clip(0.6 - 0.3 * complexity + 0.2 * random.gauss(0, 0.1), 0, 1),
            'clarity': np.clip(0.8 - 0.4 * complexity + 0.2 * random.gauss(0, 0.1), 0, 1),
            'elegance': np.clip(0.3 + 0.5 * uniqueness + 0.2 * random.gauss(0, 0.1), 0, 1)
        }
        
        self.feedback_db[topic] = feedback
        return feedback

# Global RLHF simulator (replace with real interface)
rlhf_interface = HumanFeedbackSimulator()

# ======================
# 3. DATA PREPARATION
# ======================
class ResearchTopicDataset(Dataset):
    def __init__(self, topics, tokenizer, max_length=15):
        self.tokenizer = tokenizer
        self.sequences = [self.tokenizer.encode(topic) for topic in topics]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        pad_len = max(0, self.max_length - len(seq))
        return torch.tensor(seq + [0] * pad_len, dtype=torch.long)

class TopicTokenizer:
    def __init__(self, topics):
        self.vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        word_freq = Counter()
        
        for topic in topics:
            words = topic.split()
            word_freq.update(words)
            
        # Add to vocabulary (minimum 2 occurrences)
        for word, count in word_freq.items():
            if count >= 2 and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        words = text.split()
        tokens = [2]  # Start with <sos>
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(1)  # <unk>
        tokens.append(3)  # <eos>
        return tokens
    
    def decode(self, tokens):
        words = []
        for token in tokens:
            if token in [0, 2, 3]:  # Skip special tokens
                continue
            words.append(self.inv_vocab.get(token, '<?>'))
        return " ".join(words)

# ======================
# 4. TRANSFORMER MODEL
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ResearchTopicGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        if src_mask is None:
            src_mask = self.generate_square_mask(src.size(1)).to(src.device)
        
        output = self.transformer(src, src_mask)
        return self.fc(output)
    
    def generate_square_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# ======================
# 5. BAYESIAN INNOVATION SEEDER (Enhanced)
# ======================
class BayesianFirstWordSelector:
    def __init__(self, initial_topics, embedding_model):
        self.word_stats = defaultdict(lambda: {
            'frequency': 0,
            'attempts': 0,
            'successes': 0,
            'embedding': None,
            'entropy': 0
        })
        
        self.embedding_model = embedding_model
        self.embedding_cache = {}
        self._initialize_stats(initial_topics)
        self.base_lr = 0.3
        self.lr_decay = 0.95
        
    def _get_embedding(self, word):
        """Get embedding with caching"""
        if word in self.embedding_cache:
            return self.embedding_cache[word]
            
        with torch.no_grad():
            inputs = utility_tokenizer(word, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        # Manage cache size
        if len(self.embedding_cache) >= CONFIG["embedding_cache_size"]:
            oldest_key = next(iter(self.embedding_cache))
            self.embedding_cache.pop(oldest_key)
            
        self.embedding_cache[word] = embedding
        return embedding
        
    def _initialize_stats(self, topics):
        """Initialize word statistics from existing topics"""
        for topic in topics:
            words = topic.split()
            if words:
                first_word = words[0]
                self.word_stats[first_word]['frequency'] += 1
        
        total_topics = len(topics)
        for word, stats in self.word_stats.items():
            stats['embedding'] = self._get_embedding(word)
            p_word = stats['frequency'] / total_topics
            stats['entropy'] = -p_word * math.log(p_word + 1e-12)
    
    def update(self, word, utility_scores):
        """Bayesian update with adaptive learning rate"""
        weights = CONFIG["reward_weights"]
        composite = sum(w * utility_scores[k] for w, k in zip(
            weights, ['validity', 'novelty', 'coherence', 'domain_relevance']
        ))
        
        attempts = self.word_stats[word]['attempts']
        learning_rate = self.base_lr * (self.lr_decay ** attempts)
        
        self.word_stats[word]['attempts'] += 1
        self.word_stats[word]['successes'] += composite * learning_rate
    
    def sample_word(self, temperature=0.8):
        """Sample first word using Bayesian distribution"""
        words = list(self.word_stats.keys())
        if not words:
            return random.choice(["Bayesian", "Neural", "Quantum"])
            
        log_probs = []
        total_freq = sum(s['frequency'] for s in self.word_stats.values())
        
        for word in words:
            stats = self.word_stats[word]
            prior = stats['frequency'] / total_freq
            utility = stats['successes'] / (stats['attempts'] + 1e-5)
            information = math.exp(stats['entropy'])
            posterior = prior * utility * information
            log_probs.append(math.log(posterior + 1e-12) / temperature)
        
        max_log = max(log_probs)
        probs = [math.exp(lp - max_log) for lp in log_probs]
        total = sum(probs)
        probs = [p/total for p in probs]
        
        return np.random.choice(words, p=probs)
    
    def get_word_innovation_potential(self, word):
        """Compute innovation potential metric H(w) × T_mod(v)"""
        stats = self.word_stats.get(word)
        if not stats:
            return 0.0
        
        H_w = stats['entropy']
        embeddings = [s['embedding'] for s in self.word_stats.values() if s['embedding'] is not None]
        
        if not embeddings:
            return H_w
            
        avg_embedding = np.mean(embeddings, axis=0)
        T_mod = np.linalg.norm(stats['embedding'] - avg_embedding)
        return H_w * T_mod

# ======================
# 6. KUHNIAN MONITOR (Enhanced)
# ======================
class KuhnianMonitor:
    def __init__(self, window_size=50, threshold=0.15):
        self.novelty_history = []
        self.window_size = window_size
        self.threshold = threshold
        self.paradigm_shifts = 0
        self.momentum = CONFIG["kuhnian_momentum"]
        
    def track_novelty(self, topic):
        """Compute novelty score for a topic"""
        words = topic.split()
        unique_words = len(set(words))
        return unique_words / len(words) if words else 0
    
    def add_topic(self, topic):
        novelty = self.track_novelty(topic)
        self.novelty_history.append(novelty)
        return novelty
    
    def check_paradigm_shift(self):
        """Check if we need a paradigm shift with momentum"""
        if len(self.novelty_history) < 2 * self.window_size:
            return False
        
        recent = self.novelty_history[-self.window_size:]
        previous = self.novelty_history[-2*self.window_size:-self.window_size]
        
        recent_avg = np.mean(recent)
        previous_avg = np.mean(previous) * self.momentum  # Apply momentum
        
        if recent_avg < previous_avg - self.threshold:
            self.paradigm_shifts += 1
            return True
        return False
    
    def reset_history(self):
        """Reset after paradigm shift"""
        self.novelty_history = []

# ======================
# 7. CONCEPT DIVERSITY TRACKER (Enhanced)
# ======================
class ConceptDiversityTracker:
    def __init__(self, window_size=100, decay_factor=0.95):
        self.concept_counts = Counter()
        self.concept_embeddings = {}
        self.concept_history = deque(maxlen=window_size)
        self.decay_factor = decay_factor
        self.novel_concept_bonus = 0.2
        self.avg_embedding = None
        
    def _get_concept_embedding(self, concept):
        """Get or compute concept embedding"""
        if concept not in self.concept_embeddings:
            with torch.no_grad():
                inputs = utility_tokenizer(concept, return_tensors='pt').to(device)
                outputs = utility_model.bert(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            self.concept_embeddings[concept] = embedding
        return self.concept_embeddings[concept]
        
    def update(self, topic):
        """Update concept frequencies with decay"""
        # Apply decay to all concepts
        for concept in self.concept_counts:
            self.concept_counts[concept] *= self.decay_factor
        
        # Add new concepts
        concepts = set(topic.split())
        for concept in concepts:
            self.concept_counts[concept] += 1.0
            self._get_concept_embedding(concept)  # Ensure embedding exists
        
        # Update average embedding
        if concepts:
            embeddings = [self.concept_embeddings[c] for c in concepts]
            self.avg_embedding = np.mean(embeddings, axis=0) if self.avg_embedding is None else \
                0.9 * self.avg_embedding + 0.1 * np.mean(embeddings, axis=0)
        
        # Update history
        self.concept_history.extend(concepts)
    
    def get_diversity_score(self, topic):
        """Enhanced diversity with semantic similarity"""
        concepts = set(topic.split())
        if not concepts or self.avg_embedding is None:
            return 0.0
        
        total_count = sum(self.concept_counts.values())
        novelty_scores = []
        
        for concept in concepts:
            # Frequency-based novelty
            concept_freq = self.concept_counts.get(concept, 0) / total_count
            
            # Semantic similarity penalty
            embedding = self.concept_embeddings.get(concept)
            if embedding is not None and self.avg_embedding is not None:
                similarity = cosine_similarity([embedding], [self.avg_embedding])[0][0]
                semantic_penalty = CONFIG["diversity_semantic_weight"] * similarity
            else:
                semantic_penalty = 0
                
            novelty = 1.0 - (concept_freq * (1 + semantic_penalty))
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    def get_diversity_bonus(self, topic):
        """Get reward bonus for using novel concepts"""
        diversity_score = self.get_diversity_score(topic)
        return diversity_score * self.novel_concept_bonus

# ======================
# 8. POLICY GRADIENT TRAINER (RL) WITH RLHF
# ======================
class PolicyGradientTrainer:
    def __init__(self, generator, tokenizer, lr=0.0001):
        self.generator = generator
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(generator.parameters(), lr=lr)
        self.baseline = 0.7
        self.baseline_decay = 0.95
        
    def update(self, topics, rewards, diversity_bonuses, human_feedbacks=None):
        """Update generator with RLHF integration"""
        self.generator.train()
        total_loss = 0
        human_weights = CONFIG["human_weights"]
        rlhf_weights = CONFIG["rlhf_weights"]
        
        for i, (topic, reward, diversity_bonus) in enumerate(zip(topics, rewards, diversity_bonuses)):
            # Compute final reward with RLHF
            if human_feedbacks and i < len(human_feedbacks) and human_feedbacks[i]:
                hf = human_feedbacks[i]
                R_human = sum(w * hf[k] for w, k in zip(
                    human_weights, ['novelty', 'usefulness', 'clarity', 'elegance']
                ))
                conciseness = max(0, 1 - len(topic.split()) / 20)  # Reward concise topics
                adjusted_reward = (
                    rlhf_weights[0] * R_human +
                    rlhf_weights[1] * reward +
                    rlhf_weights[2] * conciseness
                )
            else:
                adjusted_reward = reward + diversity_bonus
            
            # Tokenize and process
            tokens = self.tokenizer.encode(topic)
            src = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
            tgt = torch.tensor([tokens[1:]], dtype=torch.long).to(device)
            
            output = self.generator(src)
            log_probs = F.log_softmax(output, dim=-1)
            
            advantage = adjusted_reward - self.baseline
            loss = 0
            for t in range(output.size(1)):
                nll = F.nll_loss(
                    log_probs[:, t, :],
                    tgt[:, t],
                    ignore_index=0,
                    reduction='none'
                )
                loss += (nll * advantage).mean()
            
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer.step()
        
        # Update baseline
        self.baseline = (self.baseline_decay * self.baseline + 
                         (1 - self.baseline_decay) * np.mean(rewards))
        return total_loss / len(topics)

# ======================
# 9. TOPIC GENERATION & VALIDATION
# ======================
def generate_topic(model, tokenizer, device, seed=None, max_words=20, 
                  temperature=0.7, concept_counts=None, diversity_weight=0.3):
    model.eval()
    
    if seed is None:
        seed = random.choice(["Bayesian", "Neural", "Quantum"])
    
    tokens = tokenizer.encode(seed)
    if tokens and tokens[-1] == 3:
        tokens = tokens[:-1]
    
    word_count = len(seed.split())
    
    # Normalize concept frequencies
    total_count = sum(concept_counts.values()) if concept_counts else 1
    concept_probs = {concept: count/total_count for concept, count in concept_counts.items()}
    
    for _ in range(max_words - word_count):
        with torch.no_grad():
            inputs = torch.tensor([tokens], dtype=torch.long).to(device)
            mask = model.generate_square_mask(inputs.size(1)).to(device)
            
            output = model(inputs, mask)
            logits = output[0, -1, :] / temperature
            
            # Diversity adjustment
            if concept_counts:
                for token_id, token in tokenizer.inv_vocab.items():
                    if token in concept_probs:
                        penalty = concept_probs[token] * diversity_weight
                        logits[token_id] -= penalty
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == 3:
                break
                
            tokens.append(next_token)
            decoded_word = tokenizer.inv_vocab.get(next_token, '<?>')
            word_count += len(decoded_word.split())
            
            if word_count >= max_words:
                break
    
    # Convert to text
    if len(tokens) > 1:
        topic_text = tokenizer.decode(tokens[1:-1])
    else:
        topic_text = seed
        
    words = topic_text.split()[:max_words]
    return " ".join(words)

# ======================
# 10. MAIN INNOVATION PIPELINE WITH RLHF
# ======================
class InnovationPipeline:
    def __init__(self, topics_csv, num_epochs=20):
        self.df = pd.read_csv(topics_csv)
        self.topics = self.df['Research Topic'].tolist()
        self.num_epochs = num_epochs
        
        # Initialize components
        self.tokenizer = TopicTokenizer(self.topics)
        self.dataset = ResearchTopicDataset(self.topics, self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.model = ResearchTopicGenerator(vocab_size=len(self.tokenizer.vocab)).to(device)
        
        self.first_word_selector = BayesianFirstWordSelector(
            self.topics, 
            utility_model.bert
        )
        
        self.kuhnian_monitor = KuhnianMonitor()
        self.diversity_tracker = ConceptDiversityTracker()
        self.device = device
        self.model.to(self.device)
        self.pg_trainer = PolicyGradientTrainer(self.model, self.tokenizer)
        self.innovation_history = []
        self.paradigm_shifts = 0
        self.human_feedback_log = []
    
    def train_model(self):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.dataloader:
                batch = batch.to(self.device)
                src = batch[:, :-1]
                tgt = batch[:, 1:]
                
                output = self.model(src)
                loss = criterion(output.view(-1, output.size(-1)), 
                                tgt.contiguous().view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs} | Loss: {avg_loss:.4f}")
            
            seed = self.first_word_selector.sample_word()
            sample = generate_topic(
                self.model, 
                self.tokenizer, 
                self.device, 
                seed,
                concept_counts=self.diversity_tracker.concept_counts
            )
            innovation_potential = self.first_word_selector.get_word_innovation_potential(seed)
            print(f"Sample Topic: {sample}")
            print(f"Innovation Potential: {innovation_potential:.4f}")
    
    def collect_human_feedback(self, topics):
        """Collect human feedback for selected topics"""
        feedback_batch = []
        selected_indices = []
        
        # Select topics for feedback
        for i, topic in enumerate(topics):
            if random.random() < CONFIG["feedback_probability"]:
                feedback = rlhf_interface.get_feedback(topic)
                feedback_batch.append(feedback)
                selected_indices.append(i)
                self.human_feedback_log.append({
                    "topic": topic,
                    "feedback": feedback
                })
        
        # Ensure minimum feedback
        if len(feedback_batch) < CONFIG["min_feedback_topics"]:
            needed = CONFIG["min_feedback_topics"] - len(feedback_batch)
            for i in random.sample(range(len(topics)), min(needed, len(topics))):
                if i not in selected_indices:
                    feedback = rlhf_interface.get_feedback(topics[i])
                    feedback_batch.append(feedback)
                    selected_indices.append(i)
                    self.human_feedback_log.append({
                        "topic": topics[i],
                        "feedback": feedback
                    })
        
        # Create full feedback list (None where no feedback)
        full_feedback = [None] * len(topics)
        for idx, fb in zip(selected_indices, feedback_batch):
            full_feedback[idx] = fb
        
        return full_feedback
    
    def rl_innovate(self, num_batches, topics_per_batch=10):
        print("\n===== RL INNOVATION PHASE =====")
        
        for batch_idx in range(num_batches):
            topics = []
            utilities = []
            rewards = []
            diversity_bonuses = []
            innovation_potentials = []
            human_feedbacks = None
            
            for _ in range(topics_per_batch):
                seed = self.first_word_selector.sample_word()
                topic = generate_topic(
                    self.model, 
                    self.tokenizer, 
                    self.device, 
                    seed,
                    max_words=20,
                    concept_counts=self.diversity_tracker.concept_counts
                )
                novelty = self.kuhnian_monitor.add_topic(topic)
                self.diversity_tracker.update(topic)
                
                utility = get_utility_scores_batch(topic)  # Note: In production use batch version
                base_reward = sum(
                    w * utility[k] for w, k in zip(
                        CONFIG["reward_weights"],
                        ['validity', 'novelty', 'coherence', 'domain_relevance']
                    )
                )
                diversity_bonus = self.diversity_tracker.get_diversity_bonus(topic)
                total_reward = base_reward + diversity_bonus
                
                self.first_word_selector.update(seed, utility)
                innovation_potential = self.first_word_selector.get_word_innovation_potential(seed)
                
                result = {
                    "topic": topic,
                    "seed": seed,
                    "utility": utility,
                    "base_reward": base_reward,
                    "diversity_bonus": diversity_bonus,
                    "total_reward": total_reward,
                    "innovation_potential": innovation_potential,
                    "paradigm_shift": False
                }
                
                topics.append(topic)
                utilities.append(utility)
                rewards.append(base_reward)
                diversity_bonuses.append(diversity_bonus)
                innovation_potentials.append(innovation_potential)
                self.innovation_history.append(result)
                
                if self.kuhnian_monitor.check_paradigm_shift():
                    self._initiate_paradigm_shift()
                    result["paradigm_shift"] = True
                    self.paradigm_shifts += 1
            
            # Collect human feedback periodically
            if batch_idx % CONFIG["rlhf_interval"] == 0:
                human_feedbacks = self.collect_human_feedback(topics)
                print(f"Collected human feedback for {sum(fb is not None for fb in human_feedbacks)} topics")
            
            # Policy update with RLHF
            loss = self.pg_trainer.update(topics, rewards, diversity_bonuses, human_feedbacks)
            
            # Print batch statistics
            avg_reward = np.mean(rewards)
            avg_diversity = np.mean(diversity_bonuses)
            avg_potential = np.mean(innovation_potentials)
            avg_validity = np.mean([u['validity'] for u in utilities])
            transformational_count = sum(ip > 0.5 for ip in innovation_potentials)
            
            print(f"\nRL Batch {batch_idx+1}/{num_batches}")
            print(f"Avg Reward: {avg_reward:.4f} | Diversity Bonus: {avg_diversity:.4f}")
            print(f"Transformational: {transformational_count}/{topics_per_batch} | Loss: {loss:.4f}")
            print(f"Avg Validity: {avg_validity:.4f}")
            print("Sample Topics:")
            for i, topic in enumerate(topics[:3], 1):
                reward = rewards[i-1]
                bonus = diversity_bonuses[i-1]
                potential = innovation_potentials[i-1]
                print(f"{i}. {topic}")
                print(f"   Reward: {reward:.4f} + Diversity: {bonus:.4f} = {reward+bonus:.4f}")
                print(f"   Innovation Potential: {potential:.4f}")
    
    def _initiate_paradigm_shift(self):
        print("\n=== PARADIGM SHIFT INITIATED ===")
        for word in self.first_word_selector.word_stats:
            self.first_word_selector.word_stats[word]['attempts'] = 0
            self.first_word_selector.word_stats[word]['successes'] = 0
        
        self.kuhnian_monitor.reset_history()
        self.diversity_tracker = ConceptDiversityTracker()
        print("Diversity tracking reset for new paradigm")
    
    def save_innovation_history(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "topics": self.innovation_history,
                "feedback": self.human_feedback_log
            }, f, indent=2)
    
    def generate_report(self):
        valid_count = sum(1 for item in self.innovation_history 
                         if item['utility']['validity'] > 0.5)
        novelty_scores = [item['utility']['novelty'] for item in self.innovation_history]
        diversity_bonuses = [item['diversity_bonus'] for item in self.innovation_history]
        innovation_potentials = [item['innovation_potential'] for item in self.innovation_history]
        
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
        avg_diversity = np.mean(diversity_bonuses) if diversity_bonuses else 0
        avg_potential = np.mean(innovation_potentials) if innovation_potentials else 0
        transformational_ratio = sum(ip > 0.5 for ip in innovation_potentials) / max(1, len(innovation_potentials))
        
        print("\n=== INNOVATION REPORT ===")
        print(f"Total Topics: {len(self.innovation_history)}")
        print(f"Validity Rate: {valid_count/len(self.innovation_history)*100:.1f}%")
        print(f"Avg Novelty: {avg_novelty:.3f} | Transformational Ratio: {transformational_ratio:.3f}")
        print(f"Paradigm Shifts: {self.paradigm_shifts}")
        print(f"Human Feedback Collected: {len(self.human_feedback_log)}")
        
        # Show best RLHF-rated topics
        if self.human_feedback_log:
            human_ratings = []
            for item in self.human_feedback_log:
                fb = item["feedback"]
                rating = sum(w * fb[k] for w, k in zip(
                    CONFIG["human_weights"], 
                    ['novelty', 'usefulness', 'clarity', 'elegance']
                ))
                human_ratings.append((item["topic"], rating))
            
            top_human = sorted(human_ratings, key=lambda x: x[1], reverse=True)[:5]
            print("\nTop Human-Rated Topics:")
            for i, (topic, score) in enumerate(top_human, 1):
                print(f"{i}. [{score:.3f}] {topic}")

# ======================
# 11. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    pipeline = InnovationPipeline('preexisting_research_topics.csv', num_epochs=100)
    
    print("===== SUPERVISED TRAINING PHASE =====")
    pipeline.train_model()
    
    pipeline.rl_innovate(num_batches=20, topics_per_batch=50)
    
    pipeline.save_innovation_history('innovation_history.json')
    pipeline.generate_report()
    
    torch.save({
        'model_state': pipeline.model.state_dict(),
        'vocab': pipeline.tokenizer.vocab,
        'first_word_stats': pipeline.first_word_selector.word_stats,
        'kuhnian_state': pipeline.kuhnian_monitor.__dict__,
        'diversity_state': pipeline.diversity_tracker.__dict__
    }, 'innovation_engine.pt')
    print("\nModel saved to innovation_engine.pt")