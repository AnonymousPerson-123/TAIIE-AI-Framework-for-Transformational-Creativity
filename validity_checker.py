import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import random
import os

# ======================
# 1. LOAD PREEXISTING RESEARCH TOPICS
# ======================
def load_preexisting_topics(csv_path):
    """Load and label preexisting research topics"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}. Using synthetic data only.")
        return pd.DataFrame(columns=['topic', 'validity', 'novelty', 'coherence', 'domain_relevance'])
    
    df = pd.read_csv(csv_path)
    topics = df['Research Topic'].tolist()
    
    # Create labeled dataset for real topics
    data = []
    for topic in topics:
        # All preexisting topics are valid by definition
        data.append([
            topic,
            1,                          # validity = 1
            random.uniform(0.7, 0.95),   # novelty (high)
            random.uniform(0.8, 0.98),   # coherence (high)
            random.uniform(0.85, 0.99)   # domain relevance (high)
        ])
    
    return pd.DataFrame(data, columns=[
        'topic', 'validity', 'novelty', 'coherence', 'domain_relevance'
    ])

# ======================
# 2. CREATE ENHANCED LABELED DATASET
# ======================
def create_enhanced_dataset(num_synthetic=10000, real_topics_csv='preexisting_research_topics.csv'):
    """Create dataset combining synthetic and real research topics"""
    # Load preexisting topics
    real_df = load_preexisting_topics(real_topics_csv)
    
    # Create synthetic dataset
    synthetic_df = create_synthetic_dataset(num_synthetic)
    
    # Combine datasets
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    
    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    print(f"Created enhanced dataset with {len(real_df)} real topics and {len(synthetic_df)} synthetic topics")
    return combined_df

def create_synthetic_dataset(num_samples):
    """Create a synthetic dataset of valid and invalid research topics"""
    # Valid research topics patterns
    valid_patterns = [
        "{method} for {domain}",
        "{base} {method}",
        "Applications of {field} in {domain}",
        "{field} approaches to {domain} problems",
        "Novel {method} in {domain}"
    ]
    
    # Invalid research topics patterns
    invalid_patterns = [
        "{method} with {random_object}",
        "{random_adj} {random_object}",
        "{method} for {random_adj} {random_object}",
        "{random_verb} {random_object} with {method}",
        "{field} for {random_adj} {domain}"
    ]
    
    # Define components
    methods = ["Deep Learning", "Bayesian", "Reinforcement Learning", "Causal Inference", 
              "Federated Learning", "Generative Models", "Time Series Analysis", 
              "Dimensionality Reduction", "Ensemble Methods", "Graph Neural Networks"]
    
    bases = ["Quantum", "Neuro-symbolic", "Topological", "Spatio-temporal", "Adversarial", 
            "Explainable", "Differential", "Probabilistic", "Geometric", "Multi-modal"]
    
    fields = ["Statistics", "Machine Learning", "Data Science", "AI", "Computer Science"]
    
    domains = ["Healthcare", "Finance", "Cybersecurity", "Genomics", "Climate Science", 
              "Robotics", "Autonomous Vehicles", "Social Media", "Education", "Agriculture"]
    
    random_objects = ["Coffee", "Toothbrush", "Pizza", "Furniture", "Gardening", "Shoes", 
                     "Musical Instruments", "Pets", "Jewelry", "Sports"]
    
    random_adjs = ["Fuzzy", "Quantum", "Neural", "Stochastic", "Bayesian", "Deep", 
                  "Reinforcement", "Generative", "Adversarial", "Explainable"]
    
    random_verbs = ["Optimizing", "Classifying", "Predicting", "Generating", "Simulating"]
    
    # Create dataset
    data = []
    for i in range(num_samples):
        if i < num_samples * 0.7:  # 70% valid samples
            pattern = random.choice(valid_patterns)
            topic = pattern.format(
                method=random.choice(methods),
                domain=random.choice(domains),
                base=random.choice(bases),
                field=random.choice(fields)
            )
            validity = 1
            novelty = random.uniform(0.6, 0.95)  # Higher novelty for valid
            coherence = random.uniform(0.7, 0.98)
            domain_relevance = random.uniform(0.8, 0.99)
        else:  # 30% invalid samples
            pattern = random.choice(invalid_patterns)
            topic = pattern.format(
                method=random.choice(methods),
                domain=random.choice(domains),
                random_object=random.choice(random_objects),
                random_adj=random.choice(random_adjs),
                random_verb=random.choice(random_verbs),
                field=random.choice(fields)
            )
            validity = 0
            novelty = random.uniform(0.1, 0.6)  # Lower novelty for invalid
            coherence = random.uniform(0.1, 0.6)
            domain_relevance = random.uniform(0.1, 0.5)
        
        data.append([topic, validity, novelty, coherence, domain_relevance])
    
    return pd.DataFrame(data, columns=[
        'topic', 'validity', 'novelty', 'coherence', 'domain_relevance'
    ])

# ======================
# 3. DATASET CLASS
# ======================
class TopicDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=64):
        self.topics = dataframe['topic'].values
        self.labels = dataframe[['validity', 'novelty', 'coherence', 'domain_relevance']].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.topics)
    
    def __getitem__(self, idx):
        topic = str(self.topics[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            topic,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# ======================
# 4. UTILITY MODEL
# ======================
class TopicUtilityModel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)

# ======================
# 5. TRAINING FUNCTION
# ======================
def train_model(train_loader, val_loader):
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TopicUtilityModel().to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(10):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/10 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'topic_utility_model.pt')
            print("Saved best model")
    
    return model

# ======================
# 6. VALIDATION FUNCTION
# ======================
def validate_topic(topic, model, tokenizer, device, threshold=0.72):
    model.eval()
    encoding = tokenizer.encode_plus(
        topic,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        scores = torch.sigmoid(outputs).cpu().numpy()[0]
    
    return {
        'validity': float(scores[0]),
        'novelty': float(scores[1]),
        'coherence': float(scores[2]),
        'domain_relevance': float(scores[3]),
        'is_valid': scores[0] > threshold
    }

# ======================
# 7. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Create enhanced dataset
    print("Creating enhanced dataset...")
    enhanced_df = create_enhanced_dataset(num_synthetic=10000, 
                                         real_topics_csv='preexisting_research_topics.csv')
    
    # Split into train and validation
    train_df, val_df = train_test_split(enhanced_df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = TopicDataset(train_df, tokenizer)
    val_dataset = TopicDataset(val_df, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train or load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Try to load pre-trained model
        model = TopicUtilityModel().to(device)
        model.load_state_dict(torch.load('topic_utility_model.pt', map_location=device))
        print("Loaded pre-trained utility model")
    except:
        print("Training new utility model...")
        model = train_model(train_loader, val_loader)
    
    # Test validation function with sample topics
    test_topics = [
        "Deep Learning for Medical Diagnosis",
        "Quantum Pizza Delivery Optimization",
        "Federated Learning in Healthcare Systems",
        "Bayesian Toothbrush Design",
        "Causal Inference for Climate Policy"
    ]
    
    # Add some real topics from your dataset
    if not enhanced_df.empty:
        real_samples = enhanced_df.sample(3)['topic'].tolist()
        test_topics.extend(real_samples)
    
    print("\nValidation Results:")
    for topic in test_topics:
        result = validate_topic(topic, model, tokenizer, device)
        print(f"\nTopic: {topic}")
        print(f"Validity: {result['validity']:.4f} | Novelty: {result['novelty']:.4f}")
        print(f"Coherence: {result['coherence']:.4f} | Relevance: {result['domain_relevance']:.4f}")
        print(f"Valid: {result['is_valid']}")
    
    # Save the tokenizer for future use
    tokenizer.save_pretrained("topic_validator_tokenizer")
    print("\nSaved tokenizer for future use")