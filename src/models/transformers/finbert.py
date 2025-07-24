"""
FinBERT Sentiment Analysis Model

Transformer-based model for financial sentiment analysis using FinBERT.
Analyzes news, social media, and financial reports for sentiment.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import structlog
from pathlib import Path
import json

from ..base import ClassificationModel, ClassificationConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class FinBERTConfig(ClassificationConfig):
    """Configuration for FinBERT sentiment model"""
    
    # Model selection
    model_name: str = "ProsusAI/finbert"  # Pre-trained FinBERT model
    max_length: int = 512  # Maximum sequence length
    
    # Training parameters
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Fine-tuning parameters
    freeze_base_model: bool = False
    freeze_embeddings: bool = True
    unfreeze_last_n_layers: int = 2
    
    # Sentiment classes
    sentiment_classes: List[str] = None
    use_neutral_class: bool = True
    
    # Data preprocessing
    clean_text: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    lowercase: bool = False  # Usually keep case for BERT
    
    # Inference
    batch_size_inference: int = 32
    
    # Multi-source sentiment
    aggregate_sources: bool = True
    source_weights: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.sentiment_classes is None:
            if self.use_neutral_class:
                self.sentiment_classes = ['negative', 'neutral', 'positive']
            else:
                self.sentiment_classes = ['negative', 'positive']
        
        if self.source_weights is None:
            self.source_weights = {
                'news': 0.4,
                'social_media': 0.2,
                'analyst_reports': 0.3,
                'sec_filings': 0.1
            }
        
        self.n_classes = len(self.sentiment_classes)
        self.class_names = self.sentiment_classes
        self.model_type = ModelType.SENTIMENT_ANALYSIS
        self.name = "FinBERT"


class FinancialTextDataset(Dataset):
    """Dataset for financial text data"""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: AutoTokenizer = None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class FinBERT(ClassificationModel):
    """FinBERT model for financial sentiment analysis"""
    
    def __init__(self, config: FinBERTConfig):
        super().__init__(config)
        self.config: FinBERTConfig = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        
    def build_model(self) -> AutoModelForSequenceClassification:
        """Build the FinBERT model"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.n_classes,
            ignore_mismatched_sizes=True  # Allow different number of classes
        )
        
        # Freeze layers if configured
        if self.config.freeze_base_model:
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier head
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            
            # Unfreeze last n layers if specified
            if self.config.unfreeze_last_n_layers > 0:
                # For BERT, unfreeze last n encoder layers
                if hasattr(self.model, 'bert'):
                    n_layers = len(self.model.bert.encoder.layer)
                    for i in range(n_layers - self.config.unfreeze_last_n_layers, n_layers):
                        for param in self.model.bert.encoder.layer[i].parameters():
                            param.requires_grad = True
        
        elif self.config.freeze_embeddings:
            # Only freeze embeddings
            if hasattr(self.model, 'bert'):
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = False
        
        # Move to device
        self.model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("FinBERT model built",
                   total_params=total_params,
                   trainable_params=trainable_params,
                   frozen_ratio=1 - (trainable_params / total_params))
        
        return self.model
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not self.config.clean_text:
            return text
        
        # Remove URLs
        if self.config.remove_urls:
            import re
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions
        if self.config.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lowercase if configured (usually not for BERT)
        if self.config.lowercase:
            text = text.lower()
        
        return text
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray, List[str]],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[Dataset, Optional[np.ndarray]]:
        """Prepare data for FinBERT"""
        
        # Extract text data
        if isinstance(data, pd.DataFrame):
            if 'text' in data.columns:
                texts = data['text'].tolist()
            elif 'content' in data.columns:
                texts = data['content'].tolist()
            else:
                raise ValueError("DataFrame must have 'text' or 'content' column")
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError("Data must be DataFrame with text column or list of strings")
        
        # Preprocess texts
        texts = [self.preprocess_text(text) for text in texts]
        
        # Process labels
        y = None
        if labels is not None:
            if isinstance(labels, pd.Series):
                y = labels.values
            else:
                y = labels
            
            # Encode labels if string
            if y.dtype == object:
                y = self.encode_labels(y, fit=is_training)
        
        # Create dataset
        dataset = FinancialTextDataset(
            texts=texts,
            labels=y.tolist() if y is not None else None,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        return dataset, y
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for evaluation"""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # Get predicted classes
        preds = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        train_data: Union[pd.DataFrame, List[str]],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the FinBERT model"""
        
        # Build model if not already built
        if self.model is None or self.tokenizer is None:
            self.build_model()
        
        # Prepare datasets
        train_dataset, _ = self.prepare_data(train_data, train_labels, is_training=True)
        
        eval_dataset = None
        if validation_data:
            eval_dataset, _ = self.prepare_data(
                validation_data[0], 
                validation_data[1], 
                is_training=False
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.save_path / "checkpoints"),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=str(self.config.save_path / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            greater_is_better=True,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            gradient_accumulation_steps=1,
            learning_rate=self.config.learning_rate,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info("Starting FinBERT training")
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        
        # Get training history
        history = {
            'train_loss': train_result.training_loss,
            'train_samples': len(train_dataset),
            'train_steps': train_result.global_step
        }
        
        # Add evaluation metrics if available
        if eval_dataset:
            eval_result = trainer.evaluate()
            history['eval_metrics'] = eval_result
        
        # Update metadata
        self.is_trained = True
        self.metadata['last_trained'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(train_dataset)
        self.training_history.append(history)
        
        logger.info("FinBERT training completed",
                   train_loss=history['train_loss'],
                   steps=history['train_steps'])
        
        return history
    
    def predict(
        self,
        data: Union[pd.DataFrame, List[str]],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with FinBERT"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        dataset, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_inference,
            shuffle=False
        )
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Decode labels if needed
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.decode_labels(predictions)
        
        return predictions
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, List[str]],
        **kwargs
    ) -> np.ndarray:
        """Get prediction probabilities"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        dataset, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_inference,
            shuffle=False
        )
        
        # Get probabilities
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get probabilities
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def analyze_text_sentiment(
        self,
        text: str,
        return_scores: bool = True
    ) -> Union[str, Dict[str, float]]:
        """Analyze sentiment of a single text"""
        
        # Get prediction probabilities
        probs = self.predict_proba([text])[0]
        
        if return_scores:
            # Return sentiment scores
            scores = {}
            for i, class_name in enumerate(self.config.sentiment_classes):
                scores[class_name] = float(probs[i])
            return scores
        else:
            # Return predicted class
            pred_idx = np.argmax(probs)
            return self.config.sentiment_classes[pred_idx]
    
    def analyze_multi_source_sentiment(
        self,
        sources_data: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Analyze sentiment from multiple sources with weighted aggregation"""
        
        results = {}
        all_scores = []
        
        # Analyze each source
        for source, texts in sources_data.items():
            if not texts:
                continue
            
            # Get sentiment scores for all texts
            source_scores = []
            for text in texts:
                scores = self.analyze_text_sentiment(text, return_scores=True)
                source_scores.append(scores)
            
            # Average scores for this source
            avg_scores = {}
            for sentiment in self.config.sentiment_classes:
                avg_scores[sentiment] = np.mean([s[sentiment] for s in source_scores])
            
            results[source] = {
                'scores': avg_scores,
                'dominant_sentiment': max(avg_scores, key=avg_scores.get),
                'n_texts': len(texts)
            }
            
            # Weight by source importance
            weight = self.config.source_weights.get(source, 0.1)
            weighted_scores = {k: v * weight for k, v in avg_scores.items()}
            all_scores.append(weighted_scores)
        
        # Aggregate weighted scores
        if all_scores and self.config.aggregate_sources:
            aggregated_scores = {}
            for sentiment in self.config.sentiment_classes:
                total_score = sum(scores[sentiment] for scores in all_scores)
                total_weight = sum(self.config.source_weights.get(source, 0.1) 
                                 for source in sources_data.keys() if sources_data[source])
                aggregated_scores[sentiment] = total_score / total_weight
            
            results['aggregated'] = {
                'scores': aggregated_scores,
                'dominant_sentiment': max(aggregated_scores, key=aggregated_scores.get)
            }
        
        return results
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save FinBERT model"""
        # Save base model data
        base_path = super().save(path)
        
        # Save transformer model and tokenizer
        if self.model is not None and self.tokenizer is not None:
            model_dir = base_path.parent / f"{base_path.stem}_transformer"
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            self.model.save_pretrained(model_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(model_dir)
            
            # Save additional metadata
            metadata_path = model_dir / "metadata.json"
            metadata = {
                'config': self.config.to_dict(),
                'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("FinBERT model saved", path=str(model_dir))
        
        return base_path
    
    @classmethod
    def load(cls, path: Path) -> 'FinBERT':
        """Load FinBERT model"""
        # Load base model data
        model_instance = super().load(path)
        
        # Load transformer model and tokenizer
        model_dir = path.parent / f"{path.stem}_transformer"
        if model_dir.exists():
            # Load tokenizer
            model_instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Load model
            model_instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model_instance.model.to(model_instance.device)
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Restore label encoder
                if metadata.get('label_encoder_classes'):
                    model_instance.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
            
            logger.info("FinBERT model loaded", path=str(model_dir))
        
        return model_instance