from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
from sentence_transformers import SentenceTransformer

@dataclass
class EmotionScore:
    category: str
    score: float
    confidence: float

@dataclass
class DetailedEmotion:
    primary: EmotionScore
    secondary: Optional[EmotionScore]
    intensity: float
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)

@dataclass
class MultiModalSentiment:
    text_sentiment: float
    audio_sentiment: Optional[float]
    combined_sentiment: float
    confidence: float

class EmotionClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            config.get('bert_model', 'rinna/japanese-roberta-base')
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            len(config['emotion_categories'])
        )
        self.emotion_categories = config['emotion_categories']

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class MultiModalFusion(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.text_dim = config.get('text_embedding_dim', 768)
        self.audio_dim = config.get('audio_feature_dim', 128)
        
        # 特徴量変換層
        self.text_transform = nn.Linear(self.text_dim, 256)
        self.audio_transform = nn.Linear(self.audio_dim, 256)
        
        # Attention層
        self.attention = nn.MultiheadAttention(256, 4)
        
        # 結合層
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        # 特徴量変換
        text_hidden = self.text_transform(text_features)
        audio_hidden = self.audio_transform(audio_features)
        
        # Self-Attention
        text_attended, _ = self.attention(
            text_hidden.unsqueeze(0),
            text_hidden.unsqueeze(0),
            text_hidden.unsqueeze(0)
        )
        
        # 特徴量の結合
        combined = torch.cat([
            text_attended.squeeze(0),
            audio_hidden
        ], dim=-1)
        
        # 最終的な感情スコア
        return torch.sigmoid(self.fusion(combined))

@dataclass
class SentimentResult:
    positive_score: float
    negative_score: float
    neutral_score: Optional[float]
    detailed_emotions: Dict[str, float]
    confidence: float

class BertSentimentAnalyzer:
    """マルチモーダル対応の感情分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # テキスト感情分析モデル
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.get('model_name', 'rinna/japanese-roberta-base')
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('model_name', 'rinna/japanese-roberta-base')
        )
        
        # 詳細感情分析モデル
        self.emotion_classifier = EmotionClassifier(config).to(self.device)
        
        # マルチモーダル結合モデル
        self.fusion_model = MultiModalFusion(config).to(self.device)
        
        # 設定
        self.max_length = config.get('max_length', 512)
        self.emotion_threshold = config.get('emotion_threshold', 0.5)
        self.emotion_categories = config.get('emotion_categories', [
            '喜び', '悲しみ', '怒り', '恐れ', '驚き', '嫌悪'
        ])
        
        # 感情強度の重み
        self.intensity_weights = config.get('intensity_weights', {
            'とても': 1.5,
            'かなり': 1.3,
            'やや': 0.7,
            'すこし': 0.5,
            'ほんの': 0.3
        })

    def analyze_sentiment(
        self,
        text: str,
        audio_features: Optional[np.ndarray] = None
    ) -> SentimentResult:
        """感情分析の実行"""
        # テキストの感情分析
        text_sentiment = self._analyze_text_sentiment(text)
        
        # 詳細感情の分析
        detailed_emotions = self._analyze_detailed_emotions(text)
        
        # 信頼度の計算
        confidence = self._calculate_confidence(text_sentiment)
        
        # マルチモーダル分析
        if audio_features is not None:
            combined_sentiment = self._combine_modalities(
                text_sentiment,
                audio_features
            )
            # マルチモーダルの結果を反映
            text_sentiment.update(combined_sentiment)
            confidence *= combined_sentiment['confidence']
        
        return SentimentResult(
            positive_score=text_sentiment['positive'],
            negative_score=text_sentiment['negative'],
            neutral_score=text_sentiment.get('neutral'),
            detailed_emotions=detailed_emotions,
            confidence=confidence
        )

    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """テキストベースの感情分析"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = F.softmax(outputs.logits, dim=1)[0]
            
        return {
            'positive': float(scores[1]),
            'negative': float(scores[0]),
            'neutral': float(scores[2]) if scores.shape[0] > 2 else None
        }

    def _analyze_detailed_emotions(self, text: str) -> Dict[str, float]:
        """詳細な感情カテゴリの分析"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            emotion_scores = F.softmax(
                self.emotion_classifier(
                    inputs['input_ids'],
                    inputs['attention_mask']
                ),
                dim=1
            )[0]
        
        return {
            category: float(score)
            for category, score in zip(self.emotion_categories, emotion_scores)
            if float(score) > self.emotion_threshold
        }

    def _combine_modalities(
        self,
        text_sentiment: Dict[str, float],
        audio_features: np.ndarray
    ) -> Dict[str, float]:
        """テキストと音声特徴量の結合"""
        # テキスト特徴量の準備
        text_features = torch.tensor([
            text_sentiment['positive'],
            text_sentiment['negative'],
            text_sentiment.get('neutral', 0.0)
        ]).float().to(self.device)
        
        # 音声特徴量の準備
        audio_tensor = torch.from_numpy(audio_features).float().to(self.device)
        
        # モダリティの結合
        with torch.no_grad():
            combined_score = self.fusion_model(text_features, audio_tensor)
            
        # 信頼度の計算
        confidence = float(torch.sigmoid(combined_score.std()))
        
        return {
            'combined_score': float(combined_score.mean()),
            'confidence': confidence
        }

    def _calculate_confidence(self, sentiment_scores: Dict[str, float]) -> float:
        """感情分析の信頼度計算"""
        # スコアの標準偏差を使用
        scores = np.array([
            score for score in sentiment_scores.values()
            if score is not None
        ])
        return float(np.std(scores))

    def analyze_emotion_trends(
        self,
        texts: List[str],
        window_size: int = 5
    ) -> List[Dict[str, float]]:
        """感情の時系列トレンド分析"""
        # 各テキストの感情スコアを計算
        sentiment_scores = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            sentiment_scores.append({
                'positive': sentiment.positive_score,
                'negative': sentiment.negative_score,
                'neutral': sentiment.neutral_score or 0.0
            })
        
        # 移動平均の計算
        trends = []
        for i in range(len(sentiment_scores)):
            start = max(0, i - window_size + 1)
            window = sentiment_scores[start:i + 1]
            
            # 各感情カテゴリの平均を計算
            avg_scores = {}
            for category in ['positive', 'negative', 'neutral']:
                scores = [s[category] for s in window]
                avg_scores[category] = sum(scores) / len(scores)
            
            trends.append(avg_scores)
        
        return trends

    def get_emotion_intensity(self, text: str) -> float:
        """感情の強度を計算"""
        # 感情強度表現の検出
        intensity = 1.0
        for word, weight in self.intensity_weights.items():
            if word in text:
                intensity *= weight
                break
        
        # 感情スコアの強度
        sentiment = self.analyze_sentiment(text)
        emotion_scores = [
            sentiment.positive_score,
            sentiment.negative_score
        ]
        if sentiment.neutral_score is not None:
            emotion_scores.append(sentiment.neutral_score)
        
        # 最大感情スコアと強度の組み合わせ
        return max(emotion_scores) * intensity