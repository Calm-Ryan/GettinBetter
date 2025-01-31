from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import asyncio
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
from sudachipy import dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
from sentence_transformers import SentenceTransformer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import networkx as nx
import spacy
import ray
from dask.distributed import Client

from .sentiment import BertSentimentAnalyzer

@dataclass
class Morpheme:
    surface: str
    base_form: str
    pos: str
    reading: str

@dataclass
class MorphologicalResult:
    morphemes: List[Morpheme]
    sentence_boundaries: List[int]

@dataclass
class KeywordResult:
    keywords: List[Tuple[str, float]]
    score_method: str
    algorithm_scores: Dict[str, List[Tuple[str, float]]]  # 各アルゴリズムのスコア

@dataclass
class SentimentResult:
    positive_score: float
    negative_score: float
    neutral_score: Optional[float]
    detailed_emotions: Dict[str, float]  # 詳細な感情スコア

@dataclass
class TopicResult:
    topics: List[Tuple[str, float]]
    dominant_topic: str
    topic_distribution: Dict[str, float]
    coherence_score: float

@dataclass
class MultiModalFeatures:
    text_embeddings: np.ndarray
    audio_features: Optional[np.ndarray]
    combined_features: Optional[np.ndarray]

@dataclass
class CompleteAnalysisResult:
    text: str
    morphological_result: MorphologicalResult
    keyword_result: KeywordResult
    sentiment_result: SentimentResult
    topic_result: TopicResult
    multimodal_features: Optional[MultiModalFeatures]
    timestamp: datetime

class CacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.cache = {}
        self.max_size = config.get('max_size', 1000)
        self.ttl = config.get('ttl', 3600)  # 1時間
        
    def get(self, key: str) -> Optional[Any]:
        """キャッシュからの値取得"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any):
        """キャッシュへの値設定"""
        if len(self.cache) >= self.max_size:
            self._cleanup()
        self.cache[key] = (value, time.time())

    def _cleanup(self):
        """期限切れエントリの削除"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, ts) in self.cache.items()
            if current_time - ts >= self.ttl
        ]
        for k in expired_keys:
            del self.cache[k]

        # まだ容量が足りない場合は古いものから削除
        if len(self.cache) >= self.max_size:
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1][1]  # タイムスタンプでソート
            )
            for k, _ in sorted_items[:len(sorted_items) // 2]:
                del self.cache[k]

class IMorphologicalAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> MorphologicalResult:
        pass

    @abstractmethod
    def get_base_forms(self, text: str) -> List[str]:
        pass

class IKeywordExtractor(ABC):
    @abstractmethod
    def extract_keywords(self, text: str) -> KeywordResult:
        pass

    @abstractmethod
    def get_top_keywords(self, text: str, limit: int) -> List[Tuple[str, float]]:
        pass

class ISentimentAnalyzer(ABC):
    @abstractmethod
    def analyze_sentiment(self, text: str) -> SentimentResult:
        pass

    @abstractmethod
    def get_emotion_scores(self, text: str) -> Dict[str, float]:
        pass

class ITopicAnalyzer(ABC):
    @abstractmethod
    def extract_topics(self, text: str) -> TopicResult:
        pass

    @abstractmethod
    def get_topic_distribution(self, text: str) -> Dict[str, float]:
        pass

@ray.remote
class SudachiAnalyzer(IMorphologicalAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        self.tokenizer = dictionary.Dictionary().create()
        self.mode = config.get('mode', 'C')
        self.pos_filters = config.get('pos_filters', ['名詞', '動詞', '形容詞'])

    def analyze(self, text: str) -> MorphologicalResult:
        """テキストの形態素解析を実行"""
        morphemes = self.tokenizer.tokenize(text, self.mode)
        filtered_morphemes = self._filter_morphemes(morphemes)
        
        return MorphologicalResult(
            morphemes=[
                Morpheme(
                    surface=m.surface(),
                    base_form=m.dictionary_form(),
                    pos=m.part_of_speech()[0],
                    reading=m.reading_form()
                ) for m in filtered_morphemes
            ],
            sentence_boundaries=self._detect_boundaries(morphemes)
        )

    def get_base_forms(self, text: str) -> List[str]:
        """基本形の取得"""
        morphemes = self.tokenizer.tokenize(text, self.mode)
        filtered_morphemes = self._filter_morphemes(morphemes)
        return [m.dictionary_form() for m in filtered_morphemes]

    def _filter_morphemes(self, morphemes: List[Any]) -> List[Any]:
        """品詞フィルタリング"""
        return [
            m for m in morphemes
            if any(pos in m.part_of_speech()[0] for pos in self.pos_filters)
        ]

    def _detect_boundaries(self, morphemes: List[Any]) -> List[int]:
        """文境界の検出"""
        boundaries = []
        current_position = 0
        for m in morphemes:
            current_position += len(m.surface())
            if m.part_of_speech()[0] == '補助記号' and m.surface() in ['。', '！', '？']:
                boundaries.append(current_position)
        return boundaries

class HybridKeywordExtractor(IKeywordExtractor):
    def __init__(self, config: Dict[str, Any]):
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.get('max_features', 1000),
            stop_words=config.get('stop_words', 'japanese'),
            token_pattern=r'[^\s]+'
        )
        
        # BERT
        self.bert_model = SentenceTransformer(
            config.get('bert_model', 'rinna/japanese-roberta-base')
        )
        
        # TextRank
        self.nlp = spacy.load('ja_core_news_lg')
        
        self.max_keywords = config.get('max_keywords', 20)
        self.algorithm_weights = config.get('algorithm_weights', {
            'tfidf': 0.3,
            'bert': 0.4,
            'textrank': 0.3
        })

    def extract_keywords(self, text: str) -> KeywordResult:
        """ハイブリッドアプローチでキーワード抽出"""
        # 各アルゴリズムの実行
        tfidf_keywords = self._extract_tfidf_keywords(text)
        bert_keywords = self._extract_bert_keywords(text)
        textrank_keywords = self._extract_textrank_keywords(text)
        
        # スコアの統合
        combined_scores = defaultdict(float)
        for word, score in tfidf_keywords:
            combined_scores[word] += score * self.algorithm_weights['tfidf']
        for word, score in bert_keywords:
            combined_scores[word] += score * self.algorithm_weights['bert']
        for word, score in textrank_keywords:
            combined_scores[word] += score * self.algorithm_weights['textrank']
        
        # 最終キーワードの選択
        keywords = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_keywords]
        
        return KeywordResult(
            keywords=keywords,
            score_method='hybrid',
            algorithm_scores={
                'tfidf': tfidf_keywords,
                'bert': bert_keywords,
                'textrank': textrank_keywords
            }
        )

    def get_top_keywords(self, text: str, limit: int) -> List[Tuple[str, float]]:
        """上位キーワードの取得"""
        result = self.extract_keywords(text)
        return result.keywords[:limit]

    def _extract_tfidf_keywords(self, text: str) -> List[Tuple[str, float]]:
        """TF-IDFによるキーワード抽出"""
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        return [
            (feature_names[i], float(scores[i]))
            for i in scores.argsort()[::-1][:self.max_keywords]
        ]

    def _extract_bert_keywords(self, text: str) -> List[Tuple[str, float]]:
        """BERTによるキーワード抽出"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # 文のエンコード
        sentence_embeddings = self.bert_model.encode(sentences)
        
        # 単語の重要度計算
        word_scores = defaultdict(float)
        for sent, embedding in zip(doc.sents, sentence_embeddings):
            for token in sent:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    word_scores[token.text] += np.dot(
                        embedding,
                        self.bert_model.encode([token.text])[0]
                    )
        
        return sorted(
            word_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_keywords]

    def _extract_textrank_keywords(self, text: str) -> List[Tuple[str, float]]:
        """TextRankによるキーワード抽出"""
        doc = self.nlp(text)
        
        # グラフの構築
        graph = nx.Graph()
        for sent in doc.sents:
            words = [
                token.text for token in sent
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']
            ]
            
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    if graph.has_edge(words[i], words[j]):
                        graph[words[i]][words[j]]['weight'] += 1
                    else:
                        graph.add_edge(words[i], words[j], weight=1)
        
        # TextRankの計算
        scores = nx.pagerank(graph)
        
        return sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_keywords]

@ray.remote
class TopicAnalyzer(ITopicAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        self.num_topics = config.get('num_topics', 10)
        self.passes = config.get('passes', 20)
        self.dictionary = None
        self.lda_model = None
        self.nlp = spacy.load('ja_core_news_lg')

    def extract_topics(self, text: str) -> TopicResult:
        """トピック分析の実行"""
        # 形態素解析
        doc = self.nlp(text)
        tokens = [
            token.text for token in doc
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']
        ]
        
        # 辞書とコーパスの作成
        if not self.dictionary:
            self.dictionary = Dictionary([tokens])
        corpus = [self.dictionary.doc2bow(tokens)]
        
        # LDAモデルの学習
        if not self.lda_model:
            self.lda_model = LdaModel(
                corpus,
                num_topics=self.num_topics,
                id2word=self.dictionary,
                passes=self.passes
            )
        
        # トピック分布の取得
        topic_dist = self.lda_model.get_document_topics(corpus[0])
        
        # トピックの解釈
        topics = []
        for topic_id, prob in topic_dist:
            words = self.lda_model.show_topic(topic_id)
            topic_words = [(word, float(prob)) for word, prob in words]
            topics.append((f"Topic {topic_id}", float(prob)))
        
        # 最も確率の高いトピックを特定
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        
        return TopicResult(
            topics=topics,
            dominant_topic=dominant_topic,
            topic_distribution=dict(topics),
            coherence_score=self._calculate_coherence()
        )

    def get_topic_distribution(self, text: str) -> Dict[str, float]:
        """トピック分布の取得"""
        result = self.extract_topics(text)
        return result.topic_distribution

    def _calculate_coherence(self) -> float:
        """トピック一貫性スコアの計算"""
        if not self.lda_model:
            return 0.0
        return 0.0  # 簡略化のため

class DistributedTextAnalysisManager:
    def __init__(self, config: Dict[str, Any]):
        # Ray初期化
        ray.init(ignore_reinit_error=True)
        
        # Daskクライアント初期化
        self.dask_client = Client()
        
        # 分析コンポーネント
        self.morphological_analyzer = ray.remote(SudachiAnalyzer).remote(
            config['morphological']
        )
        self.keyword_extractor = HybridKeywordExtractor(config['keyword'])
        self.sentiment_analyzer = ray.remote(BertSentimentAnalyzer).remote(
            config['sentiment']
        )
        self.topic_analyzer = ray.remote(TopicAnalyzer).remote(config['topic'])
        
        # BERT model for text embeddings
        self.bert_model = SentenceTransformer(
            config.get('bert_model', 'rinna/japanese-roberta-base')
        )
        
        # キャッシュ
        self.cache_manager = CacheManager(config['cache'])

    async def analyze_text(
        self,
        text: str,
        audio_features: Optional[np.ndarray] = None
    ) -> CompleteAnalysisResult:
        """非同期での並列解析実行"""
        cache_key = self._generate_cache_key(text)
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        # 並列タスクの実行
        tasks = [
            self._run_morphological_analysis(text),
            self._run_keyword_extraction(text),
            self._run_sentiment_analysis(text),
            self._run_topic_analysis(text)
        ]
        
        if audio_features is not None:
            tasks.append(self._extract_multimodal_features(text, audio_features))
        
        results = await asyncio.gather(*tasks)
        complete_result = self._merge_results(
            text,
            results,
            audio_features is not None
        )
        
        self.cache_manager.set(cache_key, complete_result)
        return complete_result

    async def _run_morphological_analysis(self, text: str) -> MorphologicalResult:
        """形態素解析の非同期実行"""
        future = self.morphological_analyzer.analyze.remote(text)
        return await asyncio.to_thread(ray.get, future)

    async def _run_keyword_extraction(self, text: str) -> KeywordResult:
        """キーワード抽出の非同期実行"""
        return await asyncio.to_thread(
            self.keyword_extractor.extract_keywords,
            text
        )

    async def _run_sentiment_analysis(self, text: str) -> SentimentResult:
        """感情分析の非同期実行"""
        future = self.sentiment_analyzer.analyze_sentiment.remote(text)
        return await asyncio.to_thread(ray.get, future)

    async def _run_topic_analysis(self, text: str) -> TopicResult:
        """トピック分析の非同期実行"""
        future = self.topic_analyzer.extract_topics.remote(text)
        return await asyncio.to_thread(ray.get, future)

    async def _extract_multimodal_features(
        self,
        text: str,
        audio_features: np.ndarray
    ) -> MultiModalFeatures:
        """マルチモーダル特徴量の抽出"""
        text_embeddings = await asyncio.to_thread(
            self.bert_model.encode,
            [text]
        )
        
        combined_features = np.concatenate([
            text_embeddings[0],
            audio_features
        ])
        
        return MultiModalFeatures(
            text_embeddings=text_embeddings[0],
            audio_features=audio_features,
            combined_features=combined_features
        )

    def _merge_results(
        self,
        text: str,
        results: List[Any],
        has_audio: bool
    ) -> CompleteAnalysisResult:
        """解析結果のマージ"""
        return CompleteAnalysisResult(
            text=text,
            morphological_result=results[0],
            keyword_result=results[1],
            sentiment_result=results[2],
            topic_result=results[3],
            multimodal_features=results[4] if has_audio else None,
            timestamp=datetime.now()
        )

    def _generate_cache_key(self, text: str) -> str:
        """キャッシュキーの生成"""
        return f"analysis_{hash(text)}"

    async def close(self):
        """リソースの解放"""
        ray.shutdown()
        await self.dask_client.close()