from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import torch
from faster_whisper import WhisperModel
import onnxruntime as ort
from pathlib import Path
from .processor import AudioSegment

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """音声認識結果を格納するデータクラス"""
    text: str
    start_time: float
    end_time: float
    confidence: float
    segment_id: str
    speaker_id: Optional[str] = None
    words: List[Dict[str, Any]] = None  # 単語レベルのタイムスタンプ

class TranscriptionError(Exception):
    """音声認識時のエラー"""
    pass

class ModelError(Exception):
    """モデル関連のエラー"""
    pass

class WhisperTranscriber:
    """Faster-Whisperモデルによる音声認識を実行するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定パラメータを含む辞書
                - model_name: モデル名（デフォルト: 'large-v3'）
                - device: 実行デバイス（デフォルト: 'cuda' if available else 'cpu'）
                - compute_type: 計算精度（デフォルト: 'float16'）
                - batch_size: バッチサイズ（デフォルト: 16）
                - language: 言語（デフォルト: 'ja'）
                - beam_size: ビームサイズ（デフォルト: 5）
                - word_timestamps: 単語レベルのタイムスタンプ（デフォルト: True）
        """
        self.model = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_type = config.get('compute_type', 'float16')
        self.batch_size = config.get('batch_size', 16)
        self.model_name = config.get('model_name', 'large-v3')
        self.language = config.get('language', 'ja')
        self.beam_size = config.get('beam_size', 5)
        self.word_timestamps = config.get('word_timestamps', True)
        
        # 日本語特化の設定
        self.initial_prompt = config.get('initial_prompt', 
            "以下は日本語の音声を文字に起こしたものです。")
        
        # ONNXランタイムの設定
        self.use_onnx = config.get('use_onnx', True)
        if self.use_onnx:
            self._setup_onnx_session()
        
        logger.info(f"Initializing WhisperTranscriber with model: {self.model_name} "
                   f"on device: {self.device}, compute_type: {self.compute_type}")

    def _setup_onnx_session(self):
        """ONNXランタイムセッションのセットアップ"""
        try:
            # GPUが利用可能な場合はGPUプロバイダーを使用
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.device == 'cuda' else ['CPUExecutionProvider']
            
            # セッションオプションの設定
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 4
            
            self.onnx_session = None  # モデルロード時に初期化
            logger.info("ONNX runtime session configured successfully")
        except Exception as e:
            logger.warning(f"Failed to setup ONNX session: {str(e)}")
            self.use_onnx = False

    def load_model(self):
        """Faster-Whisperモデルをロードする"""
        try:
            logger.info(f"Loading Faster-Whisper model: {self.model_name}")
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=Path.home() / ".cache" / "whisper"
            )
            logger.info("Model loaded successfully")
            
            # ONNXモデルの変換（有効な場合）
            if self.use_onnx:
                self._convert_to_onnx()
        except Exception as e:
            raise ModelError(f"Failed to load Whisper model: {str(e)}")

    def _convert_to_onnx(self):
        """WhisperモデルをONNX形式に変換"""
        try:
            if self.model is None:
                return
            
            onnx_path = Path.home() / ".cache" / "whisper" / f"whisper_{self.model_name}.onnx"
            if not onnx_path.exists():
                logger.info("Converting model to ONNX format...")
                # モデルの変換とONNXファイルの保存
                self.model.model.to_onnx(onnx_path)
            
            # ONNXセッションの作成
            self.onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.device == 'cuda' else ['CPUExecutionProvider']
            )
            logger.info("ONNX model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to convert/load ONNX model: {str(e)}")
            self.use_onnx = False

    def batch_transcribe(self, segments: List[AudioSegment]) -> List[TranscriptionResult]:
        """音声セグメントのバッチ処理による書き起こし

        Args:
            segments: 音声セグメントのリスト

        Returns:
            List[TranscriptionResult]: 認識結果のリスト

        Raises:
            TranscriptionError: 認識処理に失敗した場合
        """
        try:
            if self.model is None:
                self.load_model()

            results = []
            for i in range(0, len(segments), self.batch_size):
                batch = segments[i:i + self.batch_size]
                batch_audio = [seg.audio for seg in batch]
                
                # バッチ処理による推論
                batch_results = []
                for audio, segment in zip(batch_audio, batch):
                    # Faster-Whisperによる認識
                    segments_result, _ = self.model.transcribe(
                        audio,
                        language=self.language,
                        beam_size=self.beam_size,
                        word_timestamps=self.word_timestamps,
                        initial_prompt=self.initial_prompt,
                        condition_on_previous_text=True
                    )
                    
                    # 結果の整形
                    for seg in segments_result:
                        words_info = [
                            {
                                'text': word.word,
                                'start': word.start,
                                'end': word.end,
                                'probability': word.probability
                            }
                            for word in seg.words
                        ] if seg.words else []
                        
                        results.append(TranscriptionResult(
                            text=seg.text,
                            start_time=segment.start_time + seg.start,
                            end_time=segment.start_time + seg.end,
                            confidence=seg.avg_logprob,
                            segment_id=segment.segment_id,
                            words=words_info
                        ))
                
                logger.debug(f"Processed batch {i//self.batch_size + 1}/"
                           f"{(len(segments) + self.batch_size - 1)//self.batch_size}")
            
            return results
        except Exception as e:
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")

class TranscriptionProcessor:
    """認識結果の後処理を担当するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定パラメータを含む辞書
                - confidence_threshold: 信頼度閾値（デフォルト: 0.6）
                - min_segment_duration: 最小セグメント長（秒）（デフォルト: 1.0）
                - max_merge_interval: 最大マージ間隔（秒）（デフォルト: 0.5）
        """
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.min_segment_duration = config.get('min_segment_duration', 1.0)
        self.max_merge_interval = config.get('max_merge_interval', 0.5)

    def _should_merge(self, current: TranscriptionResult, next_result: TranscriptionResult) -> bool:
        """2つのセグメントをマージすべきか判定する"""
        time_gap = next_result.start_time - current.end_time
        return time_gap <= self.max_merge_interval

    def _merge_segments(self, current: TranscriptionResult, next_result: TranscriptionResult) -> TranscriptionResult:
        """2つのセグメントをマージする"""
        # 単語情報のマージ
        merged_words = []
        if current.words:
            merged_words.extend(current.words)
        if next_result.words:
            merged_words.extend(next_result.words)

        return TranscriptionResult(
            text=f"{current.text} {next_result.text}",
            start_time=current.start_time,
            end_time=next_result.end_time,
            confidence=min(current.confidence, next_result.confidence),
            segment_id=f"{current.segment_id}_{next_result.segment_id}",
            speaker_id=current.speaker_id,
            words=merged_words if merged_words else None
        )

    def merge_results(self, results: List[TranscriptionResult]) -> List[TranscriptionResult]:
        """連続する認識結果のマージ処理"""
        if not results:
            return []

        merged = []
        current = results[0]
        
        for result in results[1:]:
            if self._should_merge(current, result):
                current = self._merge_segments(current, result)
            else:
                if current.end_time - current.start_time >= self.min_segment_duration:
                    merged.append(current)
                current = result
        
        if current.end_time - current.start_time >= self.min_segment_duration:
            merged.append(current)
        
        return merged

    def filter_low_confidence(self, results: List[TranscriptionResult], 
                            threshold: Optional[float] = None) -> List[TranscriptionResult]:
        """低信頼度の結果をフィルタリング"""
        threshold = threshold or self.confidence_threshold
        return [r for r in results if r.confidence >= threshold]

    def format_output(self, results: List[TranscriptionResult]) -> Dict[str, Any]:
        """認識結果を出力形式に整形する"""
        return {
            'segments': [
                {
                    'text': r.text,
                    'start_time': r.start_time,
                    'end_time': r.end_time,
                    'confidence': r.confidence,
                    'segment_id': r.segment_id,
                    'speaker_id': r.speaker_id,
                    'words': r.words
                }
                for r in results
            ],
            'metadata': {
                'total_segments': len(results),
                'total_duration': sum(r.end_time - r.start_time for r in results),
                'average_confidence': sum(r.confidence for r in results) / len(results) if results else 0,
                'language': 'ja'
            }
        }