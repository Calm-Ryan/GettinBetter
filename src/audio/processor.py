from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import librosa
import logging
import webrtcvad
import soundfile as sf
from pathlib import Path
import wave
from rnnoise import RNNoise
import struct

logger = logging.getLogger(__name__)

@dataclass
class AudioData:
    """前処理済み音声データを保持するデータクラス"""
    raw_audio: np.ndarray
    sample_rate: int
    duration: float
    channels: int
    file_metadata: Dict[str, Any]
    vad_segments: List[Tuple[float, float]]  # VADで検出した音声区間

@dataclass
class AudioSegment:
    """分割された音声セグメントを表現するデータクラス"""
    audio: np.ndarray
    start_time: float
    end_time: float
    segment_id: str
    is_speech: bool  # VADによる音声判定結果

class AudioProcessError(Exception):
    """音声処理時のエラー"""
    pass

class AudioProcessor:
    """音声ファイルの前処理と分割を担当するクラス（WebRTCVADとRNNoise使用）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 設定パラメータを含む辞書
                - sample_rate: サンプリングレート（デフォルト: 16000）
                - max_duration: 最大セグメント長（秒）（デフォルト: 30）
                - vad_mode: VADの感度（0-3）（デフォルト: 3）
                - vad_frame_ms: VADフレーム長（ms）（デフォルト: 30）
                - min_speech_duration: 最小音声区間長（秒）（デフォルト: 0.3）
        """
        self.sample_rate = config.get('sample_rate', 16000)
        self.max_duration = config.get('max_duration', 30)
        self.vad_mode = config.get('vad_mode', 3)
        self.vad_frame_ms = config.get('vad_frame_ms', 30)
        self.min_speech_duration = config.get('min_speech_duration', 0.3)
        
        # VADとノイズ除去の初期化
        self.vad = webrtcvad.Vad(self.vad_mode)
        self.denoiser = RNNoise()

    def preprocess(self, file_path: str) -> AudioData:
        """音声ファイルの前処理を行う

        Args:
            file_path: 処理する音声ファイルのパス

        Returns:
            AudioData: 前処理済み音声データ

        Raises:
            AudioProcessError: 音声ファイルの読み込みや処理に失敗した場合
        """
        try:
            # 音声ファイルの存在確認
            if not Path(file_path).exists():
                raise AudioProcessError(f"File not found: {file_path}")

            # 音声ファイルの読み込み
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # ノイズ除去（RNNoise使用）
            audio = self._remove_noise_rnnoise(audio)
            
            # VADによる音声区間検出
            vad_segments = self._detect_speech_vad(audio)
            
            # AudioDataオブジェクトの作成
            return AudioData(
                raw_audio=audio,
                sample_rate=sr,
                duration=len(audio) / sr,
                channels=1,
                file_metadata=self._extract_metadata(file_path),
                vad_segments=vad_segments
            )
        except Exception as e:
            raise AudioProcessError(f"Failed to preprocess audio file: {str(e)}")

    def _remove_noise_rnnoise(self, audio: np.ndarray) -> np.ndarray:
        """RNNoiseを使用したノイズ除去

        Args:
            audio: 入力音声データ

        Returns:
            np.ndarray: ノイズ除去後の音声データ
        """
        # RNNoiseはfloat32の[-1, 1]範囲のデータを期待
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 音声を適切なフレームサイズに分割
        frame_size = 480  # RNNoiseの要求するフレームサイズ
        num_frames = len(audio) // frame_size
        
        # 余りのサンプルを処理するためのパディング
        remainder = len(audio) % frame_size
        if remainder > 0:
            padding = np.zeros(frame_size - remainder, dtype=np.float32)
            audio = np.concatenate([audio, padding])
            num_frames += 1
        
        # フレームごとにノイズ除去を適用
        denoised_audio = np.zeros_like(audio)
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end]
            denoised_frame = self.denoiser.process_frame(frame)
            denoised_audio[start:end] = denoised_frame
        
        # パディングを除去
        if remainder > 0:
            denoised_audio = denoised_audio[:-remainder]
        
        return denoised_audio

    def _detect_speech_vad(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """WebRTCVADを使用した音声区間検出

        Args:
            audio: 入力音声データ

        Returns:
            List[Tuple[float, float]]: 検出された音声区間（開始時間、終了時間）のリスト
        """
        # 音声データを適切なフォーマットに変換
        samples_per_frame = int(self.sample_rate * self.vad_frame_ms / 1000)
        audio_frames = []
        for i in range(0, len(audio), samples_per_frame):
            frame = audio[i:i + samples_per_frame]
            if len(frame) < samples_per_frame:
                # 最後のフレームが不完全な場合はパディング
                frame = np.pad(frame, (0, samples_per_frame - len(frame)))
            # int16に変換（WebRTCVADの要求）
            frame_int16 = (frame * 32768).astype(np.int16)
            audio_frames.append(frame_int16)

        # VADを適用
        speech_frames = []
        for i, frame in enumerate(audio_frames):
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            if is_speech:
                speech_frames.append(i)

        # 連続する音声フレームをマージ
        speech_segments = []
        start_frame = None
        for i, frame in enumerate(speech_frames):
            if start_frame is None:
                start_frame = frame
            elif frame - speech_frames[i-1] > 1:
                # ギャップを検出
                start_time = start_frame * self.vad_frame_ms / 1000
                end_time = speech_frames[i-1] * self.vad_frame_ms / 1000
                if end_time - start_time >= self.min_speech_duration:
                    speech_segments.append((start_time, end_time))
                start_frame = frame

        # 最後の音声区間を処理
        if start_frame is not None and speech_frames:
            start_time = start_frame * self.vad_frame_ms / 1000
            end_time = speech_frames[-1] * self.vad_frame_ms / 1000
            if end_time - start_time >= self.min_speech_duration:
                speech_segments.append((start_time, end_time))

        return speech_segments

    def split_audio(self, audio: AudioData) -> List[AudioSegment]:
        """音声を適切な長さのセグメントに分割

        Args:
            audio: 分割する音声データ

        Returns:
            List[AudioSegment]: 分割された音声セグメントのリスト
        """
        try:
            segments = []
            
            # VADで検出された音声区間に基づいて分割
            for start_time, end_time in audio.vad_segments:
                # 最大長を超える場合は分割
                current_time = start_time
                while current_time < end_time:
                    segment_end = min(current_time + self.max_duration, end_time)
                    
                    # 時間をサンプル数に変換
                    start_sample = int(current_time * audio.sample_rate)
                    end_sample = int(segment_end * audio.sample_rate)
                    
                    # セグメントの作成
                    segment = self._create_segment(
                        audio.raw_audio[start_sample:end_sample],
                        current_time,
                        segment_end,
                        is_speech=True
                    )
                    segments.append(segment)
                    
                    current_time = segment_end
            
            return segments
        except Exception as e:
            raise AudioProcessError(f"Failed to split audio: {str(e)}")

    def _create_segment(self, audio: np.ndarray, start_time: float, 
                       end_time: float, is_speech: bool) -> AudioSegment:
        """音声セグメントを作成する

        Args:
            audio: セグメントの音声データ
            start_time: 開始時間（秒）
            end_time: 終了時間（秒）
            is_speech: VADによる音声判定結果

        Returns:
            AudioSegment: 作成されたセグメント
        """
        return AudioSegment(
            audio=audio,
            start_time=start_time,
            end_time=end_time,
            segment_id=f"segment_{start_time:.3f}_{end_time:.3f}",
            is_speech=is_speech
        )

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """音声ファイルのメタデータを抽出する

        Args:
            file_path: 音声ファイルのパス

        Returns:
            Dict[str, Any]: メタデータを含む辞書
        """
        path = Path(file_path)
        metadata = {
            'filename': path.name,
            'file_size': path.stat().st_size,
            'format': path.suffix[1:],  # 拡張子から.を除去
        }
        
        # WAVファイルの場合は追加情報を取得
        if path.suffix.lower() == '.wav':
            try:
                with wave.open(str(path), 'rb') as wav:
                    metadata.update({
                        'channels': wav.getnchannels(),
                        'sample_width': wav.getsampwidth(),
                        'frame_rate': wav.getframerate(),
                        'n_frames': wav.getnframes(),
                        'compression_type': wav.getcomptype(),
                        'compression_name': wav.getcompname()
                    })
            except Exception as e:
                logger.warning(f"Failed to extract WAV metadata: {str(e)}")
        
        return metadata