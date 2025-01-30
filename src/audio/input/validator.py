"""音声ファイルバリデーションモジュール"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import soundfile as sf
import librosa
import numpy as np

from .exceptions import ValidationError

@dataclass
class AudioQualityResult:
    """音声品質検証結果"""
    is_valid: bool
    sample_rate: int
    channels: int
    bit_depth: str
    noise_level: float
    duration: float
    issues: List[str]

class FileValidator:
    """音声ファイルバリデーター"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: バリデーション設定
                max_file_size: 最大ファイルサイズ（バイト）
                allowed_formats: 許可される音声フォーマット
                max_duration: 最大音声長（秒）
                min_sample_rate: 最小サンプルレート
                max_channels: 最大チャンネル数
        """
        self.max_file_size = config['max_file_size']
        self.allowed_formats = config['allowed_formats']
        self.max_duration = config['max_duration']
        self.min_sample_rate = config['min_sample_rate']
        self.max_channels = config['max_channels']

    def validate_format(self, file_path: str) -> bool:
        """ファイル形式の検証

        Args:
            file_path: 検証対象ファイルのパス

        Returns:
            bool: 検証結果

        Raises:
            ValidationError: 無効なファイル形式の場合
        """
        extension = file_path.split('.')[-1].lower()
        if extension not in self.allowed_formats:
            raise ValidationError(
                f"無効なファイル形式です: {extension}",
                {"allowed_formats": self.allowed_formats}
            )
        return True

    def validate_size(self, file_path: str) -> bool:
        """ファイルサイズの検証

        Args:
            file_path: 検証対象ファイルのパス

        Returns:
            bool: 検証結果

        Raises:
            ValidationError: ファイルサイズが上限を超える場合
        """
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            raise ValidationError(
                f"ファイルサイズが上限を超えています: {file_size} > {self.max_file_size}",
                {"file_size": file_size, "max_size": self.max_file_size}
            )
        return True

    def _estimate_noise_level(self, audio: np.ndarray) -> float:
        """ノイズレベルの推定

        Args:
            audio: 音声データ配列

        Returns:
            float: 推定ノイズレベル（0-1）
        """
        # 無音区間のRMSを計算
        frame_length = 2048
        hop_length = 512
        
        # フレームごとのRMS計算
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # 下位10%をノイズレベルとして使用
        noise_threshold = np.percentile(rms, 10)
        return float(noise_threshold)

    def validate_audio_quality(self, file_path: str) -> AudioQualityResult:
        """音声品質の検証

        Args:
            file_path: 検証対象ファイルのパス

        Returns:
            AudioQualityResult: 検証結果

        Raises:
            ValidationError: 音声品質の検証に失敗した場合
        """
        try:
            # 基本情報の取得
            audio_info = sf.info(file_path)
            issues = []

            # サンプルレートチェック
            if audio_info.samplerate < self.min_sample_rate:
                issues.append(f"サンプルレートが低すぎます: {audio_info.samplerate} < {self.min_sample_rate}")

            # チャンネル数チェック
            if audio_info.channels > self.max_channels:
                issues.append(f"チャンネル数が多すぎます: {audio_info.channels} > {self.max_channels}")

            # 音声長チェック
            if audio_info.duration > self.max_duration:
                issues.append(f"音声長が長すぎます: {audio_info.duration} > {self.max_duration}")

            # 音声データ読み込みとノイズレベル推定
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            noise_level = self._estimate_noise_level(audio)

            # ノイズレベルチェック
            if noise_level > 0.1:  # 10%以上をノイズ過多とする
                issues.append(f"ノイズレベルが高すぎます: {noise_level:.2f}")

            return AudioQualityResult(
                is_valid=len(issues) == 0,
                sample_rate=audio_info.samplerate,
                channels=audio_info.channels,
                bit_depth=str(audio_info.subtype),
                noise_level=noise_level,
                duration=audio_info.duration,
                issues=issues
            )

        except Exception as e:
            raise ValidationError(
                f"音声品質の検証に失敗しました: {str(e)}",
                {"file_path": file_path}
            )

    def validate(self, file_path: str) -> AudioQualityResult:
        """全ての検証を実行

        Args:
            file_path: 検証対象ファイルのパス

        Returns:
            AudioQualityResult: 検証結果

        Raises:
            ValidationError: いずれかの検証に失敗した場合
        """
        self.validate_format(file_path)
        self.validate_size(file_path)
        return self.validate_audio_quality(file_path)