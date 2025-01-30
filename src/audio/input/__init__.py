"""音声入力層パッケージ

このパッケージは音声認識システムの入力層を提供します。
主な機能：
- 音声ファイルのバリデーション
- ファイルアップロード
- キュー管理
- 音声認識層との連携
"""

from .exceptions import (
    InputLayerError,
    ValidationError,
    StorageError,
    QueueError
)
from .validator import FileValidator, AudioQualityResult
from .uploader import FileUploader
from .queue_manager import InputQueueManager, QueueItem
from .controller import InputController

__version__ = '1.0.0'
__all__ = [
    'InputController',
    'FileUploader',
    'FileValidator',
    'InputQueueManager',
    'AudioQualityResult',
    'QueueItem',
    'InputLayerError',
    'ValidationError',
    'StorageError',
    'QueueError'
]

# 型ヒント用のエイリアス
from typing import Dict, Any, BinaryIO, Optional, Callable

FileMetadata = Dict[str, Any]
ProcessingCallback = Callable[[str, FileMetadata], None]
ErrorCallback = Callable[[str, Exception], None]