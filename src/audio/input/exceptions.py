"""入力層の例外定義モジュール"""

class InputLayerError(Exception):
    """入力層の基底エラー"""
    pass

class ValidationError(InputLayerError):
    """バリデーションエラー"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class StorageError(InputLayerError):
    """ストレージ関連エラー"""
    def __init__(self, message: str, storage_info: dict = None):
        super().__init__(message)
        self.storage_info = storage_info or {}

class QueueError(InputLayerError):
    """キュー関連エラー"""
    def __init__(self, message: str, queue_info: dict = None):
        super().__init__(message)
        self.queue_info = queue_info or {}