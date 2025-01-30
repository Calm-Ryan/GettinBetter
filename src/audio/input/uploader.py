"""ファイルアップロードモジュール"""

import os
import uuid
from typing import Dict, Any, BinaryIO
from datetime import datetime

from .validator import FileValidator, AudioQualityResult
from .exceptions import ValidationError, StorageError
from src.storage.minio_client import MinioClient

class FileUploader:
    """ファイルアップローダー"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: アップローダー設定
                storage: ストレージ設定
                validation: バリデーション設定
        """
        self.storage_client = MinioClient(config['storage'])
        self.validator = FileValidator(config['validation'])

    def _save_temp_file(self, file: BinaryIO) -> str:
        """一時ファイルの保存

        Args:
            file: アップロードされたファイルオブジェクト

        Returns:
            str: 一時ファイルのパス
        """
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4()}.tmp")
        
        with open(temp_path, "wb") as f:
            f.write(file.read())
        
        return temp_path

    def _generate_file_id(self) -> str:
        """ファイルIDの生成

        Returns:
            str: 生成されたファイルID
        """
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def upload_file(self, file: BinaryIO, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """ファイルのアップロード処理

        Args:
            file: アップロードするファイル
            metadata: メタデータ

        Returns:
            Dict[str, Any]: アップロード結果
                file_id: ファイルID
                storage_path: ストレージパス
                quality_result: 音声品質検証結果
                metadata: 更新されたメタデータ

        Raises:
            ValidationError: バリデーションエラーの場合
            StorageError: ストレージエラーの場合
        """
        metadata = metadata or {}
        temp_path = None
        
        try:
            # 一時ファイルとして保存
            temp_path = self._save_temp_file(file)
            
            # バリデーション
            self.validator.validate_format(temp_path)
            self.validator.validate_size(temp_path)
            quality_result = self.validator.validate_audio_quality(temp_path)
            
            if not quality_result.is_valid:
                raise ValidationError(
                    f"音声品質の問題: {', '.join(quality_result.issues)}",
                    {"quality_result": quality_result.__dict__}
                )
            
            # ファイルID生成
            file_id = self._generate_file_id()
            
            # メタデータの更新
            updated_metadata = {
                **metadata,
                'upload_time': datetime.now().isoformat(),
                'file_id': file_id,
                'quality_info': quality_result.__dict__
            }
            
            # ストレージに保存
            storage_path = self.storage_client.save_file(
                file_path=temp_path,
                file_id=file_id,
                metadata=updated_metadata
            )
            
            return {
                'file_id': file_id,
                'storage_path': storage_path,
                'quality_result': quality_result,
                'metadata': updated_metadata
            }
            
        finally:
            # 一時ファイルの削除
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def get_file_status(self, file_id: str) -> Dict[str, Any]:
        """ファイル状態の取得

        Args:
            file_id: ファイルID

        Returns:
            Dict[str, Any]: ファイルの状態情報

        Raises:
            StorageError: ストレージエラーの場合
        """
        return self.storage_client.get_metadata(file_id)

    def delete_file(self, file_id: str) -> bool:
        """ファイルの削除

        Args:
            file_id: 削除するファイルのID

        Returns:
            bool: 削除成功の場合True

        Raises:
            StorageError: 削除エラーの場合
        """
        return self.storage_client.delete_file(file_id)