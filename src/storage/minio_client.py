"""MinIOストレージクライアントモジュール"""

from typing import Dict, Any, BinaryIO, Optional
from minio import Minio
from minio.error import S3Error
import os
import json
from datetime import datetime

from src.audio.input.exceptions import StorageError

class MinioClient:
    """MinIOストレージクライアント"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: MinIO設定
                endpoint: MinIOエンドポイント
                access_key: アクセスキー
                secret_key: シークレットキー
                bucket: バケット名
        """
        try:
            self.client = Minio(
                endpoint=config['endpoint'].replace('http://', '').replace('https://', ''),
                access_key=config['access_key'],
                secret_key=config['secret_key'],
                secure=config['endpoint'].startswith('https')
            )
            self.bucket = config['bucket']
            self._ensure_bucket_exists()
        except Exception as e:
            raise StorageError(f"MinIO接続エラー: {str(e)}")

    def _ensure_bucket_exists(self):
        """バケットの存在確認と作成"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            raise StorageError(f"バケット作成エラー: {str(e)}")

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """メタデータの準備

        Args:
            metadata: 元のメタデータ

        Returns:
            Dict[str, str]: S3互換のメタデータ
        """
        # メタデータはすべて文字列である必要がある
        prepared = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                prepared[key] = json.dumps(value)
            else:
                prepared[key] = str(value)
        
        # タイムスタンプの追加
        prepared['upload_time'] = datetime.utcnow().isoformat()
        return prepared

    def save_file(self, file_path: str, file_id: str, metadata: Dict[str, Any]) -> str:
        """ファイルの保存

        Args:
            file_path: 保存するファイルのパス
            file_id: ファイルID
            metadata: メタデータ

        Returns:
            str: 保存されたファイルのパス

        Raises:
            StorageError: 保存に失敗した場合
        """
        try:
            # メタデータの準備
            prepared_metadata = self._prepare_metadata(metadata)
            
            # ファイルのアップロード
            self.client.fput_object(
                bucket_name=self.bucket,
                object_name=file_id,
                file_path=file_path,
                metadata=prepared_metadata
            )
            
            return f"{self.bucket}/{file_id}"
        except Exception as e:
            raise StorageError(
                f"ファイル保存エラー: {str(e)}",
                {"file_id": file_id, "bucket": self.bucket}
            )

    def get_file(self, file_id: str, destination_path: str) -> str:
        """ファイルの取得

        Args:
            file_id: 取得するファイルのID
            destination_path: 保存先のパス

        Returns:
            str: 保存されたファイルのパス

        Raises:
            StorageError: 取得に失敗した場合
        """
        try:
            self.client.fget_object(
                bucket_name=self.bucket,
                object_name=file_id,
                file_path=destination_path
            )
            return destination_path
        except Exception as e:
            raise StorageError(
                f"ファイル取得エラー: {str(e)}",
                {"file_id": file_id, "bucket": self.bucket}
            )

    def get_metadata(self, file_id: str) -> Dict[str, Any]:
        """メタデータの取得

        Args:
            file_id: ファイルID

        Returns:
            Dict[str, Any]: メタデータ

        Raises:
            StorageError: メタデータの取得に失敗した場合
        """
        try:
            obj = self.client.stat_object(self.bucket, file_id)
            metadata = obj.metadata

            # JSON文字列の復元
            for key, value in metadata.items():
                try:
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    continue

            return metadata
        except Exception as e:
            raise StorageError(
                f"メタデータ取得エラー: {str(e)}",
                {"file_id": file_id, "bucket": self.bucket}
            )

    def delete_file(self, file_id: str) -> bool:
        """ファイルの削除

        Args:
            file_id: 削除するファイルのID

        Returns:
            bool: 削除成功の場合True

        Raises:
            StorageError: 削除に失敗した場合
        """
        try:
            self.client.remove_object(self.bucket, file_id)
            return True
        except Exception as e:
            raise StorageError(
                f"ファイル削除エラー: {str(e)}",
                {"file_id": file_id, "bucket": self.bucket}
            )