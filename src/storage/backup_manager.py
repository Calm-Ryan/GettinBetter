"""バックアップマネージャーモジュール"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone, timedelta
import uuid
import os
import shutil
import asyncio
import logging
from pathlib import Path

from src.audio.input.exceptions import StorageError
from src.storage.minio_client import MinioClient
from src.storage.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

class BackupManager:
    """バックアップマネージャー"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: バックアップ設定
                backup_path: バックアップ保存先パス
                retention: 保持期間設定
                    daily: 日次バックアップの保持日数
                    weekly: 週次バックアップの保持週数
                    monthly: 月次バックアップの保持月数
                storage: ストレージ設定
                database: データベース設定
        """
        self.config = config
        self.backup_path = Path(config['backup_path'])
        self.retention = config['retention']
        
        # ストレージクライアントの初期化
        self.storage_client = MinioClient(config['storage'])
        self.db_client = PostgresClient(config['database'])
        
        # バックアップディレクトリの作成
        self.backup_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """初期化処理"""
        await self.db_client.initialize()

    async def close(self):
        """終了処理"""
        await self.db_client.close()

    async def create_backup(self, backup_type: str = 'daily') -> str:
        """バックアップの作成

        Args:
            backup_type: バックアップタイプ（'daily', 'weekly', 'monthly'）

        Returns:
            str: バックアップID

        Raises:
            StorageError: バックアップ作成に失敗した場合
        """
        backup_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        backup_dir = self.backup_path / f"{backup_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{backup_id}"

        try:
            # バックアップディレクトリの作成
            backup_dir.mkdir(parents=True)

            # オブジェクトストレージのバックアップ
            objects_backup = await self._backup_objects(backup_dir)

            # データベースのバックアップ
            db_backup = await self._backup_database(backup_dir)

            # バックアップメタデータの作成
            metadata = {
                'id': backup_id,
                'type': backup_type,
                'timestamp': timestamp.isoformat(),
                'objects_count': objects_backup['count'],
                'objects_size': objects_backup['size'],
                'database_size': db_backup['size']
            }

            # メタデータの保存
            await self._save_backup_metadata(backup_id, metadata)

            # 古いバックアップの削除
            await self._cleanup_old_backups(backup_type)

            logger.info(f"バックアップ作成完了: {backup_id}")
            return backup_id

        except Exception as e:
            logger.error(f"バックアップ作成エラー: {str(e)}")
            # クリーンアップ
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise StorageError(f"バックアップ作成エラー: {str(e)}")

    async def restore_backup(self, backup_id: str) -> bool:
        """バックアップからのリストア

        Args:
            backup_id: バックアップID

        Returns:
            bool: リストア成功の場合True

        Raises:
            StorageError: リストアに失敗した場合
        """
        try:
            # バックアップメタデータの取得
            metadata = await self._get_backup_metadata(backup_id)
            backup_dir = self.backup_path / f"{metadata['type']}_{metadata['timestamp']}_{backup_id}"

            if not backup_dir.exists():
                raise StorageError(f"バックアップが見つかりません: {backup_id}")

            # リストアの実行
            await self._restore_objects(backup_dir)
            await self._restore_database(backup_dir)

            logger.info(f"リストア完了: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"リストアエラー: {str(e)}")
            raise StorageError(f"リストアエラー: {str(e)}")

    async def _backup_objects(self, backup_dir: Path) -> Dict[str, int]:
        """オブジェクトストレージのバックアップ

        Args:
            backup_dir: バックアップディレクトリ

        Returns:
            Dict[str, int]: バックアップ結果
                count: バックアップしたオブジェクト数
                size: 総サイズ
        """
        objects_dir = backup_dir / 'objects'
        objects_dir.mkdir()

        total_size = 0
        object_count = 0

        try:
            # オブジェクトの一覧取得とバックアップ
            objects = await self.storage_client.list_objects()
            for obj in objects:
                dest_path = objects_dir / obj.object_name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # オブジェクトのダウンロード
                await self.storage_client.get_file(
                    obj.object_name,
                    str(dest_path)
                )
                
                total_size += obj.size
                object_count += 1

            return {
                'count': object_count,
                'size': total_size
            }

        except Exception as e:
            logger.error(f"オブジェクトバックアップエラー: {str(e)}")
            raise

    async def _backup_database(self, backup_dir: Path) -> Dict[str, int]:
        """データベースのバックアップ

        Args:
            backup_dir: バックアップディレクトリ

        Returns:
            Dict[str, int]: バックアップ結果
                size: バックアップサイズ
        """
        db_backup_path = backup_dir / 'database.sql'

        try:
            # pg_dumpの実行
            process = await asyncio.create_subprocess_exec(
                'pg_dump',
                '-h', self.config['database']['host'],
                '-p', str(self.config['database']['port']),
                '-U', self.config['database']['user'],
                '-F', 'c',  # カスタムフォーマット
                '-f', str(db_backup_path),
                self.config['database']['database'],
                env={'PGPASSWORD': self.config['database']['password']}
            )
            
            await process.wait()
            
            if process.returncode != 0:
                raise StorageError("データベースバックアップ失敗")

            return {
                'size': db_backup_path.stat().st_size
            }

        except Exception as e:
            logger.error(f"データベースバックアップエラー: {str(e)}")
            raise

    async def _save_backup_metadata(self, backup_id: str, metadata: Dict[str, Any]):
        """バックアップメタデータの保存"""
        query = """
        INSERT INTO backups (
            id, backup_date, backup_size, file_count,
            status, storage_path, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        try:
            await self.db_client.pool.execute(
                query,
                backup_id,
                datetime.fromisoformat(metadata['timestamp']),
                metadata['objects_size'] + metadata['database_size'],
                metadata['objects_count'],
                'completed',
                str(self.backup_path),
                json.dumps(metadata)
            )
        except Exception as e:
            logger.error(f"メタデータ保存エラー: {str(e)}")
            raise

    async def _cleanup_old_backups(self, backup_type: str):
        """古いバックアップの削除"""
        retention_days = {
            'daily': self.retention['daily'],
            'weekly': self.retention['weekly'] * 7,
            'monthly': self.retention['monthly'] * 30
        }[backup_type]

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

        try:
            # 古いバックアップの検索
            query = """
            SELECT id, storage_path, metadata
            FROM backups
            WHERE backup_date < $1 AND status = 'completed'
            """
            
            old_backups = await self.db_client.pool.fetch(query, cutoff_date)

            # バックアップの削除
            for backup in old_backups:
                backup_path = Path(backup['storage_path'])
                if backup_path.exists():
                    shutil.rmtree(backup_path)

                # メタデータの更新
                await self.db_client.pool.execute(
                    "UPDATE backups SET status = 'deleted' WHERE id = $1",
                    backup['id']
                )

        except Exception as e:
            logger.error(f"バックアップクリーンアップエラー: {str(e)}")
            raise

    async def verify_backup_integrity(self, backup_id: str) -> bool:
        """バックアップの整合性検証

        Args:
            backup_id: バックアップID

        Returns:
            bool: 整合性チェック成功の場合True

        Raises:
            StorageError: 検証に失敗した場合
        """
        try:
            metadata = await self._get_backup_metadata(backup_id)
            backup_dir = self.backup_path / f"{metadata['type']}_{metadata['timestamp']}_{backup_id}"

            if not backup_dir.exists():
                raise StorageError(f"バックアップが見つかりません: {backup_id}")

            # オブジェクトの検証
            objects_dir = backup_dir / 'objects'
            if not objects_dir.exists():
                raise StorageError("オブジェクトバックアップが見つかりません")

            # データベースバックアップの検証
            db_backup_path = backup_dir / 'database.sql'
            if not db_backup_path.exists():
                raise StorageError("データベースバックアップが見つかりません")

            # サイズの検証
            actual_size = sum(f.stat().st_size for f in objects_dir.rglob('*') if f.is_file())
            actual_size += db_backup_path.stat().st_size

            expected_size = metadata['objects_size'] + metadata['database_size']
            if actual_size != expected_size:
                raise StorageError("バックアップサイズが一致しません")

            return True

        except Exception as e:
            logger.error(f"整合性検証エラー: {str(e)}")
            raise StorageError(f"整合性検証エラー: {str(e)}")

    async def _get_backup_metadata(self, backup_id: str) -> Dict[str, Any]:
        """バックアップメタデータの取得"""
        query = "SELECT * FROM backups WHERE id = $1"
        
        try:
            record = await self.db_client.pool.fetchrow(query, backup_id)
            if not record:
                raise StorageError(f"バックアップが見つかりません: {backup_id}")

            metadata = dict(record)
            if metadata.get('metadata'):
                metadata['metadata'] = json.loads(metadata['metadata'])
            return metadata

        except Exception as e:
            logger.error(f"メタデータ取得エラー: {str(e)}")
            raise