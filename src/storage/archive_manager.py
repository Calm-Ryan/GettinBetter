"""アーカイブマネージャーモジュール"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

from src.audio.input.exceptions import StorageError
from src.storage.minio_client import MinioClient
from src.storage.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

class ArchiveManager:
    """アーカイブマネージャー"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: アーカイブ設定
                archive_bucket: アーカイブ用バケット名
                age_threshold: アーカイブ対象となる経過日数
                batch_size: 一度にアーカイブする最大ファイル数
                storage: ストレージ設定
                database: データベース設定
        """
        self.config = config
        self.archive_bucket = config['archive_bucket']
        self.age_threshold = config['age_threshold']
        self.batch_size = config.get('batch_size', 100)

        # クライアントの初期化
        self.storage_client = MinioClient(config['storage'])
        self.db_client = PostgresClient(config['database'])

    async def initialize(self):
        """初期化処理"""
        await self.db_client.initialize()
        await self._ensure_archive_bucket_exists()

    async def close(self):
        """終了処理"""
        await self.db_client.close()

    async def _ensure_archive_bucket_exists(self):
        """アーカイブバケットの存在確認と作成"""
        try:
            if not self.storage_client.client.bucket_exists(self.archive_bucket):
                self.storage_client.client.make_bucket(self.archive_bucket)
        except Exception as e:
            raise StorageError(f"アーカイブバケット作成エラー: {str(e)}")

    async def execute_archiving(self) -> Dict[str, int]:
        """アーカイブ処理の実行

        Returns:
            Dict[str, int]: アーカイブ結果
                archived_count: アーカイブしたファイル数
                total_size: アーカイブした総サイズ

        Raises:
            StorageError: アーカイブ処理に失敗した場合
        """
        try:
            # アーカイブ対象の特定
            targets = await self._identify_archive_targets()
            
            archived_count = 0
            total_size = 0

            for target in targets:
                try:
                    # ファイルのアーカイブ
                    archive_result = await self._archive_file(target)
                    
                    # メタデータの更新
                    await self._update_file_status(
                        target['id'],
                        'archived',
                        archive_result['archive_path']
                    )
                    
                    archived_count += 1
                    total_size += target['file_size']

                except Exception as e:
                    logger.error(f"ファイルアーカイブエラー {target['id']}: {str(e)}")
                    continue

            return {
                'archived_count': archived_count,
                'total_size': total_size
            }

        except Exception as e:
            logger.error(f"アーカイブ処理エラー: {str(e)}")
            raise StorageError(f"アーカイブ処理エラー: {str(e)}")

    async def _identify_archive_targets(self) -> List[Dict[str, Any]]:
        """アーカイブ対象の特定"""
        query = """
        SELECT id, filename, file_size, storage_path, created_at
        FROM audio_files
        WHERE status = 'active'
        AND created_at < NOW() - INTERVAL '%s days'
        AND NOT EXISTS (
            SELECT 1 FROM access_stats
            WHERE access_stats.file_id = audio_files.id
            AND accessed_at > NOW() - INTERVAL '30 days'
        )
        ORDER BY created_at ASC
        LIMIT $1
        """

        try:
            return await self.db_client.pool.fetch(
                query % self.age_threshold,
                self.batch_size
            )
        except Exception as e:
            logger.error(f"アーカイブ対象特定エラー: {str(e)}")
            raise

    async def _archive_file(self, file_info: Dict[str, Any]) -> Dict[str, str]:
        """ファイルのアーカイブ処理

        Args:
            file_info: ファイル情報

        Returns:
            Dict[str, str]: アーカイブ結果
                archive_path: アーカイブパス

        Raises:
            StorageError: アーカイブに失敗した場合
        """
        try:
            # アーカイブパスの生成
            archive_date = file_info['created_at'].strftime('%Y/%m/%d')
            archive_path = f"{archive_date}/{file_info['id']}"

            # ファイルの移動
            await self._move_to_archive(
                file_info['storage_path'],
                archive_path
            )

            return {
                'archive_path': f"{self.archive_bucket}/{archive_path}"
            }

        except Exception as e:
            logger.error(f"ファイルアーカイブエラー: {str(e)}")
            raise

    async def _move_to_archive(self, source_path: str, archive_path: str):
        """ファイルのアーカイブバケットへの移動"""
        try:
            # ファイルのコピー
            self.storage_client.client.copy_object(
                self.archive_bucket,
                archive_path,
                f"{self.storage_client.bucket}/{source_path}"
            )

            # 元ファイルの削除
            self.storage_client.client.remove_object(
                self.storage_client.bucket,
                source_path
            )

        except Exception as e:
            raise StorageError(f"ファイル移動エラー: {str(e)}")

    async def _update_file_status(
        self,
        file_id: str,
        status: str,
        archive_path: str
    ):
        """ファイルステータスの更新"""
        query = """
        UPDATE audio_files
        SET status = $1,
            storage_path = $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $3
        """

        try:
            await self.db_client.pool.execute(
                query,
                status,
                archive_path,
                file_id
            )
        except Exception as e:
            logger.error(f"ステータス更新エラー: {str(e)}")
            raise

    async def restore_from_archive(self, file_id: str) -> Dict[str, Any]:
        """アーカイブからのファイル復元

        Args:
            file_id: ファイルID

        Returns:
            Dict[str, Any]: 復元したファイルの情報

        Raises:
            StorageError: 復元に失敗した場合
        """
        try:
            # ファイル情報の取得
            file_info = await self._get_archived_file_info(file_id)
            if not file_info:
                raise StorageError(f"アーカイブファイルが見つかりません: {file_id}")

            # 復元先パスの生成
            restore_path = f"restored/{file_id}"

            # ファイルの復元
            await self._restore_file(
                file_info['storage_path'],
                restore_path
            )

            # ステータスの更新
            await self._update_file_status(
                file_id,
                'active',
                restore_path
            )

            return {
                'id': file_id,
                'storage_path': restore_path,
                'status': 'active'
            }

        except Exception as e:
            logger.error(f"アーカイブ復元エラー: {str(e)}")
            raise StorageError(f"アーカイブ復元エラー: {str(e)}")

    async def _get_archived_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """アーカイブファイル情報の取得"""
        query = """
        SELECT id, storage_path, status
        FROM audio_files
        WHERE id = $1 AND status = 'archived'
        """

        try:
            return await self.db_client.pool.fetchrow(query, file_id)
        except Exception as e:
            logger.error(f"アーカイブ情報取得エラー: {str(e)}")
            raise

    async def _restore_file(self, archive_path: str, restore_path: str):
        """ファイルの復元処理"""
        try:
            # アーカイブからのコピー
            self.storage_client.client.copy_object(
                self.storage_client.bucket,
                restore_path,
                f"{self.archive_bucket}/{archive_path}"
            )

        except Exception as e:
            raise StorageError(f"ファイル復元エラー: {str(e)}")

    async def get_archive_stats(self) -> Dict[str, Any]:
        """アーカイブ統計情報の取得

        Returns:
            Dict[str, Any]: アーカイブ統計
                total_files: アーカイブされたファイル総数
                total_size: アーカイブの総サイズ
                monthly_stats: 月別統計
        """
        try:
            stats_query = """
            SELECT 
                COUNT(*) as total_files,
                SUM(file_size) as total_size,
                DATE_TRUNC('month', created_at) as month
            FROM audio_files
            WHERE status = 'archived'
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY month DESC
            """

            rows = await self.db_client.pool.fetch(stats_query)
            
            monthly_stats = {}
            total_files = 0
            total_size = 0

            for row in rows:
                month_key = row['month'].strftime('%Y-%m')
                monthly_stats[month_key] = {
                    'files': row['total_files'],
                    'size': row['total_size']
                }
                total_files += row['total_files']
                total_size += row['total_size']

            return {
                'total_files': total_files,
                'total_size': total_size,
                'monthly_stats': monthly_stats
            }

        except Exception as e:
            logger.error(f"アーカイブ統計取得エラー: {str(e)}")
            raise StorageError(f"アーカイブ統計取得エラー: {str(e)}")