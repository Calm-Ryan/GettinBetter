"""ストレージファサードモジュール"""

from typing import Dict, Any, List, Optional, BinaryIO
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import uuid

from src.audio.input.exceptions import StorageError
from src.storage.minio_client import MinioClient
from src.storage.postgres_client import PostgresClient
from src.storage.backup_manager import BackupManager
from src.storage.archive_manager import ArchiveManager
from src.storage.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

class StorageFacade:
    """ストレージファサード"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: ストレージ設定
                storage: オブジェクトストレージ設定
                database: データベース設定
                backup: バックアップ設定
                archive: アーカイブ設定
                metrics: メトリクス設定
        """
        self.config = config
        
        # コンポーネントの初期化
        self.storage_client = MinioClient(config['storage'])
        self.db_client = PostgresClient(config['database'])
        self.backup_manager = BackupManager(config['backup'])
        self.archive_manager = ArchiveManager(config['archive'])
        self.metrics_collector = MetricsCollector(config['metrics'])

        # バックグラウンドタスク
        self.background_tasks = []

    async def initialize(self):
        """初期化処理"""
        try:
            # 各コンポーネントの初期化
            await self.db_client.initialize()
            await self.backup_manager.initialize()
            await self.archive_manager.initialize()
            await self.metrics_collector.initialize()

            # バックグラウンドタスクの開始
            self.background_tasks.extend([
                asyncio.create_task(self._run_periodic_backup()),
                asyncio.create_task(self._run_periodic_archiving()),
                asyncio.create_task(self._run_metrics_collection())
            ])

            logger.info("ストレージ層の初期化が完了しました")

        except Exception as e:
            logger.error(f"ストレージ層の初期化に失敗しました: {str(e)}")
            raise StorageError(f"初期化エラー: {str(e)}")

    async def close(self):
        """終了処理"""
        # バックグラウンドタスクの停止
        for task in self.background_tasks:
            task.cancel()
        
        # 各コンポーネントの終了処理
        await self.db_client.close()
        await self.backup_manager.close()
        await self.archive_manager.close()
        await self.metrics_collector.close()

    async def save_audio_file(
        self,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """音声ファイルの保存

        Args:
            file_path: 保存するファイルのパス
            metadata: メタデータ
                filename: ファイル名
                file_size: ファイルサイズ
                mime_type: MIMEタイプ
                duration: 再生時間
                sample_rate: サンプルレート
                channels: チャンネル数

        Returns:
            Dict[str, Any]: 保存結果
                file_id: ファイルID
                storage_path: ストレージパス

        Raises:
            StorageError: 保存に失敗した場合
        """
        try:
            start_time = datetime.now()

            # ファイルの保存
            file_id = metadata.get('file_id', str(uuid.uuid4()))
            storage_path = await self.storage_client.save_file(
                file_path,
                file_id,
                metadata
            )

            # メタデータの保存
            metadata['storage_path'] = storage_path
            await self.db_client.save_audio_metadata(file_id, metadata)

            # メトリクスの記録
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_operation(
                'save_audio_file',
                duration,
                success=True
            )

            return {
                'file_id': file_id,
                'storage_path': storage_path
            }

        except Exception as e:
            self.metrics_collector.record_error('save_audio_file_error')
            logger.error(f"ファイル保存エラー: {str(e)}")
            raise StorageError(f"ファイル保存エラー: {str(e)}")

    async def get_audio_file(
        self,
        file_id: str,
        destination_path: str
    ) -> Dict[str, Any]:
        """音声ファイルの取得

        Args:
            file_id: ファイルID
            destination_path: 保存先のパス

        Returns:
            Dict[str, Any]: ファイル情報

        Raises:
            StorageError: 取得に失敗した場合
        """
        try:
            start_time = datetime.now()

            # メタデータの取得
            metadata = await self.db_client.get_audio_metadata(file_id)

            # ファイルの取得
            file_path = await self.storage_client.get_file(
                file_id,
                destination_path
            )

            # アクセス記録
            await self.db_client.record_access(
                file_id,
                'download',
                {'destination': destination_path}
            )

            # メトリクスの記録
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_operation(
                'get_audio_file',
                duration,
                success=True
            )

            return {
                'file_path': file_path,
                'metadata': metadata
            }

        except Exception as e:
            self.metrics_collector.record_error('get_audio_file_error')
            logger.error(f"ファイル取得エラー: {str(e)}")
            raise StorageError(f"ファイル取得エラー: {str(e)}")

    async def save_analysis_result(
        self,
        audio_file_id: str,
        analysis_type: str,
        result_data: Dict[str, Any],
        model_version: Optional[str] = None
    ) -> str:
        """解析結果の保存

        Args:
            audio_file_id: 音声ファイルID
            analysis_type: 解析タイプ
            result_data: 解析結果データ
            model_version: モデルバージョン

        Returns:
            str: 解析結果ID

        Raises:
            StorageError: 保存に失敗した場合
        """
        try:
            start_time = datetime.now()

            # 解析結果の保存
            result_id = await self.db_client.save_analysis_result(
                audio_file_id,
                analysis_type,
                result_data,
                model_version,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # メトリクスの記録
            self.metrics_collector.record_operation(
                'save_analysis_result',
                (datetime.now() - start_time).total_seconds(),
                success=True
            )

            return result_id

        except Exception as e:
            self.metrics_collector.record_error('save_analysis_result_error')
            logger.error(f"解析結果保存エラー: {str(e)}")
            raise StorageError(f"解析結果保存エラー: {str(e)}")

    async def create_backup(self, backup_type: str = 'daily') -> str:
        """バックアップの作成

        Args:
            backup_type: バックアップタイプ

        Returns:
            str: バックアップID

        Raises:
            StorageError: バックアップ作成に失敗した場合
        """
        try:
            start_time = datetime.now()

            # バックアップの作成
            backup_id = await self.backup_manager.create_backup(backup_type)

            # メトリクスの記録
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_operation(
                'create_backup',
                duration,
                success=True
            )

            return backup_id

        except Exception as e:
            self.metrics_collector.record_error('create_backup_error')
            logger.error(f"バックアップ作成エラー: {str(e)}")
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
            start_time = datetime.now()

            # リストアの実行
            success = await self.backup_manager.restore_backup(backup_id)

            # メトリクスの記録
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_operation(
                'restore_backup',
                duration,
                success=success
            )

            return success

        except Exception as e:
            self.metrics_collector.record_error('restore_backup_error')
            logger.error(f"リストアエラー: {str(e)}")
            raise StorageError(f"リストアエラー: {str(e)}")

    async def get_storage_metrics(self) -> Dict[str, Any]:
        """ストレージメトリクスの取得

        Returns:
            Dict[str, Any]: ストレージメトリクス

        Raises:
            StorageError: メトリクス取得に失敗した場合
        """
        try:
            return await self.metrics_collector.get_storage_metrics()
        except Exception as e:
            logger.error(f"メトリクス取得エラー: {str(e)}")
            raise StorageError(f"メトリクス取得エラー: {str(e)}")

    async def _run_periodic_backup(self):
        """定期バックアップの実行"""
        while True:
            try:
                # 日次バックアップの実行
                await self.create_backup('daily')
                # 24時間待機
                await asyncio.sleep(24 * 60 * 60)
            except Exception as e:
                logger.error(f"定期バックアップエラー: {str(e)}")
                await asyncio.sleep(60 * 60)  # エラー時は1時間待機

    async def _run_periodic_archiving(self):
        """定期アーカイブの実行"""
        while True:
            try:
                # アーカイブの実行
                await self.archive_manager.execute_archiving()
                # 1時間待機
                await asyncio.sleep(60 * 60)
            except Exception as e:
                logger.error(f"定期アーカイブエラー: {str(e)}")
                await asyncio.sleep(15 * 60)  # エラー時は15分待機

    async def _run_metrics_collection(self):
        """メトリクス収集の実行"""
        while True:
            try:
                # メトリクスの収集
                await self.metrics_collector.collect_metrics()
                # 5分待機
                await asyncio.sleep(5 * 60)
            except Exception as e:
                logger.error(f"メトリクス収集エラー: {str(e)}")
                await asyncio.sleep(60)  # エラー時は1分待機