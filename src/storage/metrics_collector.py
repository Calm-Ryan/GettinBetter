"""メトリクスコレクターモジュール"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone, timedelta
import logging
from prometheus_client import Counter, Histogram, Gauge
import asyncio
from collections import defaultdict

from src.audio.input.exceptions import StorageError
from src.storage.minio_client import MinioClient
from src.storage.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

class MetricsCollector:
    """メトリクスコレクター"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: メトリクス設定
                collection_interval: 収集間隔（秒）
                storage: ストレージ設定
                database: データベース設定
        """
        self.config = config
        self.collection_interval = config.get('collection_interval', 300)  # デフォルト5分

        # クライアントの初期化
        self.storage_client = MinioClient(config['storage'])
        self.db_client = PostgresClient(config['database'])

        # Prometheusメトリクスの初期化
        self._initialize_metrics()

        # メトリクスキャッシュ
        self.metrics_cache = {}
        self.cache_timestamp = None
        self.cache_ttl = timedelta(minutes=5)

    def _initialize_metrics(self):
        """Prometheusメトリクスの初期化"""
        # ストレージメトリクス
        self.storage_usage = Gauge(
            'storage_bytes_total',
            'Total storage usage in bytes',
            ['storage_type']
        )

        self.file_count = Gauge(
            'storage_files_total',
            'Total number of files',
            ['status']
        )

        # 操作メトリクス
        self.operation_counter = Counter(
            'storage_operations_total',
            'Total number of storage operations',
            ['operation_type']
        )

        self.operation_latency = Histogram(
            'storage_operation_latency_seconds',
            'Latency of storage operations',
            ['operation_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        # エラーメトリクス
        self.error_counter = Counter(
            'storage_errors_total',
            'Total number of storage errors',
            ['error_type']
        )

        # バックアップメトリクス
        self.backup_size = Gauge(
            'backup_size_bytes',
            'Size of the latest backup in bytes'
        )

        self.backup_duration = Histogram(
            'backup_duration_seconds',
            'Duration of backup operations',
            buckets=[60, 300, 600, 1800, 3600]
        )

        # アーカイブメトリクス
        self.archived_files = Gauge(
            'archived_files_total',
            'Total number of archived files'
        )

        self.archive_size = Gauge(
            'archive_size_bytes',
            'Total size of archived files in bytes'
        )

    async def initialize(self):
        """初期化処理"""
        await self.db_client.initialize()
        await self.start_collection()

    async def close(self):
        """終了処理"""
        await self.db_client.close()

    async def start_collection(self):
        """メトリクス収集の開始"""
        while True:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"メトリクス収集エラー: {str(e)}")
                await asyncio.sleep(60)  # エラー時は1分待機

    async def collect_metrics(self):
        """メトリクスの収集"""
        try:
            # ストレージ使用状況の収集
            storage_stats = await self._collect_storage_stats()
            self.storage_usage.labels('object_storage').set(storage_stats['object_storage_size'])
            self.storage_usage.labels('database').set(storage_stats['database_size'])

            # ファイル統計の収集
            file_stats = await self._collect_file_stats()
            for status, count in file_stats.items():
                self.file_count.labels(status).set(count)

            # アーカイブ統計の収集
            archive_stats = await self._collect_archive_stats()
            self.archived_files.set(archive_stats['total_files'])
            self.archive_size.set(archive_stats['total_size'])

            # パフォーマンス統計の収集
            await self._collect_performance_stats()

        except Exception as e:
            logger.error(f"メトリクス収集エラー: {str(e)}")
            self.error_counter.labels('collection_error').inc()

    async def _collect_storage_stats(self) -> Dict[str, int]:
        """ストレージ使用状況の収集"""
        try:
            # オブジェクトストレージのサイズ
            object_size = await self._get_object_storage_size()

            # データベースのサイズ
            db_size = await self._get_database_size()

            return {
                'object_storage_size': object_size,
                'database_size': db_size
            }

        except Exception as e:
            logger.error(f"ストレージ統計収集エラー: {str(e)}")
            raise

    async def _get_object_storage_size(self) -> int:
        """オブジェクトストレージのサイズ取得"""
        query = """
        SELECT SUM(file_size) as total_size
        FROM audio_files
        WHERE status = 'active'
        """
        result = await self.db_client.pool.fetchval(query)
        return result or 0

    async def _get_database_size(self) -> int:
        """データベースサイズの取得"""
        query = """
        SELECT pg_database_size($1) as size
        """
        result = await self.db_client.pool.fetchval(
            query,
            self.config['database']['database']
        )
        return result or 0

    async def _collect_file_stats(self) -> Dict[str, int]:
        """ファイル統計の収集"""
        query = """
        SELECT status, COUNT(*) as count
        FROM audio_files
        GROUP BY status
        """
        rows = await self.db_client.pool.fetch(query)
        return {row['status']: row['count'] for row in rows}

    async def _collect_archive_stats(self) -> Dict[str, int]:
        """アーカイブ統計の収集"""
        query = """
        SELECT 
            COUNT(*) as total_files,
            SUM(file_size) as total_size
        FROM audio_files
        WHERE status = 'archived'
        """
        row = await self.db_client.pool.fetchrow(query)
        return {
            'total_files': row['total_files'] or 0,
            'total_size': row['total_size'] or 0
        }

    async def _collect_performance_stats(self):
        """パフォーマンス統計の収集"""
        query = """
        SELECT 
            operation_type,
            AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration,
            COUNT(*) as count
        FROM operation_logs
        WHERE start_time > NOW() - INTERVAL '5 minutes'
        GROUP BY operation_type
        """
        rows = await self.db_client.pool.fetch(query)
        
        for row in rows:
            self.operation_counter.labels(row['operation_type']).inc(row['count'])
            self.operation_latency.labels(row['operation_type']).observe(row['avg_duration'])

    async def get_storage_metrics(self) -> Dict[str, Any]:
        """ストレージメトリクスの取得

        Returns:
            Dict[str, Any]: ストレージメトリクス
                storage_usage: ストレージ使用状況
                file_stats: ファイル統計
                performance: パフォーマンス統計
                errors: エラー統計
        """
        # キャッシュの確認
        now = datetime.now(timezone.utc)
        if (self.cache_timestamp and 
            now - self.cache_timestamp < self.cache_ttl and
            self.metrics_cache):
            return self.metrics_cache

        try:
            metrics = {
                'storage_usage': {
                    'object_storage': await self._get_object_storage_size(),
                    'database': await self._get_database_size()
                },
                'file_stats': await self._collect_file_stats(),
                'archive_stats': await self._collect_archive_stats(),
                'performance': await self._get_performance_metrics(),
                'errors': await self._get_error_metrics()
            }

            # キャッシュの更新
            self.metrics_cache = metrics
            self.cache_timestamp = now

            return metrics

        except Exception as e:
            logger.error(f"メトリクス取得エラー: {str(e)}")
            raise StorageError(f"メトリクス取得エラー: {str(e)}")

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクスの取得"""
        query = """
        SELECT 
            operation_type,
            COUNT(*) as total_operations,
            AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration,
            MAX(EXTRACT(EPOCH FROM (end_time - start_time))) as max_duration
        FROM operation_logs
        WHERE start_time > NOW() - INTERVAL '1 hour'
        GROUP BY operation_type
        """
        rows = await self.db_client.pool.fetch(query)
        
        return {
            row['operation_type']: {
                'total_operations': row['total_operations'],
                'avg_duration': row['avg_duration'],
                'max_duration': row['max_duration']
            }
            for row in rows
        }

    async def _get_error_metrics(self) -> Dict[str, Any]:
        """エラーメトリクスの取得"""
        query = """
        SELECT 
            error_type,
            COUNT(*) as error_count
        FROM error_logs
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY error_type
        """
        rows = await self.db_client.pool.fetch(query)
        
        return {
            row['error_type']: row['error_count']
            for row in rows
        }

    def record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool = True
    ):
        """操作の記録

        Args:
            operation_type: 操作タイプ
            duration: 処理時間（秒）
            success: 成功したかどうか
        """
        self.operation_counter.labels(operation_type).inc()
        self.operation_latency.labels(operation_type).observe(duration)
        
        if not success:
            self.error_counter.labels(operation_type).inc()

    def record_error(self, error_type: str):
        """エラーの記録

        Args:
            error_type: エラータイプ
        """
        self.error_counter.labels(error_type).inc()