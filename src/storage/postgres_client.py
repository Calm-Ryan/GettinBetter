"""PostgreSQLデータベースクライアントモジュール"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timezone
import asyncpg
from asyncpg import Pool
import uuid

from src.audio.input.exceptions import StorageError

class PostgresClient:
    """PostgreSQLデータベースクライアント"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: PostgreSQL設定
                host: データベースホスト
                port: ポート番号
                user: ユーザー名
                password: パスワード
                database: データベース名
                min_connections: 最小コネクション数
                max_connections: 最大コネクション数
        """
        self.config = config
        self.pool: Optional[Pool] = None

    async def initialize(self):
        """データベース接続の初期化"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                min_size=self.config.get('min_connections', 5),
                max_size=self.config.get('max_connections', 20),
                command_timeout=60.0
            )
        except Exception as e:
            raise StorageError(f"データベース接続エラー: {str(e)}")

    async def close(self):
        """データベース接続のクローズ"""
        if self.pool:
            await self.pool.close()

    async def save_audio_metadata(self, file_id: str, metadata: Dict[str, Any]) -> bool:
        """音声ファイルメタデータの保存

        Args:
            file_id: ファイルID
            metadata: メタデータ
                filename: ファイル名
                file_size: ファイルサイズ
                mime_type: MIMEタイプ
                duration: 再生時間
                sample_rate: サンプルレート
                channels: チャンネル数
                storage_path: ストレージパス
                additional_metadata: その他のメタデータ

        Returns:
            bool: 保存成功の場合True

        Raises:
            StorageError: 保存に失敗した場合
        """
        if not self.pool:
            raise StorageError("データベース未初期化")

        query = """
        INSERT INTO audio_files (
            id, filename, file_size, mime_type, duration,
            sample_rate, channels, storage_path, status, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        query,
                        file_id,
                        metadata['filename'],
                        metadata['file_size'],
                        metadata['mime_type'],
                        metadata.get('duration'),
                        metadata.get('sample_rate'),
                        metadata.get('channels'),
                        metadata['storage_path'],
                        'active',
                        json.dumps(metadata.get('additional_metadata', {}))
                    )
            return True
        except Exception as e:
            raise StorageError(f"メタデータ保存エラー: {str(e)}")

    async def save_analysis_result(
        self, 
        audio_file_id: str, 
        analysis_type: str, 
        result_data: Dict[str, Any],
        model_version: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> str:
        """解析結果の保存

        Args:
            audio_file_id: 音声ファイルID
            analysis_type: 解析タイプ
            result_data: 解析結果データ
            model_version: モデルバージョン
            processing_time: 処理時間

        Returns:
            str: 解析結果ID

        Raises:
            StorageError: 保存に失敗した場合
        """
        if not self.pool:
            raise StorageError("データベース未初期化")

        query = """
        INSERT INTO analysis_results (
            id, audio_file_id, analysis_type, result_data,
            model_version, processing_time
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """

        try:
            result_id = str(uuid.uuid4())
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        query,
                        result_id,
                        audio_file_id,
                        analysis_type,
                        json.dumps(result_data),
                        model_version,
                        processing_time
                    )
            return result_id
        except Exception as e:
            raise StorageError(f"解析結果保存エラー: {str(e)}")

    async def get_audio_metadata(self, file_id: str) -> Dict[str, Any]:
        """音声ファイルメタデータの取得

        Args:
            file_id: ファイルID

        Returns:
            Dict[str, Any]: メタデータ

        Raises:
            StorageError: 取得に失敗した場合
        """
        if not self.pool:
            raise StorageError("データベース未初期化")

        query = """
        SELECT * FROM audio_files WHERE id = $1
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, file_id)
                if not row:
                    raise StorageError(f"ファイルが見つかりません: {file_id}")
                
                # Record to dict conversion
                metadata = dict(row)
                # JSONBフィールドのデコード
                if metadata.get('metadata'):
                    metadata['additional_metadata'] = json.loads(metadata.pop('metadata'))
                return metadata
        except Exception as e:
            raise StorageError(f"メタデータ取得エラー: {str(e)}")

    async def get_analysis_results(
        self, 
        audio_file_id: str,
        analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """解析結果の取得

        Args:
            audio_file_id: 音声ファイルID
            analysis_type: 解析タイプ（指定しない場合は全ての解析結果を取得）

        Returns:
            List[Dict[str, Any]]: 解析結果のリスト

        Raises:
            StorageError: 取得に失敗した場合
        """
        if not self.pool:
            raise StorageError("データベース未初期化")

        query = """
        SELECT * FROM analysis_results 
        WHERE audio_file_id = $1
        """
        params = [audio_file_id]

        if analysis_type:
            query += " AND analysis_type = $2"
            params.append(analysis_type)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                results = []
                for row in rows:
                    result = dict(row)
                    # JSONBフィールドのデコード
                    if result.get('result_data'):
                        result['result_data'] = json.loads(result['result_data'])
                    results.append(result)
                return results
        except Exception as e:
            raise StorageError(f"解析結果取得エラー: {str(e)}")

    async def record_access(
        self,
        file_id: str,
        access_type: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """アクセス記録の保存

        Args:
            file_id: ファイルID
            access_type: アクセスタイプ（例: 'download', 'view'）
            client_info: クライアント情報

        Returns:
            bool: 保存成功の場合True

        Raises:
            StorageError: 保存に失敗した場合
        """
        if not self.pool:
            raise StorageError("データベース未初期化")

        query = """
        INSERT INTO access_stats (
            file_id, access_type, client_info
        ) VALUES ($1, $2, $3)
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    file_id,
                    access_type,
                    json.dumps(client_info) if client_info else None
                )
            return True
        except Exception as e:
            raise StorageError(f"アクセス記録保存エラー: {str(e)}")

    async def get_storage_metrics(self) -> Dict[str, Any]:
        """ストレージ使用状況の取得

        Returns:
            Dict[str, Any]: ストレージメトリクス
                total_size: 総ストレージ使用量
                file_count: ファイル数
                status_counts: ステータス別ファイル数
                daily_uploads: 日別アップロード数

        Raises:
            StorageError: 取得に失敗した場合
        """
        if not self.pool:
            raise StorageError("データベース未初期化")

        try:
            async with self.pool.acquire() as conn:
                # 総ストレージ使用量とファイル数
                storage_query = """
                SELECT 
                    SUM(file_size) as total_size,
                    COUNT(*) as file_count
                FROM audio_files
                """
                storage_stats = await conn.fetchrow(storage_query)

                # ステータス別ファイル数
                status_query = """
                SELECT status, COUNT(*) as count
                FROM audio_files
                GROUP BY status
                """
                status_rows = await conn.fetch(status_query)
                status_counts = {row['status']: row['count'] for row in status_rows}

                # 日別アップロード数（直近7日間）
                upload_query = """
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as count
                FROM audio_files
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                """
                upload_rows = await conn.fetch(upload_query)
                daily_uploads = {
                    str(row['date']): row['count'] 
                    for row in upload_rows
                }

                return {
                    'total_size': storage_stats['total_size'],
                    'file_count': storage_stats['file_count'],
                    'status_counts': status_counts,
                    'daily_uploads': daily_uploads
                }
        except Exception as e:
            raise StorageError(f"メトリクス取得エラー: {str(e)}")