"""入力層コントローラーモジュール"""

import logging
from typing import Dict, Any, BinaryIO, Optional, Callable
from datetime import datetime
import threading
from queue import Empty
import time

from .exceptions import InputLayerError, ValidationError, StorageError, QueueError
from .uploader import FileUploader
from .queue_manager import InputQueueManager, QueueItem

logger = logging.getLogger(__name__)

class InputController:
    """入力層コントローラー"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 入力層設定
                validation: バリデーション設定
                storage: ストレージ設定
                queue: キュー設定
        """
        self.config = config
        self.file_uploader = FileUploader(config)
        self.queue_manager = InputQueueManager(config['queue'])
        
        # コールバック関数
        self._processing_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._error_callback: Optional[Callable[[str, Exception], None]] = None
        
        # 処理スレッド
        self._processing_thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()

    def start(self):
        """処理の開始"""
        if self._processing_thread is None or not self._processing_thread.is_alive():
            self._should_stop.clear()
            self._processing_thread = threading.Thread(
                target=self._processing_worker,
                daemon=True
            )
            self._processing_thread.start()
            logger.info("入力処理スレッドを開始しました")

    def stop(self):
        """処理の停止"""
        if self._processing_thread and self._processing_thread.is_alive():
            self._should_stop.set()
            self._processing_thread.join()
            logger.info("入力処理スレッドを停止しました")

    def set_processing_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """処理コールバックの設定

        Args:
            callback: コールバック関数
                     引数: (file_id, metadata)
        """
        self._processing_callback = callback

    def set_error_callback(self, callback: Callable[[str, Exception], None]):
        """エラーコールバックの設定

        Args:
            callback: コールバック関数
                     引数: (file_id, error)
        """
        self._error_callback = callback

    def handle_file_upload(
        self,
        file: BinaryIO,
        metadata: Dict[str, Any] = None,
        priority: int = 0
    ) -> Dict[str, Any]:
        """ファイルアップロードの処理

        Args:
            file: アップロードするファイル
            metadata: メタデータ
            priority: 処理優先度（低いほど優先）

        Returns:
            Dict[str, Any]: 処理結果
                file_id: ファイルID
                status: 処理状態
                upload_time: アップロード時刻
                metadata: 更新されたメタデータ

        Raises:
            InputLayerError: 処理エラーの場合
        """
        try:
            # ファイルのアップロードと検証
            upload_result = self.file_uploader.upload_file(file, metadata)
            file_id = upload_result['file_id']
            
            # キューへの追加
            self.queue_manager.add_to_queue(
                file_id=file_id,
                priority=priority,
                metadata=upload_result['metadata']
            )
            
            return {
                'file_id': file_id,
                'status': 'queued',
                'upload_time': datetime.now().isoformat(),
                'metadata': upload_result['metadata']
            }
            
        except (ValidationError, StorageError, QueueError) as e:
            logger.error(f"ファイルアップロードエラー: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"予期せぬエラー: {str(e)}", exc_info=True)
            raise InputLayerError(f"ファイル処理エラー: {str(e)}")

    def get_file_status(self, file_id: str) -> Dict[str, Any]:
        """ファイル状態の取得

        Args:
            file_id: ファイルID

        Returns:
            Dict[str, Any]: ファイルの状態情報
        """
        try:
            metadata = self.file_uploader.get_file_status(file_id)
            queue_status = self.queue_manager.get_queue_status()
            
            status = 'unknown'
            if file_id in queue_status['processing_files']:
                status = 'processing'
            elif file_id in queue_status['failed_files']:
                status = 'failed'
                metadata['retry_info'] = queue_status['failed_files'][file_id]
            
            return {
                'file_id': file_id,
                'status': status,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"状態取得エラー: {str(e)}", exc_info=True)
            raise InputLayerError(f"状態取得エラー: {str(e)}")

    def cancel_processing(self, file_id: str) -> bool:
        """処理のキャンセル

        Args:
            file_id: キャンセルするファイルのID

        Returns:
            bool: キャンセル成功の場合True
        """
        try:
            # キューから削除
            self.queue_manager.mark_as_completed(file_id)
            # ファイルの削除
            self.file_uploader.delete_file(file_id)
            return True
        except Exception as e:
            logger.error(f"処理キャンセルエラー: {str(e)}", exc_info=True)
            return False

    def _processing_worker(self):
        """処理ワーカー"""
        while not self._should_stop.is_set():
            try:
                # 次の処理対象を取得
                item = self.queue_manager.get_next_file()
                if item is None:
                    time.sleep(1)  # キューが空の場合は待機
                    continue
                
                try:
                    # コールバックの実行
                    if self._processing_callback:
                        self._processing_callback(item.file_id, item.metadata)
                    
                    # 処理完了の記録
                    self.queue_manager.mark_as_completed(item.file_id)
                    
                except Exception as e:
                    logger.error(f"処理エラー: {str(e)}", exc_info=True)
                    # エラーコールバックの実行
                    if self._error_callback:
                        self._error_callback(item.file_id, e)
                    # 失敗の記録と再試行の判断
                    self.queue_manager.mark_as_failed(item.file_id, e)
                
            except Exception as e:
                logger.error(f"ワーカーエラー: {str(e)}", exc_info=True)
                time.sleep(1)  # エラー時は少し待機

    def get_queue_metrics(self) -> Dict[str, Any]:
        """キューメトリクスの取得

        Returns:
            Dict[str, Any]: キューの状態メトリクス
        """
        queue_status = self.queue_manager.get_queue_status()
        return {
            'queue_size': queue_status['queue_size'],
            'processing_count': queue_status['processing_count'],
            'failed_count': queue_status['failed_count'],
            'timestamp': datetime.now().isoformat()
        }