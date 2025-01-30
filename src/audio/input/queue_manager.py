"""入力キュー管理モジュール"""

from queue import PriorityQueue, Empty, Full
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set
import threading
import time

from .exceptions import QueueError

@dataclass
class QueueItem:
    """キューアイテム"""
    priority: int
    file_id: str
    timestamp: datetime
    retry_count: int = 0
    metadata: Dict[str, Any] = None

    def __lt__(self, other):
        return self.priority < other.priority

class InputQueueManager:
    """入力キュー管理クラス"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: キュー設定
                max_queue_size: 最大キューサイズ
                cleanup_interval: クリーンアップ間隔（秒）
                retry_count: 最大再試行回数
                retry_delay: 再試行間隔（秒）
        """
        self.queue = PriorityQueue(maxsize=config['max_queue_size'])
        self.processing_files: Set[str] = set()
        self.failed_files: Dict[str, QueueItem] = {}
        
        self.max_retry_count = config['retry_count']
        self.retry_delay = config['retry_delay']
        self.cleanup_interval = config['cleanup_interval']
        
        # クリーンアップスレッドの開始
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """クリーンアップスレッドの開始"""
        def cleanup_worker():
            while True:
                self._cleanup_old_entries()
                time.sleep(self.cleanup_interval)
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()

    def _cleanup_old_entries(self):
        """古いエントリのクリーンアップ"""
        current_time = datetime.now()
        # 24時間以上経過したファイルを削除
        expired_time = current_time - timedelta(hours=24)
        
        # 失敗したファイルのクリーンアップ
        expired_files = [
            file_id for file_id, item in self.failed_files.items()
            if item.timestamp < expired_time
        ]
        for file_id in expired_files:
            del self.failed_files[file_id]

    def add_to_queue(self, file_id: str, priority: int = 0, metadata: Dict[str, Any] = None) -> bool:
        """キューへのファイル追加

        Args:
            file_id: ファイルID
            priority: 優先度（低い値ほど優先度が高い）
            metadata: メタデータ

        Returns:
            bool: 追加成功の場合True

        Raises:
            QueueError: キューが満杯の場合
        """
        try:
            item = QueueItem(
                priority=priority,
                file_id=file_id,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.queue.put_nowait(item)
            return True
        except Full:
            raise QueueError(
                "処理キューが満杯です",
                {"queue_size": self.queue.qsize()}
            )

    def get_next_file(self) -> Optional[QueueItem]:
        """次の処理対象ファイルの取得

        Returns:
            Optional[QueueItem]: 次の処理対象ファイル。キューが空の場合はNone
        """
        try:
            item = self.queue.get_nowait()
            self.processing_files.add(item.file_id)
            return item
        except Empty:
            return None

    def mark_as_completed(self, file_id: str):
        """処理完了の記録

        Args:
            file_id: 完了したファイルのID
        """
        if file_id in self.processing_files:
            self.processing_files.remove(file_id)
        if file_id in self.failed_files:
            del self.failed_files[file_id]

    def mark_as_failed(self, file_id: str, error: Exception = None):
        """処理失敗の記録

        Args:
            file_id: 失敗したファイルのID
            error: エラー情報

        Returns:
            bool: 再試行が予定される場合True
        """
        if file_id in self.processing_files:
            self.processing_files.remove(file_id)

        # 失敗情報の取得または作成
        failed_item = self.failed_files.get(file_id)
        if failed_item is None:
            # キューから取得できない場合は新規作成
            failed_item = QueueItem(
                priority=0,
                file_id=file_id,
                timestamp=datetime.now()
            )

        failed_item.retry_count += 1
        failed_item.timestamp = datetime.now()

        # 再試行判定
        if failed_item.retry_count <= self.max_retry_count:
            # 再試行のスケジュール
            self.failed_files[file_id] = failed_item
            # 遅延を付けて再キュー
            threading.Timer(
                self.retry_delay,
                lambda: self.retry_failed_file(file_id)
            ).start()
            return True
        else:
            # 最大再試行回数を超えた場合
            if file_id in self.failed_files:
                del self.failed_files[file_id]
            return False

    def retry_failed_file(self, file_id: str):
        """失敗したファイルの再試行

        Args:
            file_id: 再試行するファイルのID
        """
        if file_id in self.failed_files:
            item = self.failed_files[file_id]
            try:
                # 優先度を下げて再キュー
                self.queue.put_nowait(item)
                del self.failed_files[file_id]
            except Full:
                # キューが満杯の場合は後で再試行
                pass

    def get_queue_status(self) -> Dict[str, Any]:
        """キューの状態取得

        Returns:
            Dict[str, Any]: キューの状態情報
        """
        return {
            "queue_size": self.queue.qsize(),
            "processing_count": len(self.processing_files),
            "failed_count": len(self.failed_files),
            "processing_files": list(self.processing_files),
            "failed_files": {
                file_id: {
                    "retry_count": item.retry_count,
                    "last_retry": item.timestamp.isoformat()
                }
                for file_id, item in self.failed_files.items()
            }
        }