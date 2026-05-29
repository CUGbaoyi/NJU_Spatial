"""
持久化Judge缓存系统 - SQLite实现
支持跨模型、跨会话复用评判结果
"""
import sqlite3
import hashlib
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager


class JudgeCache:
    """
    Judge评判结果缓存

    特性:
    - 持久化存储(SQLite)
    - 线程安全
    - 自动清理过期缓存
    - 支持缓存统计
    """

    def __init__(self, cache_dir: str | None = None, db_name: str = "judge_cache.db"):
        """
        Args:
            cache_dir: 缓存目录，默认为results/cache
            db_name: 数据库文件名
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "results" / "cache"
        else:
            cache_dir = Path(cache_dir)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / db_name
        self._local = threading.local()
        self._lock = threading.Lock()

        # 初始化数据库
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """获取线程本地数据库连接"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        yield self._local.conn

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS judge_results (
                    cache_key TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    answer_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type
                ON judge_results(task_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON judge_results(created_at)
            """)

            conn.commit()

    def _make_key(
        self,
        task_type: str,
        question: str,
        reference: str,
        answer: str
    ) -> str:
        """
        生成缓存键

        策略: 题型 + 问题hash + 参考答案hash + 模型答案hash
        这样相同问题的不同答案也可以缓存
        """
        combined = f"{task_type}:{question}:{reference}:{answer}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _make_content_hash(self, content: str) -> str:
        """生成内容hash用于去重分析"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(
        self,
        task_type: str,
        question: str,
        reference: str,
        answer: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的评判结果

        Returns:
           评判结果字典，如果未命中返回None
        """
        cache_key = self._make_key(task_type, question, reference, answer)

        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT result_json FROM judge_results WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()

            if row:
                # 更新访问统计
                conn.execute("""
                    UPDATE judge_results
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()

                return json.loads(row['result_json'])

        return None

    def set(
        self,
        task_type: str,
        question: str,
        reference: str,
        answer: str,
        result: Dict[str, Any]
    ) -> None:
        """
        存储评判结果到缓存
        """
        cache_key = self._make_key(task_type, question, reference, answer)
        question_hash = self._make_content_hash(question)
        answer_hash = self._make_content_hash(answer)

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO judge_results
                (cache_key, task_type, question_hash, answer_hash, result_json, created_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)
            """, (cache_key, task_type, question_hash, answer_hash, json.dumps(result, ensure_ascii=False)))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._get_conn() as conn:
            # 总条目数
            total = conn.execute("SELECT COUNT(*) as count FROM judge_results").fetchone()['count']

            # 按题型统计
            by_type = conn.execute("""
                SELECT task_type, COUNT(*) as count
                FROM judge_results
                GROUP BY task_type
            """).fetchall()

            # 命中率统计 (通过access_count估算)
            access_stats = conn.execute("""
                SELECT
                    SUM(access_count) as total_accesses,
                    COUNT(*) as unique_entries
                FROM judge_results
            """).fetchone()

            # 最旧和最新的缓存
            timing = conn.execute("""
                SELECT
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM judge_results
            """).fetchone()

        return {
            "total_entries": total,
            "by_task_type": {row['task_type']: row['count'] for row in by_type},
            "total_accesses": access_stats['total_accesses'] or 0,
            "avg_access_per_entry": (
                access_stats['total_accesses'] / access_stats['unique_entries']
                if access_stats['unique_entries'] > 0 else 0
            ),
            "oldest_entry": timing['oldest'],
            "newest_entry": timing['newest'],
            "cache_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }

    def clear(self, task_type: str | None = None, older_than_days: int | None = None) -> int:
        """
        清理缓存

        Args:
            task_type: 只清理指定题型，None表示清理所有
            older_than_days: 只清理指定天数之前的缓存，None表示不限制时间

        Returns:
            清理的条目数
        """
        with self._get_conn() as conn:
            if task_type and older_than_days:
                cursor = conn.execute("""
                    DELETE FROM judge_results
                    WHERE task_type = ?
                    AND created_at < datetime('now', '-' || ? || ' days')
                """, (task_type, older_than_days))
            elif task_type:
                cursor = conn.execute("""
                    DELETE FROM judge_results
                    WHERE task_type = ?
                """, (task_type,))
            elif older_than_days:
                cursor = conn.execute("""
                    DELETE FROM judge_results
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                """, (older_than_days,))
            else:
                cursor = conn.execute("DELETE FROM judge_results")

            deleted_count = cursor.rowcount
            conn.commit()

            # 清理数据库空间
            conn.execute("VACUUM")

        return deleted_count

    def find_similar_answers(
        self,
        task_type: str,
        question: str,
        reference: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找相似问题的评判结果 (用于分析)

        Args:
            task_type: 题型
            question: 问题文本
            reference: 参考答案
            limit: 返回数量限制

        Returns:
            相似评判结果列表
        """
        question_hash = self._make_content_hash(question)

        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT cache_key, result_json, answer_hash, access_count
                FROM judge_results
                WHERE task_type = ? AND question_hash = ?
                ORDER BY access_count DESC
                LIMIT ?
            """, (task_type, question_hash, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "cache_key": row['cache_key'],
                    "result": json.loads(row['result_json']),
                    "answer_hash": row['answer_hash'],
                    "access_count": row['access_count']
                })

        return results

    def export_cache(self, output_path: str | None = None) -> str:
        """
        导出缓存为JSON文件

        Args:
            output_path: 输出路径，默认为cache_dir/export_<timestamp>.json
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.cache_dir / f"export_{timestamp}.json"
        else:
            output_path = Path(output_path)

        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT task_type, result_json, access_count, created_at
                FROM judge_results
            """)

            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_entries": 0,
                "entries": []
            }

            for row in cursor.fetchall():
                export_data["entries"].append({
                    "task_type": row['task_type'],
                    "result": json.loads(row['result_json']),
                    "access_count": row['access_count'],
                    "created_at": row['created_at']
                })

            export_data["total_entries"] = len(export_data["entries"])

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return str(output_path)

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')
