"""
断点续传管理 - 支持评测进度保存与恢复
"""
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import threading

@dataclass
class TaskState:
    """单个任务状态"""
    question_id: str
    model_name: str
    status: str  # pending, running, completed, failed
    score: Optional[float] = None
    model_answer: Optional[str] = None
    judge_reason: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


class CheckpointManager:
    """断点续传管理器（线程安全）"""

    def __init__(self, checkpoint_path: str = "results/eval_checkpoint.json"):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, TaskState]] = {}
        self._load()

    def _load(self):
        """从文件加载状态"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for model, tasks in raw.items():
                        self._data[model] = {
                            qid: TaskState(**v) for qid, v in tasks.items()
                        }
                print(f"从断点恢复: {self.checkpoint_path}")
            except Exception as e:
                print(f"加载断点失败: {e}，将创建新文件")
                self._data = {}

    def _save(self):
        """保存状态到文件"""
        serializable = {}
        for model, tasks in self._data.items():
            serializable[model] = {
                qid: asdict(state) for qid, state in tasks.items()
            }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def get_completed(self, model_name: str) -> Set[str]:
        """获取已完成的题目ID"""
        with self._lock:
            if model_name not in self._data:
                return set()
            return {
                qid for qid, state in self._data[model_name].items()
                if state.status == "completed"
            }

    def is_completed(self, model_name: str, question_id: str) -> bool:
        """检查题目是否已完成"""
        with self._lock:
            return question_id in self._data.get(model_name, {})

    def get_state(self, model_name: str, question_id: str) -> Optional[TaskState]:
        """获取题目状态"""
        with self._lock:
            return self._data.get(model_name, {}).get(question_id)

    def update_state(self, model_name: str, state: TaskState):
        """更新题目状态"""
        with self._lock:
            if model_name not in self._data:
                self._data[model_name] = {}
            state.timestamp = datetime.now().isoformat()
            self._data[model_name][state.question_id] = state
            self._save()

    def mark_completed(
        self,
        question_id: str,
        model_name: str,
        score: float,
        model_answer: str,
        judge_reason: Optional[str] = None
    ):
        """标记题目完成"""
        state = TaskState(
            question_id=question_id,
            model_name=model_name,
            status="completed",
            score=score,
            model_answer=model_answer,
            judge_reason=judge_reason
        )
        self.update_state(model_name, state)

    def mark_failed(self, question_id: str, model_name: str, error: str):
        """标记题目失败"""
        state = TaskState(
            question_id=question_id,
            model_name=model_name,
            status="failed",
            error=error
        )
        self.update_state(model_name, state)

    def get_progress(self, model_name: str, total_questions: int) -> Dict[str, int]:
        """获取进度统计"""
        with self._lock:
            if model_name not in self._data:
                return {"total": total_questions, "completed": 0, "failed": 0}
            states = list(self._data[model_name].values())
            return {
                "total": total_questions,
                "completed": sum(1 for s in states if s.status == "completed"),
                "failed": sum(1 for s in states if s.status == "failed")
            }

    def get_all_results(self, model_name: str) -> List[TaskState]:
        """获取模型所有结果"""
        with self._lock:
            return list(self._data.get(model_name, {}).values())

    def clear_model(self, model_name: str):
        """清除模型的所有记录"""
        with self._lock:
            if model_name in self._data:
                del self._data[model_name]
                self._save()

    def export_to_csv(self, model_name: str, questions: Dict[str, Any], output_path: str):
        """导出结果到CSV"""
        results = self.get_all_results(model_name)
        if not results:
            print(f"模型 {model_name} 没有结果可导出")
            return

        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "question_id", "task_type", "question", "reference_answer",
                "model_answer", "score", "judge_reason", "status"
            ])
            for state in results:
                q_info = questions.get(state.question_id, {})
                writer.writerow([
                    state.question_id,
                    q_info.get("task_type", ""),
                    q_info.get("question", "")[:100],
                    q_info.get("reference_answer", "")[:100],
                    (state.model_answer or "")[:200],
                    state.score,
                    state.judge_reason,
                    state.status
                ])
        print(f"已导出CSV: {output_path}")
