"""
数据加载模块 - 加载GeoBenchmark数据集
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class Question:
    """问题数据结构"""
    id: str
    task_type: str  # choice, tf, completion, noun, qa, discussion
    question: str
    choices: List[Dict[str, str]] | None  # 选择题选项
    reference_answer: str
    metadata: Dict[str, Any]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Question) and self.id == other.id


class GeoBenchmarkLoader:
    """GeoBenchmark数据加载器"""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_apstudy(self) -> List[Question]:
        """加载AP Study数据集（选择题）"""
        path = self.data_dir / "geobenchmark_apstudy.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = []
        for item in data:
            q = item['question']
            questions.append(Question(
                id=item.get('id', f"apstudy_{len(questions)}"),
                task_type="choice",
                question=q['stem'],
                choices=q.get('choices', []),
                reference_answer=item['answerKey'],
                metadata={"dataset": "apstudy"}
            ))
        return questions

    def load_npee(self) -> Dict[str, List[Question]]:
        """加载NPEE数据集（所有题型）"""
        path = self.data_dir / "geobenchmark_npee.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {
            "choice": [],
            "tf": [],
            "completion": [],
            "noun": [],
            "qa": [],
            "discussion": []
        }

        # 任务类型映射
        task_type_map = {
            "choice": "choice",
            "tf": "tf",
            "completion": "completion",
            "noun": "noun",
            "qa": "qa",
            "discussion": "discussion"
        }

        for task_type, content in data.items():
            if task_type not in task_type_map:
                continue

            mapped_type = task_type_map[task_type]
            questions_raw = content.get('question', [])
            answers_raw = content.get('answer', [])

            for i, (q, a) in enumerate(zip(questions_raw, answers_raw)):
                result[mapped_type].append(Question(
                    id=f"npee_{task_type}_{i}",
                    task_type=mapped_type,
                    question=q,
                    choices=None,
                    reference_answer=a,
                    metadata={"dataset": "npee", "original_type": task_type}
                ))

        return result

    def load_all(self) -> Dict[str, List[Question]]:
        """加载所有数据集"""
        ap_questions = self.load_apstudy()
        npee_questions = self.load_npee()

        return {
            "apstudy": ap_questions,
            "npee_choice": npee_questions["choice"],
            "npee_tf": npee_questions["tf"],
            "npee_completion": npee_questions["completion"],
            "npee_noun": npee_questions["noun"],
            "npee_qa": npee_questions["qa"],
            "npee_discussion": npee_questions["discussion"]
        }

    def get_summary(self) -> Dict[str, int]:
        """获取数据集摘要"""
        all_data = self.load_all()
        return {k: len(v) for k, v in all_data.items()}


if __name__ == "__main__":
    loader = GeoBenchmarkLoader("/Users/chenao/Desktop/全量测试")
    summary = loader.get_summary()
    print("数据集摘要:")
    for dataset, count in summary.items():
        print(f"  {dataset}: {count} 题")
    print(f"  总计: {sum(summary.values())} 题")
