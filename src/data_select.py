import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from datasets import load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    config: Optional[str]
    difficulty: str


class DataSelector:
    FIXED_DATASET_SPECS: Sequence[DatasetSpec] = (
        DatasetSpec(
            name="aime25",
            dataset_id="TIGER-Lab/AIME25",
            config=None,
            difficulty="hard",
        ),
        DatasetSpec(
            name="aime24",
            dataset_id="OpenEvals/aime_24",
            config=None,
            difficulty="hard",
        ),
    )

    DYNAMIC_DATASET_CANDIDATES: Sequence[DatasetSpec] = (
        DatasetSpec(
            name="dynamic_hmmt_feb_2025",
            dataset_id="MathArena/hmmt_feb_2025",
            config=None,
            difficulty="dynamic",
        ),
        DatasetSpec(
            name="dynamic_hmmt_nov_2025",
            dataset_id="MathArena/hmmt_nov_2025",
            config=None,
            difficulty="dynamic",
        ),
        DatasetSpec(
            name="dynamic_imo_answerbench",
            dataset_id="Hwilner/imo-answerbench",
            config=None,
            difficulty="dynamic",
        ),
    )

    def __init__(self, mode: str = "sample"):
        self.mode = mode
        random.seed(42)

    def _select_split(self, dataset_obj) -> tuple[str, List[Dict[str, Any]]]:
        if hasattr(dataset_obj, "keys"):
            for split_name in ("test", "validation", "train"):
                if split_name in dataset_obj:
                    return split_name, list(dataset_obj[split_name])
            first_split = next(iter(dataset_obj.keys()))
            return first_split, list(dataset_obj[first_split])
        return "default", list(dataset_obj)

    def _load_dataset(self, dataset_id: str, config: Optional[str]) -> List[Dict[str, Any]]:
        if config:
            dataset_obj = load_dataset(dataset_id, config)
        else:
            dataset_obj = load_dataset(dataset_id)

        split_name, items = self._select_split(dataset_obj)
        return items

    def select_and_sample_data(self) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        for spec in self.FIXED_DATASET_SPECS:
            items = self._load_dataset(spec.dataset_id, spec.config)

            if self.mode == "full":
                sampled_items = items
            else:
                sampled_items = random.sample(items, min(4, len(items)))

            for idx, item in enumerate(sampled_items):
                row = dict(item)
                row["_dataset"] = spec.name
                row["_dataset_id"] = spec.dataset_id
                row["_difficulty"] = spec.difficulty
                row["_sample_index"] = idx
                all_items.append(row)

        if len(self.DYNAMIC_DATASET_CANDIDATES) >= 1:
            selected_dynamic = random.sample(list(self.DYNAMIC_DATASET_CANDIDATES), 1)
        else:
            selected_dynamic = list(self.DYNAMIC_DATASET_CANDIDATES)

        for spec in selected_dynamic:
            items = self._load_dataset(spec.dataset_id, spec.config)

            if self.mode == "full":
                sampled_items = items
            else:
                sampled_items = random.sample(items, min(4, len(items)))

            for idx, item in enumerate(sampled_items):
                row = dict(item)
                row["_dataset"] = spec.name
                row["_dataset_id"] = spec.dataset_id
                row["_difficulty"] = spec.difficulty
                row["_sample_index"] = idx
                all_items.append(row)

        return all_items

    def save_data(self, data: List[Dict[str, Any]], output_path: str):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已选择并保存 {len(data)} 条数据到: {output_path}")

    def get_dataset_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        dataset_counts = {}
        for item in data:
            dataset = item.get("_dataset", "unknown")
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        return dataset_counts

    def print_statistics(self, data: List[Dict[str, Any]]):
        dataset_counts = self.get_dataset_statistics(data)
        for dataset, count in dataset_counts.items():
            print(f"  - {dataset}: {count} 条")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="选择数学数据集数据")
    parser.add_argument("--mode", type=str, default="full", choices=["sample", "full"])
    parser.add_argument("--output", type=str, default="DataFlywheel/data/benchmark.json")

    args = parser.parse_args()
    selector = DataSelector(mode=args.mode)
    data = selector.select_and_sample_data()
    selector.save_data(data, args.output)
    selector.print_statistics(data)


if __name__ == "__main__":
    main()