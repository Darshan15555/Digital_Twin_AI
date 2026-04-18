from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RandomKeySampler:
    max_rows: int
    random_state: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)
    _sample: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self._sample = pd.DataFrame()

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        candidate = frame.copy()
        candidate["_sample_key"] = self._rng.random(len(candidate))
        if self._sample.empty:
            self._sample = candidate.nsmallest(self.max_rows, "_sample_key")
            return
        merged = pd.concat([self._sample, candidate], ignore_index=True)
        self._sample = merged.nsmallest(self.max_rows, "_sample_key")

    def result(self) -> pd.DataFrame:
        if self._sample.empty:
            return self._sample.copy()
        return self._sample.drop(columns=["_sample_key"]).reset_index(drop=True)


@dataclass(slots=True)
class StratifiedKeySampler:
    quotas: dict[int, int]
    random_state: int = 42
    label_col: str = "target_label"
    _rng: np.random.Generator = field(init=False, repr=False)
    _per_class: dict[int, pd.DataFrame] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self._per_class: dict[int, pd.DataFrame] = {
            int(label): pd.DataFrame() for label, limit in self.quotas.items() if int(limit) > 0
        }

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty or self.label_col not in frame.columns:
            return

        for label, limit in self.quotas.items():
            if limit <= 0:
                continue
            subset = frame[frame[self.label_col] == label]
            if subset.empty:
                continue

            candidate = subset.copy()
            candidate["_sample_key"] = self._rng.random(len(candidate))

            current = self._per_class.get(label)
            if current is None or current.empty:
                self._per_class[label] = candidate.nsmallest(limit, "_sample_key")
                continue

            merged = pd.concat([current, candidate], ignore_index=True)
            self._per_class[label] = merged.nsmallest(limit, "_sample_key")

    def result(self) -> pd.DataFrame:
        frames = [frame for frame in self._per_class.values() if not frame.empty]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        if "_sample_key" in merged.columns:
            merged = merged.drop(columns=["_sample_key"])
        return merged.reset_index(drop=True)


def compute_stratified_quotas(label_counts: dict[int, int], total_rows: int) -> dict[int, int]:
    if total_rows <= 0 or not label_counts:
        return {}

    total = sum(max(0, int(count)) for count in label_counts.values())
    if total <= 0:
        return {}

    raw = {int(label): (int(count) / total) * total_rows for label, count in label_counts.items() if int(count) > 0}
    floors = {label: int(np.floor(value)) for label, value in raw.items()}
    remainder = total_rows - sum(floors.values())

    if remainder > 0:
        order = sorted(raw.items(), key=lambda item: item[1] - np.floor(item[1]), reverse=True)
        for label, _ in order[:remainder]:
            floors[label] += 1

    for label, count in label_counts.items():
        floors[label] = min(floors.get(label, 0), int(count))

    return floors
