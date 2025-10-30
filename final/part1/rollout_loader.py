from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import pickle
import re

@dataclass
class Rollout:
    idx: int
    path: str
    time: List[float]
    q_mes_all: List[List[float]]
    q_d_all: List[List[float]]
    tau_mes_all: List[List[float]]
    tau_cmd_all: List[List[float]] 

    @classmethod
    def from_dict(cls, idx: int, path: Path, d: Dict[str, Any]) -> "Rollout":
        required = ["time", "q_mes_all", "q_d_all", "tau_mes_all", "tau_cmd_all"]
        
        missing = [k for k in required if k not in d]
        if missing:
            if 'tau_cmd_all' in missing and len(missing) == 1:
                print(f"Warning: '{path}' is missing 'tau_cmd_all'. Using 'tau_mes_all' as a fallback.")
                d['tau_cmd_all'] = d['tau_mes_all']
            else:
                raise ValueError(f"{path} is missing keys: {missing}")
        
        return cls(
            idx=idx,
            path=str(path),
            time=d["time"],
            q_mes_all=d["q_mes_all"],
            q_d_all=d["q_d_all"],
            tau_mes_all=d["tau_mes_all"],
            tau_cmd_all=d["tau_cmd_all"],
        )

def _default_final_dir() -> Path:
    return Path(__file__).resolve().parent

def _find_by_index(directory: Path, i: int) -> Optional[Path]:
    for name in (f"data_{i}.pkl", f"{i}.pkl"):
        p = directory / name
        if p.is_file():
            return p
    return None

def _discover_all(directory: Path) -> List[tuple[int, Path]]:
    items: List[tuple[int, Path]] = []
    pat = re.compile(r"^(?:data_)?(\d+)\.pkl$")
    for p in directory.iterdir():
        if p.suffix == ".pkl":
            m = pat.match(p.name)
            if m:
                items.append((int(m.group(1)), p))
    items.sort(key=lambda t: t[0])
    return items

def load_rollouts(
    count: Optional[int] = None,
    indices: Optional[Iterable[int]] = None,
    directory: Optional[str | Path] = None,
    strict_missing: bool = False,
) -> List[Rollout]:
    dir_path = Path(directory) if directory is not None else _default_final_dir()
    targets: List[tuple[int, Path]] = []
    if indices is not None:
        for i in sorted(set(indices)):
            p = _find_by_index(dir_path, i)
            if p is None:
                if strict_missing:
                    raise FileNotFoundError(f"Missing rollout index {i} in {dir_path}")
            else:
                targets.append((i, p))
    elif count is not None:
        for i in range(1, count + 1):
            p = _find_by_index(dir_path, i)
            if p is None:
                if strict_missing:
                    raise FileNotFoundError(f"Missing rollout index {i} in {dir_path}")
            else:
                targets.append((i, p))
    else:
        targets = _discover_all(dir_path)
        if not targets:
            raise FileNotFoundError(f"No rollout .pkl files found in {dir_path}")
    rollouts: List[Rollout] = []
    for i, p in targets:
        with p.open("rb") as f:
            data = pickle.load(f)
        rollouts.append(Rollout.from_dict(i, p, data))
    rollouts.sort(key=lambda r: r.idx)
    return rollouts