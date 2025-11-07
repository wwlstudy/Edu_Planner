"""Utility helpers for EduPlanner prompts and dataset handling.

This module restores functionality that was omitted from the
public release.  The helpers focus on two key tasks required by
``pre_prompt.py``:

* building detailed student personas from the ability tree, and
* sampling questions from the algebra dataset for evaluation.

Both helpers are deterministic when possible so that experiments
can be reproduced reliably.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import re
from tqdm import tqdm

__all__ = [
    "get_students_ability",
    "get_selected_questions",
    "pd",
    "re",
    "tqdm",
]


def _load_ability_tree(ability_tree_path: Path) -> dict:
    """Load and return the ability tree definition.

    Parameters
    ----------
    ability_tree_path:
        Path to the ``ability_tree.json`` file bundled with the
        project.
    """

    try:
        data = json.loads(ability_tree_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Ability tree file not found: {ability_tree_path!s}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in ability tree file: {ability_tree_path!s}"
        ) from exc

    if "ability_tree" not in data:
        raise ValueError("Malformed ability tree: missing 'ability_tree' root key")
    return data["ability_tree"]


def _match_score(description_block: dict, score: int) -> str:
    """Return the description matching ``score`` from an ability block."""

    scores: List[dict] = description_block.get("Score", [])
    if not scores:
        return "Description unavailable."

    for entry in scores:
        if entry.get("score") == score:
            return entry.get("Description", "")

    # Fall back to the closest available description if the exact
    # score is not defined in the data file.
    closest = min(scores, key=lambda item: abs(int(item.get("score", 0)) - score))
    return closest.get("Description", "")


def get_students_ability(ability_tree_path: str | Path, students: Iterable[Sequence[int]]) -> str:
    """Create persona descriptions for the provided ``students``.

    Parameters
    ----------
    ability_tree_path:
        Path to ``ability_tree.json``.
    students:
        Iterable containing ability scores for each student.  Each
        student sequence should have the same number of entries as the
        ability categories in the tree.  Scores outside the documented
        range are clamped.
    """

    tree_path = Path(ability_tree_path)
    tree = _load_ability_tree(tree_path)
    abilities = tree.get("ability", [])

    if not abilities:
        raise ValueError("Ability tree does not define any abilities")

    persona_lines = ["# Student Personas"]
    for index, scores in enumerate(students, start=1):
        persona_lines.append(f"## Student {index}")
        for ability, raw_score in zip(abilities, scores):
            score = max(1, min(int(raw_score), 5))
            description = _match_score(ability, score)
            ability_name = ability.get("Name", "Ability")
            persona_lines.append(
                f"- {ability_name} (Level {score}): {description}"
            )
        persona_lines.append("")

    return "\n".join(persona_lines).strip()


def get_selected_questions(dataset_path: str | Path, question_count: int) -> List[dict]:
    """Sample ``question_count`` unique questions from ``dataset_path``.

    The function returns a list of dictionaries containing the question
    text and the final answer so that callers can decide how to use the
    data inside prompts.
    """

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file!s}")

    df = pd.read_csv(dataset_file)
    if "question" not in df.columns:
        raise ValueError("Dataset must contain a 'question' column")

    sample_size = min(int(question_count), len(df))
    # ``random_state`` ensures reproducible runs while still leveraging
    # pandas' sampling convenience.
    sampled = df.sample(n=sample_size, random_state=42)

    questions: List[dict] = []
    for _, row in sampled.iterrows():
        questions.append(
            {
                "question": str(row.get("question", "")).strip(),
                "answer": row.get("final_answer"),
            }
        )

    return questions
