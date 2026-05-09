from pathlib import Path
import sys

import pandas as pd
import pytest

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from experiments.also.evaluate_by_tag import (
    build_summary_dataframe,
    export_multi_evalset_summary_xlsx,
)


def _build_evalset_inputs() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    all_df = pd.DataFrame(
        [
            {
                "experiment": "tag_a",
                "db_episode_count": 10,
                "p1_goal": 0.8,
                "p2_goal": 0.6,
                "p1_relationship": 0.3,
                "p2_relationship": 0.5,
                "p1_knowledge": 0.7,
                "p2_knowledge": 0.9,
                "avg_final": 0.75,
                "p1_final_se": 0.11,
                "p2_final_se": 0.13,
                "p1_final_std": 0.0,
                "p2_final_std": 0.0,
            },
            {
                "experiment": "tag_b",
                "db_episode_count": 5,
                "p1_goal": 0.4,
                "p2_goal": 0.8,
                "p1_relationship": 0.6,
                "p2_relationship": 0.2,
                "p1_knowledge": 0.1,
                "p2_knowledge": 0.5,
                "avg_final": 0.33,
                "p1_final_se": 0.21,
                "p2_final_se": 0.23,
                "p1_final_std": 0.0,
                "p2_final_std": 0.0,
            },
        ]
    )
    all_dim_df = pd.DataFrame(
        [
            {
                "experiment": "tag_a",
                "db_episodes": 10,
                "p1_goal_se": 0.10,
                "p2_goal_se": 0.20,
                "p1_relationship_se": 0.30,
                "p2_relationship_se": 0.40,
                "p1_knowledge_se": 0.50,
                "p2_knowledge_se": 0.60,
                "p1_overall_se": 0.70,
                "p2_overall_se": 0.80,
            },
            {
                "experiment": "tag_b",
                "db_episodes": 5,
                "p1_goal_se": 0.15,
                "p2_goal_se": 0.05,
                "p1_relationship_se": 0.25,
                "p2_relationship_se": 0.35,
                "p1_knowledge_se": 0.45,
                "p2_knowledge_se": 0.15,
                "p1_overall_se": 0.55,
                "p2_overall_se": 0.25,
            },
        ]
    )

    hard_df = pd.DataFrame(
        [
            {
                "experiment": "tag_a",
                "db_episode_count": 4,
                "p1_goal": 1.0,
                "p2_goal": 0.2,
                "p1_relationship": 0.4,
                "p2_relationship": 0.8,
                "p1_knowledge": 0.6,
                "p2_knowledge": 0.4,
                "avg_final": 0.5,
                "p1_final_std": 0.4,
                "p2_final_std": 0.2,
                "p1_final_se": 0.0,
                "p2_final_se": 0.0,
            },
            {
                "experiment": "tag_b",
                "db_episode_count": 8,
                "p1_goal": 0.5,
                "p2_goal": 0.1,
                "p1_relationship": 0.9,
                "p2_relationship": 0.7,
                "p1_knowledge": 0.3,
                "p2_knowledge": 0.1,
                "avg_final": 0.2,
                "p1_final_std": 0.6,
                "p2_final_std": 0.2,
                "p1_final_se": 0.0,
                "p2_final_se": 0.0,
            },
        ]
    )
    hard_dim_df = pd.DataFrame(
        [
            {
                "experiment": "tag_a",
                "db_episodes": 4,
                "p1_goal_std": 0.4,
                "p2_goal_std": 0.2,
                "p1_relationship_std": 0.6,
                "p2_relationship_std": 0.8,
                "p1_knowledge_std": 0.2,
                "p2_knowledge_std": 0.4,
                "p1_overall_std": 1.0,
                "p2_overall_std": 0.6,
            },
            {
                "experiment": "tag_b",
                "db_episodes": 8,
                "p1_goal_std": 0.3,
                "p2_goal_std": 0.4,
                "p1_relationship_std": 0.5,
                "p2_relationship_std": 0.1,
                "p1_knowledge_std": 0.9,
                "p2_knowledge_std": 0.3,
                "p1_overall_std": 0.6,
                "p2_overall_std": 0.2,
            },
        ]
    )

    return {
        "all": (all_df, all_dim_df),
        "hard": (hard_df, hard_dim_df),
    }


def test_build_multi_evalset_summary_dataframe():
    summary_df = build_summary_dataframe(_build_evalset_inputs())

    assert list(summary_df.columns) == [
        "No.",
        "SOTOPIA Goal",
        "SOTOPIA Goal SE",
        "SOTOPIA Relation",
        "SOTOPIA Relation SE",
        "SOTOPIA Knowledge",
        "SOTOPIA Knowledge SE",
        "SOTOPIA Overall",
        "SOTOPIA Overall SE",
        "SOTOPIA-Hard Goal",
        "SOTOPIA-Hard Goal SE",
        "SOTOPIA-Hard Relation",
        "SOTOPIA-Hard Relation SE",
        "SOTOPIA-Hard Knowledge",
        "SOTOPIA-Hard Knowledge SE",
        "SOTOPIA-Hard Overall",
        "SOTOPIA-Hard Overall SE",
        "tag",
    ]
    assert summary_df["No."].tolist() == [1, 2]
    assert summary_df["tag"].tolist() == ["tag_a", "tag_b"]

    row_a = summary_df.iloc[0]
    assert row_a["SOTOPIA Goal"] == 0.7
    assert row_a["SOTOPIA Goal SE"] == 0.1118
    assert row_a["SOTOPIA Relation"] == 0.4
    assert row_a["SOTOPIA Relation SE"] == 0.25
    assert row_a["SOTOPIA Knowledge"] == 0.8
    assert row_a["SOTOPIA Knowledge SE"] == 0.3905
    assert row_a["SOTOPIA Overall"] == 0.75
    assert row_a["SOTOPIA Overall SE"] == 0.5315
    assert row_a["SOTOPIA-Hard Goal"] == 0.6
    assert row_a["SOTOPIA-Hard Goal SE"] == 0.0559
    assert row_a["SOTOPIA-Hard Relation"] == 0.6
    assert row_a["SOTOPIA-Hard Relation SE"] == 0.125
    assert row_a["SOTOPIA-Hard Knowledge"] == 0.5
    assert row_a["SOTOPIA-Hard Knowledge SE"] == 0.0559
    assert row_a["SOTOPIA-Hard Overall"] == 0.5
    assert row_a["SOTOPIA-Hard Overall SE"] == 0.1458


def test_export_multi_evalset_summary_xlsx(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    summary_df = build_summary_dataframe(_build_evalset_inputs())
    output_path = tmp_path / "summary.xlsx"

    export_multi_evalset_summary_xlsx(summary_df, output_path)

    loaded_df = pd.read_excel(output_path)
    workbook = openpyxl.load_workbook(output_path)
    worksheet = workbook["summary"]

    assert output_path.exists()
    assert list(loaded_df.columns) == list(summary_df.columns)
    assert loaded_df.to_dict(orient="records") == summary_df.to_dict(orient="records")
    assert worksheet["B2"].fill.fgColor.rgb == "00FFF2CC"
    assert worksheet["B3"].fill.fgColor.rgb == "00D9EAD3"
    assert worksheet["C2"].fill.fgColor.rgb in {"00000000", None}


def test_build_single_evalset_summary_dataframe():
    summary_df = build_summary_dataframe({"default": _build_evalset_inputs()["all"]})

    assert list(summary_df.columns) == [
        "No.",
        "Goal",
        "Goal SE",
        "Relation",
        "Relation SE",
        "Knowledge",
        "Knowledge SE",
        "Overall",
        "Overall SE",
        "tag",
    ]
    assert summary_df.iloc[0].to_dict() == {
        "No.": 1,
        "Goal": 0.7,
        "Goal SE": 0.1118,
        "Relation": 0.4,
        "Relation SE": 0.25,
        "Knowledge": 0.8,
        "Knowledge SE": 0.3905,
        "Overall": 0.75,
        "Overall SE": 0.5315,
        "tag": "tag_a",
    }
