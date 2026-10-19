from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_COLUMNS = {
    "sample_id",
    "vessel_name",
    "call_sign",
    "gross_tonnage",
    "international_gross_tonnage",
    "flag_name",
    "vessel_type",
    "arrival_year",
    "target_vessel_id",
}
CATALOG_COLUMNS = {
    "vessel_id",
    "vessel_type",
    "vessel_name",
    "size",
    "unit",
    "deadweight_tonnage",
    "gross_tonnage",
    "flag_name",
}
SUBMISSION_COLUMNS = {"sample_id", "target_vessel_id"}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="항만 선박 매칭 대회 1차 EDA")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--result-dir", type=Path, default=project_dir / "results")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper().strip()
    return re.sub(r"[^0-9A-Z가-힣]", "", text)


def is_hangul_name(value: object) -> bool:
    return bool(re.search(r"[가-힣]", str(value)))


def check_columns(df: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(missing)}")


def table_profile(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        rows.append(
            {
                "table": table_name,
                "column": column,
                "dtype": str(df[column].dtype),
                "rows": len(df),
                "missing_count": int(df[column].isna().sum()),
                "missing_ratio": float(df[column].isna().mean()),
                "unique_count": int(df[column].nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def mapping_profile(train: pd.DataFrame, column: str) -> dict[str, int | float]:
    grouped = train.groupby(column, dropna=False)["target_vessel_id"].agg(rows="size", targets="nunique")
    collision = grouped["targets"] > 1
    return {
        "unique_keys": int(len(grouped)),
        "collision_keys": int(collision.sum()),
        "rows_in_collision_keys": int(grouped.loc[collision, "rows"].sum()),
        "max_targets_per_key": int(grouped["targets"].max()),
    }


def save_plots(train: pd.DataFrame, profile: pd.DataFrame, result_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    train_missing = profile[profile["table"] == "train"].sort_values("missing_ratio", ascending=False)
    plt.bar(train_missing["column"], train_missing["missing_ratio"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing ratio")
    plt.title("Train missing-value ratio")
    plt.tight_layout()
    plt.savefig(result_dir / "01_train_missing_ratio.png", dpi=160)
    plt.close()

    target_counts = train.groupby("target_vessel_id").size()
    repeat_hist = target_counts.value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(repeat_hist.index.astype(str), repeat_hist.values)
    plt.xlabel("Records per vessel")
    plt.ylabel("Number of vessels")
    plt.title("Target vessel repeat distribution")
    plt.tight_layout()
    plt.savefig(result_dir / "02_target_repeat_distribution.png", dpi=160)
    plt.close()

    year_counts = train["arrival_year"].value_counts().sort_index()
    plt.figure(figsize=(9, 4))
    plt.bar(year_counts.index.astype(str), year_counts.values)
    plt.xlabel("Arrival year")
    plt.ylabel("Rows")
    plt.title("Train rows by arrival year")
    plt.tight_layout()
    plt.savefig(result_dir / "03_rows_by_arrival_year.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    submission_path = data_dir / "sample_submission.csv"
    for path in [train_path, catalog_path, submission_path]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    submission = pd.read_csv(submission_path, encoding="utf-8-sig")

    check_columns(train, TRAIN_COLUMNS, "train.csv")
    check_columns(catalog, CATALOG_COLUMNS, "vessel_catalog.csv")
    check_columns(submission, SUBMISSION_COLUMNS, "sample_submission.csv")

    profile = pd.concat(
        [
            table_profile(train, "train"),
            table_profile(catalog, "vessel_catalog"),
            table_profile(submission, "sample_submission"),
        ],
        ignore_index=True,
    )
    profile.to_csv(result_dir / "column_profile.csv", index=False, encoding="utf-8-sig")

    target_counts = train.groupby("target_vessel_id").size()
    target_distribution = target_counts.value_counts().sort_index().rename_axis("records_per_vessel").reset_index(name="vessel_count")
    target_distribution.to_csv(result_dir / "target_repeat_distribution.csv", index=False, encoding="utf-8-sig")

    year_summary = train.groupby("arrival_year").agg(rows=("sample_id", "size"), unique_vessels=("target_vessel_id", "nunique")).reset_index()
    year_summary.to_csv(result_dir / "year_summary.csv", index=False, encoding="utf-8-sig")

    joined = train.merge(
        catalog[["vessel_id", "vessel_name", "gross_tonnage", "flag_name", "vessel_type"]],
        left_on="target_vessel_id",
        right_on="vessel_id",
        how="left",
        suffixes=("_input", "_catalog"),
        validate="many_to_one",
    )
    input_gt = pd.to_numeric(joined["gross_tonnage_input"], errors="coerce")
    catalog_gt = pd.to_numeric(joined["gross_tonnage_catalog"], errors="coerce")
    both_gt = input_gt.notna() & catalog_gt.notna()
    gt_exact = pd.Series(False, index=joined.index)
    gt_exact.loc[both_gt] = np.isclose(input_gt.loc[both_gt], catalog_gt.loc[both_gt], rtol=0, atol=1e-9)

    input_name_norm = joined["vessel_name_input"].map(normalize_text)
    catalog_name_norm = joined["vessel_name_catalog"].map(normalize_text)
    input_flag_norm = joined["flag_name_input"].map(normalize_text)
    catalog_flag_norm = joined["flag_name_catalog"].map(normalize_text)
    input_type_norm = joined["vessel_type_input"].map(normalize_text)
    catalog_type_norm = joined["vessel_type_catalog"].map(normalize_text)

    hangul_mask = train["vessel_name"].map(is_hangul_name)
    train_targets = set(train["target_vessel_id"].astype(str))
    catalog_targets = set(catalog["vessel_id"].astype(str))

    summary = {
        "data_shapes": {
            "train": list(train.shape),
            "vessel_catalog": list(catalog.shape),
            "sample_submission": list(submission.shape),
        },
        "integrity": {
            "train_sample_id_unique": bool(train["sample_id"].is_unique),
            "catalog_vessel_id_unique": bool(catalog["vessel_id"].is_unique),
            "submission_sample_id_unique": bool(submission["sample_id"].is_unique),
            "all_train_targets_exist_in_catalog": bool(train_targets.issubset(catalog_targets)),
            "missing_train_targets_in_catalog": int(len(train_targets - catalog_targets)),
        },
        "target_structure": {
            "unique_train_targets": int(train["target_vessel_id"].nunique()),
            "catalog_vessels": int(catalog["vessel_id"].nunique()),
            "records_per_target_min": int(target_counts.min()),
            "records_per_target_median": float(target_counts.median()),
            "records_per_target_max": int(target_counts.max()),
            "single_record_targets": int((target_counts == 1).sum()),
            "repeated_targets": int((target_counts >= 2).sum()),
        },
        "language": {
            "hangul_rows": int(hangul_mask.sum()),
            "hangul_ratio": float(hangul_mask.mean()),
            "non_hangul_rows": int((~hangul_mask).sum()),
            "hangul_targets": int(train.loc[hangul_mask, "target_vessel_id"].nunique()),
            "non_hangul_targets": int(train.loc[~hangul_mask, "target_vessel_id"].nunique()),
        },
        "mapping_purity": {
            "call_sign": mapping_profile(train, "call_sign"),
            "vessel_name": mapping_profile(train, "vessel_name"),
        },
        "true_target_catalog_agreement": {
            "normalized_vessel_name_exact_ratio": float((input_name_norm == catalog_name_norm).mean()),
            "gross_tonnage_exact_ratio": float(gt_exact.mean()),
            "gross_tonnage_both_present_ratio": float(both_gt.mean()),
            "normalized_flag_exact_ratio": float((input_flag_norm == catalog_flag_norm).mean()),
            "normalized_vessel_type_exact_ratio": float((input_type_norm == catalog_type_norm).mean()),
        },
    }

    with (result_dir / "eda_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    save_plots(train, profile, result_dir)

    print("=" * 60)
    print("EDA 완료")
    print(f"train: {train.shape}")
    print(f"catalog: {catalog.shape}")
    print(f"학습 정답 선박 수: {summary['target_structure']['unique_train_targets']:,}")
    print(f"반복 관측 선박 수: {summary['target_structure']['repeated_targets']:,}")
    print(f"한글 기록 비율: {summary['language']['hangul_ratio']:.2%}")
    print(f"catalog 정규화 이름 일치율: {summary['true_target_catalog_agreement']['normalized_vessel_name_exact_ratio']:.2%}")
    print(f"결과 저장 위치: {result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
