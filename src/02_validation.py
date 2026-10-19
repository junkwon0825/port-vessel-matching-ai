from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "vessel_name",
    "call_sign",
    "arrival_year",
    "target_vessel_id",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="항만 선박 매칭 대회 검증 세트 생성")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--result-dir", type=Path, default=project_dir / "results" / "validation")
    parser.add_argument("--valid-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper().strip()
    return re.sub(r"[^0-9A-Z가-힣]", "", text)


def is_hangul_name(value: object) -> bool:
    return bool(re.search(r"[가-힣]", str(value)))


def validate_input(train: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(train.columns)
    if missing:
        raise ValueError(f"train.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")
    if train["sample_id"].isna().any() or not train["sample_id"].is_unique:
        raise ValueError("sample_id는 결측치가 없어야 하며 행마다 고유해야 합니다.")
    if train["target_vessel_id"].isna().any():
        raise ValueError("target_vessel_id에 결측치가 있습니다.")


def build_repeat_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반복 관측된 선박마다 가장 최근 기록 1개를 검증으로 분리한다."""
    ordered = train.sort_values(
        ["target_vessel_id", "arrival_year", "sample_id"],
        kind="mergesort",
    )
    target_size = ordered.groupby("target_vessel_id")["sample_id"].transform("size")
    repeat_rows = ordered[target_size >= 2]
    valid_index = repeat_rows.groupby("target_vessel_id", sort=False).tail(1).index
    valid = train.loc[valid_index].sort_values("sample_id").reset_index(drop=True)
    fit = train.drop(index=valid_index).sort_values("sample_id").reset_index(drop=True)

    valid_targets = set(valid["target_vessel_id"])
    fit_targets = set(fit["target_vessel_id"])
    if not valid_targets.issubset(fit_targets):
        raise AssertionError("반복 선박 검증의 모든 정답 선박은 학습 부분에도 존재해야 합니다.")
    return fit, valid


def choose_stratified_targets(train: pd.DataFrame, valid_fraction: float, seed: int) -> set[str]:
    """한글·비한글 선박 비율을 유지하며 검증용 target_vessel_id를 선택한다."""
    target_language = train.assign(is_hangul=train["vessel_name"].map(is_hangul_name)).groupby("target_vessel_id")["is_hangul"].max()
    rng = np.random.default_rng(seed)
    selected: set[str] = set()

    for language in [False, True]:
        target_ids = target_language[target_language == language].index.astype(str).to_numpy()
        target_ids = rng.permutation(target_ids)
        count = int(round(len(target_ids) * valid_fraction))
        count = min(max(count, 1), len(target_ids) - 1)
        selected.update(target_ids[:count].tolist())
    return selected


def build_unseen_split(train: pd.DataFrame, valid_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """target_vessel_id가 학습과 검증에 겹치지 않는 어려운 검증 세트를 만든다."""
    valid_targets = choose_stratified_targets(train, valid_fraction, seed)
    valid_mask = train["target_vessel_id"].astype(str).isin(valid_targets)
    fit = train.loc[~valid_mask].sort_values("sample_id").reset_index(drop=True)
    valid = train.loc[valid_mask].sort_values("sample_id").reset_index(drop=True)

    overlap = set(fit["target_vessel_id"]) & set(valid["target_vessel_id"])
    if overlap:
        raise AssertionError(f"미관측 선박 검증에 target 누수가 있습니다: {len(overlap)}개")
    return fit, valid


def make_unique_map(fit: pd.DataFrame, column: str) -> dict[str, str]:
    keys = fit[column].map(normalize_text)
    frame = pd.DataFrame({"key": keys, "target": fit["target_vessel_id"].astype(str)})
    frame = frame[frame["key"] != ""]
    grouped = frame.groupby("key")["target"].agg(["first", "nunique"])
    return grouped.loc[grouped["nunique"] == 1, "first"].to_dict()


def predict_direct(fit: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """제공 train의 고유 호출부호를 우선 사용하고, 없으면 고유 선박명을 사용한다."""
    call_sign_map = make_unique_map(fit, "call_sign")
    vessel_name_map = make_unique_map(fit, "vessel_name")
    call_predictions = queries["call_sign"].map(normalize_text).map(call_sign_map)
    name_predictions = queries["vessel_name"].map(normalize_text).map(vessel_name_map)
    predictions = call_predictions.copy()
    missing_call_sign = predictions.isna()
    predictions.loc[missing_call_sign] = name_predictions.loc[missing_call_sign]
    source = np.where(call_predictions.notna(), "call_sign", np.where(name_predictions.notna(), "vessel_name", "unmatched"))
    return pd.DataFrame(
        {
            "sample_id": queries["sample_id"].to_numpy(),
            "prediction": predictions.to_numpy(),
            "source": source,
        }
    )


def score_predictions(truth: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, object]:
    merged = truth[["sample_id", "vessel_name", "target_vessel_id"]].merge(predicted, on="sample_id", how="left", validate="one_to_one")
    covered = merged["prediction"].notna()
    correct = covered & (merged["prediction"].astype(str) == merged["target_vessel_id"].astype(str))
    hangul = merged["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)

    def accuracy(mask: np.ndarray) -> float | None:
        return float(correct.to_numpy()[mask].mean()) if mask.any() else None

    hangul_accuracy = accuracy(hangul)
    non_hangul_accuracy = accuracy(~hangul)
    weighted_accuracy = None
    if hangul_accuracy is not None and non_hangul_accuracy is not None:
        weighted_accuracy = 0.7 * hangul_accuracy + 0.3 * non_hangul_accuracy

    return {
        "rows": int(len(merged)),
        "covered_rows": int(covered.sum()),
        "coverage": float(covered.mean()),
        "correct_rows": int(correct.sum()),
        "overall_accuracy": float(correct.mean()),
        "accuracy_when_covered": float(correct[covered].mean()) if covered.any() else None,
        "hangul_rows": int(hangul.sum()),
        "non_hangul_rows": int((~hangul).sum()),
        "hangul_accuracy": hangul_accuracy,
        "non_hangul_accuracy": non_hangul_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "prediction_sources": predicted["source"].value_counts().to_dict(),
    }


def check_order_invariance(fit: pd.DataFrame, valid: pd.DataFrame, seed: int) -> bool:
    original = predict_direct(fit, valid).set_index("sample_id")["prediction"].sort_index()
    shuffled_valid = valid.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled = predict_direct(fit, shuffled_valid).set_index("sample_id")["prediction"].sort_index()
    return bool(original.fillna("__NA__").equals(shuffled.fillna("__NA__")))


def split_summary(fit: pd.DataFrame, valid: pd.DataFrame) -> dict[str, int | float]:
    return {
        "fit_rows": int(len(fit)),
        "valid_rows": int(len(valid)),
        "fit_targets": int(fit["target_vessel_id"].nunique()),
        "valid_targets": int(valid["target_vessel_id"].nunique()),
        "target_overlap": int(len(set(fit["target_vessel_id"]) & set(valid["target_vessel_id"]))),
        "valid_hangul_ratio": float(valid["vessel_name"].map(is_hangul_name).mean()),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.valid_fraction < 1:
        raise ValueError("--valid-fraction은 0과 1 사이여야 합니다.")

    train_path = args.data_dir.resolve() / "train.csv"
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {train_path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    validate_input(train)

    repeat_fit, repeat_valid = build_repeat_split(train)
    unseen_fit, unseen_valid = build_unseen_split(train, args.valid_fraction, args.seed)

    assignments = train[["sample_id"]].copy()
    assignments["repeat_split"] = np.where(assignments["sample_id"].isin(set(repeat_valid["sample_id"])), "valid", "fit")
    assignments["unseen_split"] = np.where(assignments["sample_id"].isin(set(unseen_valid["sample_id"])), "valid", "fit")
    assignments.to_csv(result_dir / "validation_assignments.csv", index=False, encoding="utf-8-sig")

    repeat_predictions = predict_direct(repeat_fit, repeat_valid)
    unseen_predictions = predict_direct(unseen_fit, unseen_valid)
    repeat_predictions.to_csv(result_dir / "repeat_direct_predictions.csv", index=False, encoding="utf-8-sig")
    unseen_predictions.to_csv(result_dir / "unseen_direct_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "settings": {
            "seed": args.seed,
            "unseen_valid_fraction": args.valid_fraction,
            "official_metric": "0.7 * hangul_accuracy + 0.3 * non_hangul_accuracy",
        },
        "repeat_vessel_validation": {
            "purpose": "이미 학습에서 관측된 선박의 새로운 입출항 기록 식별 능력 측정",
            "split": split_summary(repeat_fit, repeat_valid),
            "direct_lookup": score_predictions(repeat_valid, repeat_predictions),
            "order_invariant": check_order_invariance(repeat_fit, repeat_valid, args.seed),
        },
        "unseen_vessel_validation": {
            "purpose": "학습에 없던 선박을 vessel_catalog에서 찾아야 하는 일반화 능력 측정",
            "split": split_summary(unseen_fit, unseen_valid),
            "direct_lookup": score_predictions(unseen_valid, unseen_predictions),
            "order_invariant": check_order_invariance(unseen_fit, unseen_valid, args.seed),
        },
    }

    with (result_dir / "validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    repeat_score = summary["repeat_vessel_validation"]["direct_lookup"]["weighted_accuracy"]
    unseen_score = summary["unseen_vessel_validation"]["direct_lookup"]["weighted_accuracy"]
    print("=" * 68)
    print("검증 세트 생성 완료")
    print(f"반복 선박 검증: fit={len(repeat_fit):,}, valid={len(repeat_valid):,}, target overlap={summary['repeat_vessel_validation']['split']['target_overlap']:,}")
    print(f"반복 선박 직접조회 가중 정확도: {repeat_score:.4f}")
    print(f"미관측 선박 검증: fit={len(unseen_fit):,}, valid={len(unseen_valid):,}, target overlap={summary['unseen_vessel_validation']['split']['target_overlap']:,}")
    print(f"미관측 선박 직접조회 가중 정확도: {unseen_score:.4f}")
    print(f"행 순서 독립성: repeat={summary['repeat_vessel_validation']['order_invariant']}, unseen={summary['unseen_vessel_validation']['order_invariant']}")
    print(f"결과 저장 위치: {result_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
