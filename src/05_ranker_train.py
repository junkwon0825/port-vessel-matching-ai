from __future__ import annotations

"""LightGBM LambdaRank로 후보 선박의 최종 순위를 학습하고 검증한다.

입력:
    - 02_validation.py의 unseen split
    - 04_pair_features.py의 train_pair_features.joblib
    - train.csv, vessel_catalog.csv

핵심 검증 원칙:
    - 랭커는 unseen_split == fit에서만 학습한다.
    - unseen validation의 target_vessel_id는 평가할 때만 사용한다.
    - 검증 후보에는 정답을 강제로 추가하지 않는다.
    - 행 순서를 바꿔도 예측이 같아야 한다.

두 모델을 비교한다.
    1. unweighted: 모든 입력 행의 학습 가중치가 동일
    2. official_weighted: 공식 점수 0.7(한글) + 0.3(비한글)에 맞춘 그룹 가중치

미관측 검증의 공식 가중 정확도가 더 높은 모델을 자동으로 선택해 저장한다.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


TOP_K = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="선박 후보 LightGBM 랭커 학습 및 미관측 선박 검증")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--assignment-path",
        type=Path,
        default=project_dir / "results" / "validation" / "validation_assignments.csv",
    )
    parser.add_argument(
        "--pair-feature-path",
        type=Path,
        default=project_dir / "results" / "pair_features" / "train_pair_features.joblib",
    )
    parser.add_argument(
        "--retrieval-script",
        type=Path,
        default=project_dir / "src" / "03_candidate_retrieval.py",
    )
    parser.add_argument(
        "--feature-script",
        type=Path,
        default=project_dir / "src" / "04_pair_features.py",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=project_dir / "results" / "ranker",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # 03·04번에서 사용한 검색 설정과 일치해야 한다.
    parser.add_argument("--name-candidates", type=int, default=120)
    parser.add_argument("--metadata-pairs", type=int, default=50)
    parser.add_argument("--per-pair-tonnage", type=int, default=15)
    parser.add_argument("--global-tonnage", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=600)
    return parser.parse_args()


def load_module(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Python 파일을 찾을 수 없습니다: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split(
    data_dir: Path,
    assignment_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    for path in [train_path, catalog_path, assignment_path]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    assignments = pd.read_csv(assignment_path, encoding="utf-8-sig")
    if not {"sample_id", "unseen_split"}.issubset(assignments.columns):
        raise ValueError("validation_assignments.csv에 sample_id와 unseen_split이 필요합니다.")

    marked = train.merge(
        assignments[["sample_id", "unseen_split"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if marked["unseen_split"].isna().any():
        raise ValueError("일부 train 행의 unseen_split이 없습니다. 02_validation.py를 다시 실행하세요.")
    fit_train = marked[marked["unseen_split"] == "fit"].drop(columns="unseen_split").reset_index(drop=True)
    valid = marked[marked["unseen_split"] == "valid"].drop(columns="unseen_split").reset_index(drop=True)

    overlap = set(fit_train["target_vessel_id"].astype(str)) & set(valid["target_vessel_id"].astype(str))
    if overlap:
        raise AssertionError(f"미관측 검증에 target_vessel_id 누수가 있습니다: {len(overlap)}개")
    return fit_train, valid, catalog


def validate_pair_artifact(
    artifact: dict[str, Any],
    fit_train: pd.DataFrame,
    expected_features: list[str],
) -> None:
    required = {"X", "y", "group_sizes", "sample_ids", "feature_names", "metadata"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"pair feature 파일에 필요한 키가 없습니다: {sorted(missing)}")

    X = np.asarray(artifact["X"])
    y = np.asarray(artifact["y"])
    groups = np.asarray(artifact["group_sizes"])
    sample_ids = np.asarray(artifact["sample_ids"]).astype(str)
    if list(artifact["feature_names"]) != expected_features:
        raise ValueError("04번 파일과 저장된 pair feature의 특징 순서가 다릅니다. 04번을 다시 실행하세요.")
    if X.ndim != 2 or X.shape[1] != len(expected_features):
        raise ValueError("특징 행렬의 차원이 올바르지 않습니다.")
    if len(X) != len(y) or len(X) != int(groups.sum()):
        raise ValueError("특징 행, 라벨, 그룹 크기가 일치하지 않습니다.")
    if len(groups) != len(sample_ids):
        raise ValueError("그룹 개수와 sample_id 개수가 다릅니다.")
    if int(y.sum()) != len(groups):
        raise ValueError("각 랭킹 그룹에는 정답 라벨이 정확히 하나 있어야 합니다.")
    if not np.isfinite(X).all():
        raise ValueError("특징 행렬에 NaN 또는 무한대가 있습니다.")

    fit_ids = set(fit_train["sample_id"].astype(str))
    if not set(sample_ids).issubset(fit_ids):
        raise ValueError("pair feature에 unseen validation 또는 알 수 없는 sample_id가 포함됐습니다.")


def make_official_sample_weights(
    artifact: dict[str, Any],
    fit_train: pd.DataFrame,
    is_hangul_name: Any,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """공식 0.7/0.3 점수에 맞춰 각 랭킹 그룹의 총 영향력을 조절한다."""
    language_by_sample = fit_train.set_index(fit_train["sample_id"].astype(str))["vessel_name"].map(is_hangul_name)
    group_ids = np.asarray(artifact["sample_ids"]).astype(str)
    group_hangul = language_by_sample.reindex(group_ids)
    if group_hangul.isna().any():
        raise ValueError("가중치 계산 중 일부 sample_id를 fit_train에서 찾지 못했습니다.")
    group_hangul_array = group_hangul.to_numpy(dtype=bool)
    hangul_count = int(group_hangul_array.sum())
    non_hangul_count = int((~group_hangul_array).sum())
    if hangul_count == 0 or non_hangul_count == 0:
        raise ValueError("한글과 비한글 그룹이 모두 있어야 공식 가중치를 계산할 수 있습니다.")

    group_weights = np.where(
        group_hangul_array,
        0.7 / hangul_count,
        0.3 / non_hangul_count,
    )
    # 평균 그룹 가중치를 1로 맞춰 LightGBM의 정규화 강도 변화를 줄인다.
    group_weights *= len(group_weights) / group_weights.sum()
    sample_weights = np.repeat(group_weights, np.asarray(artifact["group_sizes"], dtype=int)).astype(np.float32)
    return sample_weights, {
        "hangul_groups": hangul_count,
        "non_hangul_groups": non_hangul_count,
        "hangul_group_weight": float(group_weights[group_hangul_array][0]),
        "non_hangul_group_weight": float(group_weights[~group_hangul_array][0]),
    }


def model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "max_depth": -1,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0,
        "lambdarank_truncation_level": 20,
        "importance_type": "gain",
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def train_models(
    artifact: dict[str, Any],
    fit_train: pd.DataFrame,
    is_hangul_name: Any,
    params: dict[str, Any],
) -> tuple[dict[str, lgb.LGBMRanker], dict[str, float], dict[str, float | int]]:
    X = np.asarray(artifact["X"], dtype=np.float32)
    y = np.asarray(artifact["y"], dtype=np.int8)
    groups = np.asarray(artifact["group_sizes"], dtype=np.int32)
    official_weights, weight_summary = make_official_sample_weights(artifact, fit_train, is_hangul_name)

    models: dict[str, lgb.LGBMRanker] = {}
    times: dict[str, float] = {}
    for model_name, weights in {
        "unweighted": None,
        "official_weighted": official_weights,
    }.items():
        started = time.perf_counter()
        model = lgb.LGBMRanker(**params)
        model.fit(
            X,
            y,
            group=groups,
            sample_weight=weights,
            feature_name=list(artifact["feature_names"]),
            callbacks=[lgb.log_evaluation(period=0)],
        )
        models[model_name] = model
        times[model_name] = time.perf_counter() - started
    return models, times, weight_summary


def evaluate_models(
    valid: pd.DataFrame,
    candidate_lists: list[list[int]],
    retriever: Any,
    feature_builder: Any,
    models: dict[str, lgb.LGBMRanker],
    is_hangul_name: Any,
    prediction_batch_size: int,
) -> dict[str, pd.DataFrame]:
    """검증 후보를 강제 수정하지 않고 모델별 순위와 Top-k 적중을 계산한다."""
    if len(valid) != len(candidate_lists):
        raise ValueError("검증 행 수와 후보 목록 수가 다릅니다.")
    records_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in models}

    for start in range(0, len(valid), prediction_batch_size):
        end = min(start + prediction_batch_size, len(valid))
        matrices: list[np.ndarray] = []
        sizes: list[int] = []
        for row_number in range(start, end):
            query = valid.iloc[row_number]
            candidates = candidate_lists[row_number]
            if not candidates:
                raise AssertionError(f"후보가 하나도 없는 행이 있습니다: {query['sample_id']}")
            matrix = feature_builder.transform_group(query, candidates)
            matrices.append(matrix)
            sizes.append(len(candidates))
        batch_X = np.vstack(matrices).astype(np.float32, copy=False)
        # sklearn wrapper를 거치지 않고 Booster가 학습된 특징 순서 그대로 예측한다.
        # ndarray 입력 시 발생하는 불필요한 "feature names" 경고도 방지된다.
        scores_by_model = {name: model.booster_.predict(batch_X) for name, model in models.items()}

        offset = 0
        for local_number, row_number in enumerate(range(start, end)):
            query = valid.iloc[row_number]
            candidates = candidate_lists[row_number]
            size = sizes[local_number]
            true_id = str(query["target_vessel_id"])
            true_index = retriever.id_to_index[true_id]
            true_candidate_rank = candidates.index(true_index) + 1 if true_index in candidates else None

            for model_name, batch_scores in scores_by_model.items():
                scores = np.asarray(batch_scores[offset : offset + size], dtype=float)
                ranking = np.argsort(-scores, kind="mergesort")
                predicted_index = candidates[int(ranking[0])]
                ranked_ids = [retriever.ids[candidates[int(index)]] for index in ranking[: max(TOP_K)]]
                record: dict[str, Any] = {
                    "sample_id": str(query["sample_id"]),
                    "target_vessel_id": true_id,
                    "prediction": retriever.ids[predicted_index],
                    "correct": retriever.ids[predicted_index] == true_id,
                    "is_hangul": bool(is_hangul_name(query["vessel_name"])),
                    "candidate_count": size,
                    "true_candidate_rank": true_candidate_rank,
                    "prediction_score": float(scores[ranking[0]]),
                    "score_margin": float(scores[ranking[0]] - scores[ranking[1]]) if size > 1 else float("nan"),
                }
                for cutoff in TOP_K:
                    record[f"hit_at_{cutoff}"] = true_id in ranked_ids[:cutoff]
                records_by_model[model_name].append(record)
            offset += size

    return {name: pd.DataFrame(records) for name, records in records_by_model.items()}


def retrieval_first_predictions(
    valid: pd.DataFrame,
    candidate_lists: list[list[int]],
    retriever: Any,
    is_hangul_name: Any,
) -> pd.DataFrame:
    records = []
    for (_, query), candidates in zip(valid.iterrows(), candidate_lists):
        prediction = retriever.ids[candidates[0]] if candidates else ""
        true_id = str(query["target_vessel_id"])
        true_index = retriever.id_to_index[true_id]
        ranked_ids = [retriever.ids[index] for index in candidates[: max(TOP_K)]]
        record: dict[str, Any] = {
            "sample_id": str(query["sample_id"]),
            "target_vessel_id": true_id,
            "prediction": prediction,
            "correct": prediction == true_id,
            "is_hangul": bool(is_hangul_name(query["vessel_name"])),
            "candidate_count": len(candidates),
            "true_candidate_rank": candidates.index(true_index) + 1 if true_index in candidates else None,
            "prediction_score": float("nan"),
            "score_margin": float("nan"),
        }
        for cutoff in TOP_K:
            record[f"hit_at_{cutoff}"] = true_id in ranked_ids[:cutoff]
        records.append(
            record
        )
    return pd.DataFrame(records)


def build_language_hybrid(
    sources: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """검증에서 한글·비한글 각각 가장 정확한 예측 경로를 결합한다.

    공식 평가는 두 언어군의 정확도를 0.7/0.3으로 따로 합산한다. 따라서
    한글과 비한글에 같은 모델을 강제하기보다 각 언어군의 최적 경로를
    선택하는 것이 평가식과 실제 오류 특성에 더 잘 맞는다.
    """
    reference_ids: list[str] | None = None
    for frame in sources.values():
        ids = frame["sample_id"].astype(str).tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("언어별 결합 대상 예측 파일의 sample_id 순서가 다릅니다.")

    def language_accuracy(frame: pd.DataFrame, hangul_value: bool) -> float:
        mask = frame["is_hangul"].astype(bool) == hangul_value
        return float(frame.loc[mask, "correct"].astype(bool).mean())

    routes = {
        "hangul": max(sources, key=lambda name: (language_accuracy(sources[name], True), name)),
        "non_hangul": max(sources, key=lambda name: (language_accuracy(sources[name], False), name)),
    }
    hangul_frame = sources[routes["hangul"]]
    non_hangul_frame = sources[routes["non_hangul"]]
    hybrid = hangul_frame.copy()
    non_hangul_mask = ~hybrid["is_hangul"].astype(bool)
    hybrid.loc[non_hangul_mask, :] = non_hangul_frame.loc[non_hangul_mask, :].to_numpy()
    return hybrid, routes


def accuracy_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    hangul = predictions["is_hangul"].to_numpy(dtype=bool)
    correct = predictions["correct"].to_numpy(dtype=bool)

    def mean(mask: np.ndarray, column: str) -> float | None:
        return float(predictions.loc[mask, column].astype(bool).mean()) if mask.any() else None

    hangul_accuracy = float(correct[hangul].mean())
    non_hangul_accuracy = float(correct[~hangul].mean())
    output: dict[str, Any] = {
        "rows": int(len(predictions)),
        "overall_accuracy": float(correct.mean()),
        "hangul_rows": int(hangul.sum()),
        "non_hangul_rows": int((~hangul).sum()),
        "hangul_accuracy": hangul_accuracy,
        "non_hangul_accuracy": non_hangul_accuracy,
        "official_weighted_accuracy": 0.7 * hangul_accuracy + 0.3 * non_hangul_accuracy,
        "candidate_recall": float(predictions["true_candidate_rank"].notna().mean()),
    }
    for cutoff in TOP_K:
        column = f"hit_at_{cutoff}"
        if column not in predictions:
            continue
        hangul_hit = mean(hangul, column)
        non_hangul_hit = mean(~hangul, column)
        output[f"top_{cutoff}"] = {
            "overall": float(predictions[column].astype(bool).mean()),
            "hangul": hangul_hit,
            "non_hangul": non_hangul_hit,
            "official_weighted": 0.7 * hangul_hit + 0.3 * non_hangul_hit,
        }
    return output


def choose_strategy(metrics: dict[str, dict[str, Any]]) -> str:
    return max(
        metrics,
        key=lambda name: (
            metrics[name]["official_weighted_accuracy"],
            metrics[name]["overall_accuracy"],
            name,
        ),
    )


def check_order_invariance(
    valid: pd.DataFrame,
    retriever: Any,
    feature_builder: Any,
    model: lgb.LGBMRanker,
    retrieval_batch_size: int,
    seed: int,
) -> bool:
    sample = valid.head(min(100, len(valid))).copy()

    def predict(frame: pd.DataFrame) -> dict[str, str]:
        candidate_lists = retriever.retrieve_many(frame, retrieval_batch_size)
        output: dict[str, str] = {}
        for (_, query), candidates in zip(frame.iterrows(), candidate_lists):
            X = feature_builder.transform_group(query, candidates)
            scores = model.booster_.predict(X)
            best = candidates[int(np.argmax(scores))]
            output[str(query["sample_id"])] = retriever.ids[best]
        return output

    original = predict(sample)
    shuffled = predict(sample.sample(frac=1.0, random_state=seed).reset_index(drop=True))
    return original == shuffled


def save_feature_importance(
    model: lgb.LGBMRanker,
    feature_names: list[str],
    path: Path,
) -> None:
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    total_gain = float(gain.sum())
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": gain,
            "gain_ratio": gain / total_gain if total_gain > 0 else np.zeros_like(gain, dtype=float),
            "split": split,
        }
    ).sort_values(["gain", "split"], ascending=False)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    if args.n_estimators < 1 or args.num_leaves < 2 or args.prediction_batch_size < 1:
        raise ValueError("모델 및 배치 설정값을 확인하세요.")

    data_dir = args.data_dir.resolve()
    assignment_path = args.assignment_path.resolve()
    pair_feature_path = args.pair_feature_path.resolve()
    retrieval_script = args.retrieval_script.resolve()
    feature_script = args.feature_script.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    for path in [pair_feature_path, retrieval_script, feature_script]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    retrieval = load_module(retrieval_script, "candidate_retrieval")
    pair_features = load_module(feature_script, "pair_features")
    fit_train, valid, catalog = load_split(data_dir, assignment_path)
    artifact = joblib.load(pair_feature_path)
    validate_pair_artifact(artifact, fit_train, pair_features.FEATURE_NAMES)

    params = model_parameters(args)
    training_started = time.perf_counter()
    models, model_train_times, weight_summary = train_models(
        artifact,
        fit_train,
        retrieval.is_hangul_name,
        params,
    )
    total_training_seconds = time.perf_counter() - training_started

    index_started = time.perf_counter()
    retriever = retrieval.CandidateRetriever(
        catalog=catalog,
        fit_train=fit_train,
        name_candidates=args.name_candidates,
        metadata_pairs=args.metadata_pairs,
        per_pair_tonnage=args.per_pair_tonnage,
        global_tonnage=args.global_tonnage,
        max_candidates=args.max_candidates,
    )
    feature_builder = pair_features.PairFeatureBuilder(catalog, fit_train, retriever, retrieval)
    index_seconds = time.perf_counter() - index_started

    retrieval_started = time.perf_counter()
    candidate_lists = retriever.retrieve_many(valid, args.retrieval_batch_size)
    retrieval_seconds = time.perf_counter() - retrieval_started

    prediction_started = time.perf_counter()
    predictions = evaluate_models(
        valid=valid,
        candidate_lists=candidate_lists,
        retriever=retriever,
        feature_builder=feature_builder,
        models=models,
        is_hangul_name=retrieval.is_hangul_name,
        prediction_batch_size=args.prediction_batch_size,
    )
    prediction_seconds = time.perf_counter() - prediction_started

    retrieval_baseline = retrieval_first_predictions(
        valid,
        candidate_lists,
        retriever,
        retrieval.is_hangul_name,
    )
    baseline_metrics = accuracy_metrics(retrieval_baseline)
    prediction_sources = {"retrieval_first": retrieval_baseline, **predictions}
    language_hybrid, language_routes = build_language_hybrid(prediction_sources)
    predictions["language_hybrid"] = language_hybrid
    metrics = {name: accuracy_metrics(frame) for name, frame in predictions.items()}
    selected_strategy = choose_strategy(metrics)

    # 하이브리드가 선택되면 실제로 사용하는 랭커 중 한글 경로를 우선 저장한다.
    if selected_strategy == "language_hybrid":
        model_routes = [
            language_routes[language]
            for language in ["hangul", "non_hangul"]
            if language_routes[language] in models
        ]
        if not model_routes:
            raise AssertionError("선택된 하이브리드 전략에 저장할 랭커 모델이 없습니다.")
        selected_model_name = model_routes[0]
    else:
        selected_model_name = selected_strategy
    selected_model = models[selected_model_name]

    order_invariant = check_order_invariance(
        valid,
        retriever,
        feature_builder,
        selected_model,
        args.retrieval_batch_size,
        args.seed,
    )

    # 비교 실험 모델과 최종 선택 모델을 모두 남긴다.
    for model_name, model in models.items():
        model.booster_.save_model(result_dir / f"ranker_{model_name}.txt")
        predictions[model_name].to_csv(
            result_dir / f"unseen_predictions_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    language_hybrid.to_csv(
        result_dir / "unseen_predictions_language_hybrid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_model.booster_.save_model(result_dir / "ranker_model.txt")
    predictions[selected_strategy].to_csv(
        result_dir / "unseen_predictions_selected.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retrieval_baseline.to_csv(
        result_dir / "unseen_predictions_retrieval_first.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_feature_importance(
        selected_model,
        list(artifact["feature_names"]),
        result_dir / "feature_importance.csv",
    )

    model_metadata = {
        "selected_strategy": selected_strategy,
        "selected_model_file_variant": selected_model_name,
        "language_routes": language_routes,
        "model_type": "LightGBM LGBMRanker / lambdarank",
        "feature_names": list(artifact["feature_names"]),
        "model_parameters": params,
        "retrieval_parameters": {
            "name_candidates": args.name_candidates,
            "metadata_pairs": args.metadata_pairs,
            "per_pair_tonnage": args.per_pair_tonnage,
            "global_tonnage": args.global_tonnage,
            "max_candidates": args.max_candidates,
        },
        "language_weight_summary": weight_summary,
    }
    with (result_dir / "ranker_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(model_metadata, file, ensure_ascii=False, indent=2)

    summary = {
        "validation": "target_vessel_id-disjoint unseen-vessel split",
        "fit_rows": int(len(fit_train)),
        "valid_rows": int(len(valid)),
        "target_overlap": 0,
        "training_pair_rows": int(len(artifact["y"])),
        "feature_count": int(len(artifact["feature_names"])),
        "selected_strategy": selected_strategy,
        "selected_model_file_variant": selected_model_name,
        "language_routes": language_routes,
        "retrieval_first_baseline": baseline_metrics,
        "ranker_metrics": metrics,
        "order_invariant": order_invariant,
        "timing_seconds": {
            "model_training_total": total_training_seconds,
            "model_training_by_variant": model_train_times,
            "retrieval_index_and_feature_metadata": index_seconds,
            "valid_candidate_retrieval": retrieval_seconds,
            "valid_feature_and_prediction": prediction_seconds,
        },
    }
    with (result_dir / "ranker_validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    selected_metrics = metrics[selected_strategy]
    print("=" * 76)
    print("LightGBM 후보 랭커 학습 및 미관측 선박 검증 완료")
    print(f"학습 쌍: {len(artifact['y']):,} / 검증 행: {len(valid):,}")
    print(f"선택 전략: {selected_strategy}")
    print(f"언어별 경로: 한글={language_routes['hangul']} / 비한글={language_routes['non_hangul']}")
    print(
        "Top-1 정확도: "
        f"전체 {selected_metrics['overall_accuracy']:.4f} / "
        f"한글 {selected_metrics['hangul_accuracy']:.4f} / "
        f"비한글 {selected_metrics['non_hangul_accuracy']:.4f}"
    )
    print(f"공식 가중 정확도: {selected_metrics['official_weighted_accuracy']:.4f}")
    print(f"Top-3 가중 적중률: {selected_metrics['top_3']['official_weighted']:.4f}")
    print(f"Top-5 가중 적중률: {selected_metrics['top_5']['official_weighted']:.4f}")
    print(f"후보 재현율 상한: {selected_metrics['candidate_recall']:.4f}")
    print(
        "검색 1순위 baseline 가중 정확도: "
        f"{baseline_metrics['official_weighted_accuracy']:.4f}"
    )
    print(f"행 순서 독립성: {order_invariant}")
    print(f"모델 학습: {total_training_seconds:.2f}초 / 검증 예측: {prediction_seconds:.2f}초")
    print(f"최종 모델: {result_dir / 'ranker_model.txt'}")
    print(f"결과 저장 위치: {result_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
