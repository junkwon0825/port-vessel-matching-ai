from __future__ import annotations

"""전체 학습데이터로 최종 모델을 만들고 submission.csv를 생성한다.

개발 PC:
    python src/06_final_pipeline.py --mode train

평가 서버 또는 test.csv가 있는 환경:
    python src/06_final_pipeline.py --mode predict

한 번에 실행:
    python src/06_final_pipeline.py --mode all

최종 예측 경로:
    1. train에서 유일한 call_sign 직접 매칭
    2. train에서 유일한 vessel_name 직접 매칭
    3. 한글 선박명: 후보 검색 후 LightGBM 랭커
    4. 비한글 선박명: 후보 검색 1순위

어떤 경로도 다른 test 행이나 test 전체 분포를 사용하지 않는다.
따라서 test 행 순서를 바꿔도 각 sample_id의 예측은 동일하다.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


TEST_COLUMNS = {
    "sample_id",
    "vessel_name",
    "call_sign",
    "gross_tonnage",
    "international_gross_tonnage",
    "flag_name",
    "vessel_type",
    "arrival_year",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="선박 매칭 최종 학습 및 submission 생성")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--sample-submission-path", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=project_dir / "model")
    parser.add_argument("--output-path", type=Path, default=project_dir / "output" / "submission.csv")
    parser.add_argument("--retrieval-script", type=Path, default=project_dir / "src" / "03_candidate_retrieval.py")
    parser.add_argument("--feature-script", type=Path, default=project_dir / "src" / "04_pair_features.py")
    parser.add_argument("--ranker-script", type=Path, default=project_dir / "src" / "05_ranker_train.py")

    parser.add_argument("--hard-negatives", type=int, default=40)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--retrieval-batch-size", type=int, default=512)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

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


def check_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(missing)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retrieval_parameters(args: argparse.Namespace) -> dict[str, int]:
    return {
        "name_candidates": args.name_candidates,
        "metadata_pairs": args.metadata_pairs,
        "per_pair_tonnage": args.per_pair_tonnage,
        "global_tonnage": args.global_tonnage,
        "max_candidates": args.max_candidates,
    }


def make_retriever(
    catalog: pd.DataFrame,
    train: pd.DataFrame,
    retrieval: ModuleType,
    settings: dict[str, int],
) -> Any:
    return retrieval.CandidateRetriever(catalog=catalog, fit_train=train, **settings)


def save_feature_importance(model: lgb.LGBMRanker, feature_names: list[str], path: Path) -> None:
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


def train_final_model(
    args: argparse.Namespace,
    retrieval: ModuleType,
    pair_features: ModuleType,
    ranker_code: ModuleType,
) -> dict[str, Any]:
    """공개 train 전체를 사용해 제출용 한글 선박 랭커를 학습한다."""
    data_dir = args.data_dir.resolve()
    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    for path in [train_path, catalog_path]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    retrieval.check_columns(train, retrieval.TRAIN_COLUMNS, "train.csv")
    retrieval.check_columns(catalog, retrieval.CATALOG_COLUMNS, "vessel_catalog.csv")
    if not pair_features.TRAIN_COLUMNS.issubset(train.columns):
        raise ValueError("train.csv 컬럼이 04_pair_features.py의 요구사항과 다릅니다.")

    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    settings = retrieval_parameters(args)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    retriever = make_retriever(catalog, train, retrieval, settings)
    feature_builder = pair_features.PairFeatureBuilder(catalog, train, retriever, retrieval)
    timings["retrieval_index_and_metadata"] = time.perf_counter() - started

    started = time.perf_counter()
    candidate_lists = retriever.retrieve_many(train, args.retrieval_batch_size)
    timings["training_candidate_retrieval"] = time.perf_counter() - started

    started = time.perf_counter()
    artifact, _, diagnostics = pair_features.build_training_matrix(
        fit_train=train,
        candidate_lists=candidate_lists,
        retriever=retriever,
        feature_builder=feature_builder,
        hard_negatives=args.hard_negatives,
    )
    timings["pair_feature_generation"] = time.perf_counter() - started
    if list(artifact["feature_names"]) != list(pair_features.FEATURE_NAMES):
        raise AssertionError("최종 학습 특징 순서가 04번 코드와 다릅니다.")

    model_args = argparse.Namespace(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        seed=args.seed,
    )
    params = ranker_code.model_parameters(model_args)
    sample_weights, weight_summary = ranker_code.make_official_sample_weights(
        artifact,
        train,
        retrieval.is_hangul_name,
    )

    started = time.perf_counter()
    model = lgb.LGBMRanker(**params)
    model.fit(
        artifact["X"],
        artifact["y"],
        group=artifact["group_sizes"],
        sample_weight=sample_weights,
        feature_name=list(artifact["feature_names"]),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    timings["ranker_training"] = time.perf_counter() - started

    model_path = model_dir / "ranker_model.txt"
    metadata_path = model_dir / "model_metadata.json"
    model.booster_.save_model(model_path)
    save_feature_importance(model, list(artifact["feature_names"]), model_dir / "feature_importance.csv")

    metadata = {
        "purpose": "공개 train 전체로 학습한 최종 제출용 선박 후보 랭커",
        "training_scope": "all rows of provided train.csv",
        "prediction_strategy": {
            "direct_priority": ["unique_normalized_call_sign", "unique_normalized_vessel_name"],
            "hangul_unmatched": "LightGBM ranker",
            "non_hangul_unmatched": "candidate retrieval first",
        },
        "train_rows": int(len(train)),
        "train_targets": int(train["target_vessel_id"].nunique()),
        "catalog_rows": int(len(catalog)),
        "training_pair_rows": int(len(artifact["y"])),
        "feature_names": list(artifact["feature_names"]),
        "hard_negatives": args.hard_negatives,
        "retrieval_parameters": settings,
        "model_parameters": params,
        "language_weight_summary": weight_summary,
        "training_diagnostics": diagnostics,
        "source_sha256": {
            "train.csv": sha256_file(train_path),
            "vessel_catalog.csv": sha256_file(catalog_path),
        },
        "timing_seconds": timings,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("전체 공개 학습데이터 최종 모델 생성 완료")
    print(f"학습 행/선박: {len(train):,} / {train['target_vessel_id'].nunique():,}")
    print(f"학습 후보 쌍: {len(artifact['y']):,} / 특징: {len(artifact['feature_names'])}개")
    print(f"검색 누락 정답 강제 추가: {diagnostics['forced_positive_groups']:,}개")
    print(f"최종 모델: {model_path}")
    print(f"모델 메타데이터: {metadata_path}")
    print(f"총 학습 파이프라인 시간: {sum(timings.values()):.2f}초")
    print("=" * 76)
    return metadata


def make_unique_target_map(
    train: pd.DataFrame,
    column: str,
    normalize_text: Any,
) -> dict[str, str]:
    keys = train[column].map(normalize_text)
    frame = pd.DataFrame({"key": keys, "target": train["target_vessel_id"].astype(str)})
    frame = frame[frame["key"] != ""]
    grouped = frame.groupby("key")["target"].agg(["first", "nunique"])
    return grouped.loc[grouped["nunique"] == 1, "first"].to_dict()


def direct_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    normalize_text: Any,
) -> tuple[pd.Series, pd.Series]:
    call_sign_map = make_unique_target_map(train, "call_sign", normalize_text)
    vessel_name_map = make_unique_target_map(train, "vessel_name", normalize_text)
    call_predictions = test["call_sign"].map(normalize_text).map(call_sign_map)
    name_predictions = test["vessel_name"].map(normalize_text).map(vessel_name_map)
    predictions = call_predictions.copy()
    use_name = predictions.isna() & name_predictions.notna()
    predictions.loc[use_name] = name_predictions.loc[use_name]
    routes = pd.Series("unmatched", index=test.index, dtype=object)
    routes.loc[call_predictions.notna()] = "direct_call_sign"
    routes.loc[use_name] = "direct_vessel_name"
    return predictions, routes


def predict_unmatched(
    unmatched: pd.DataFrame,
    candidate_lists: list[list[int]],
    retriever: Any,
    feature_builder: Any,
    booster: lgb.Booster,
    is_hangul_name: Any,
    prediction_batch_size: int,
) -> tuple[dict[int, str], dict[int, str]]:
    predictions: dict[int, str] = {}
    routes: dict[int, str] = {}
    unmatched = unmatched.reset_index().rename(columns={"index": "_original_index"})

    for start in range(0, len(unmatched), prediction_batch_size):
        end = min(start + prediction_batch_size, len(unmatched))
        matrices: list[np.ndarray] = []
        matrix_rows: list[int] = []
        matrix_candidates: list[list[int]] = []

        for row_number in range(start, end):
            query = unmatched.iloc[row_number]
            candidates = candidate_lists[row_number]
            if not candidates:
                raise AssertionError(f"후보가 없는 test 행이 있습니다: {query['sample_id']}")
            original_index = int(query["_original_index"])
            if is_hangul_name(query["vessel_name"]):
                matrices.append(feature_builder.transform_group(query, candidates))
                matrix_rows.append(original_index)
                matrix_candidates.append(candidates)
            else:
                predictions[original_index] = retriever.ids[candidates[0]]
                routes[original_index] = "non_hangul_retrieval_first"

        if matrices:
            batch_X = np.vstack(matrices).astype(np.float32, copy=False)
            batch_scores = booster.predict(batch_X)
            offset = 0
            for original_index, candidates in zip(matrix_rows, matrix_candidates):
                size = len(candidates)
                scores = batch_scores[offset : offset + size]
                best_index = candidates[int(np.argmax(scores))]
                predictions[original_index] = retriever.ids[best_index]
                routes[original_index] = "hangul_ranker"
                offset += size
    return predictions, routes


def predict_submission(
    args: argparse.Namespace,
    retrieval: ModuleType,
    pair_features: ModuleType,
) -> pd.DataFrame:
    data_dir = args.data_dir.resolve()
    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    test_path = args.test_path.resolve() if args.test_path else data_dir / "test.csv"
    sample_path = (
        args.sample_submission_path.resolve()
        if args.sample_submission_path
        else data_dir / "sample_submission.csv"
    )
    model_dir = args.model_dir.resolve()
    model_path = model_dir / "ranker_model.txt"
    metadata_path = model_dir / "model_metadata.json"
    for path in [train_path, catalog_path, test_path, sample_path, model_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    check_columns(test, TEST_COLUMNS, "test.csv")
    if "target_vessel_id" in test.columns:
        raise ValueError("predict 모드의 test.csv에는 target_vessel_id가 없어야 합니다.")
    if not {"sample_id", "target_vessel_id"}.issubset(sample.columns):
        raise ValueError("sample_submission.csv 형식이 올바르지 않습니다.")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv의 sample_id는 결측 없이 고유해야 합니다.")
    if sample["sample_id"].isna().any() or not sample["sample_id"].is_unique:
        raise ValueError("sample_submission.csv의 sample_id는 결측 없이 고유해야 합니다.")
    if set(test["sample_id"].astype(str)) != set(sample["sample_id"].astype(str)):
        raise ValueError("test.csv와 sample_submission.csv의 sample_id 집합이 다릅니다.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["feature_names"] != list(pair_features.FEATURE_NAMES):
        raise ValueError("저장 모델과 현재 04번 코드의 특징 순서가 다릅니다.")
    settings = metadata["retrieval_parameters"]
    booster = lgb.Booster(model_file=str(model_path))
    if booster.num_feature() != len(pair_features.FEATURE_NAMES):
        raise ValueError("저장된 LightGBM 모델의 특징 수가 현재 코드와 다릅니다.")

    total_started = time.perf_counter()
    retriever = make_retriever(catalog, train, retrieval, settings)
    feature_builder = pair_features.PairFeatureBuilder(catalog, train, retriever, retrieval)
    predictions, routes = direct_predictions(train, test, retrieval.normalize_text)

    unmatched_mask = predictions.isna()
    unmatched = test.loc[unmatched_mask].copy()
    if len(unmatched):
        candidate_lists = retriever.retrieve_many(unmatched, args.retrieval_batch_size)
        unmatched_predictions, unmatched_routes = predict_unmatched(
            unmatched=unmatched,
            candidate_lists=candidate_lists,
            retriever=retriever,
            feature_builder=feature_builder,
            booster=booster,
            is_hangul_name=retrieval.is_hangul_name,
            prediction_batch_size=args.prediction_batch_size,
        )
        for index, prediction in unmatched_predictions.items():
            predictions.loc[index] = prediction
            routes.loc[index] = unmatched_routes[index]

    if predictions.isna().any():
        raise AssertionError(f"예측하지 못한 test 행이 있습니다: {int(predictions.isna().sum())}개")
    catalog_ids = set(catalog["vessel_id"].astype(str))
    invalid = ~predictions.astype(str).isin(catalog_ids)
    if invalid.any():
        raise AssertionError(f"catalog에 없는 vessel_id를 예측했습니다: {int(invalid.sum())}개")

    prediction_frame = pd.DataFrame(
        {
            "sample_id": test["sample_id"].astype(str),
            "target_vessel_id": predictions.astype(str),
        }
    )
    submission = sample[["sample_id"]].astype({"sample_id": str}).merge(
        prediction_frame,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if submission["target_vessel_id"].isna().any() or len(submission) != len(sample):
        raise AssertionError("submission.csv의 행 수 또는 예측값이 올바르지 않습니다.")

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    diagnostics = {
        "rows": int(len(test)),
        "route_counts": routes.value_counts().to_dict(),
        "unique_predictions": int(predictions.nunique()),
        "order_matches_sample_submission": bool(submission["sample_id"].tolist() == sample["sample_id"].astype(str).tolist()),
        "elapsed_seconds": time.perf_counter() - total_started,
        "output_path": str(output_path),
    }
    diagnostics_path = output_path.parent / "inference_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("submission.csv 생성 완료")
    print(f"test 행: {len(test):,} / 고유 예측 선박: {predictions.nunique():,}")
    print(f"예측 경로: {diagnostics['route_counts']}")
    print(f"sample_submission 순서 일치: {diagnostics['order_matches_sample_submission']}")
    print(f"추론 시간: {diagnostics['elapsed_seconds']:.2f}초")
    print(f"제출 파일: {output_path}")
    print("=" * 76)
    return submission


def main() -> None:
    args = parse_args()
    if args.hard_negatives < 1 or args.prediction_batch_size < 1 or args.retrieval_batch_size < 1:
        raise ValueError("hard negative 및 배치 설정은 1 이상이어야 합니다.")

    retrieval = load_module(args.retrieval_script.resolve(), "candidate_retrieval")
    pair_features = load_module(args.feature_script.resolve(), "pair_features")
    ranker_code = load_module(args.ranker_script.resolve(), "ranker_train")

    if args.mode in {"train", "all"}:
        train_final_model(args, retrieval, pair_features, ranker_code)

    if args.mode in {"predict", "all"}:
        test_path = args.test_path.resolve() if args.test_path else args.data_dir.resolve() / "test.csv"
        if args.mode == "all" and not test_path.exists():
            print(f"[안내] 비공개 test.csv가 없어 추론은 건너뜁니다: {test_path}")
            print("평가 서버에서는 data/test.csv가 제공되면 --mode predict로 submission.csv를 생성합니다.")
        else:
            predict_submission(args, retrieval, pair_features)


if __name__ == "__main__":
    main()
