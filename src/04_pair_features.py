from __future__ import annotations

"""후보 선박 쌍을 랭킹 모델용 수치 특징으로 변환한다.

03_candidate_retrieval.py가 한 입력 행마다 수백 개의 후보를 찾았다면,
이 파일은 각 (입력 행, 후보 선박) 쌍을 다음과 같은 숫자로 표현한다.

1. 검색 순위: 후보 검색기가 얼마나 먼저 찾았는가
2. 이름 유사도: 원문/한글 로마자화 문자열이 얼마나 비슷한가
3. 톤수 유사도: 신고 총톤수와 catalog 총톤수의 차이가 얼마나 작은가
4. 메타데이터: 신고 국적·선종과 catalog 국적·선종의 학습된 대응관계
5. 결측/규모 정보: 어떤 값이 비었고 후보 선박의 규모가 어느 정도인가

출력 joblib 파일은 05_ranker_train.py에서 바로 읽을 수 있는 학습 행렬이다.
검증 target_vessel_id는 이 단계의 검색기/통계/특징 생성에 사용하지 않는다.
"""

import argparse
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


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


FEATURE_NAMES = [
    # 검색기 자체가 부여한 순위 특징
    "retrieval_rank",
    "retrieval_rank_ratio",
    "retrieval_inverse_rank",
    "retrieval_log_rank",
    # 입력 행의 상태
    "query_is_hangul",
    "query_name_length",
    "query_roman_length",
    "query_call_sign_present",
    "query_gross_missing",
    "query_international_missing",
    # 선박명 비교
    "name_exact",
    "roman_exact",
    "name_ratio",
    "roman_ratio",
    "roman_partial_ratio",
    "roman_wratio",
    "roman_bigram_jaccard",
    "roman_trigram_jaccard",
    "roman_prefix_ratio",
    "roman_length_ratio",
    # 톤수 비교
    "gross_exact",
    "international_exact",
    "gross_relative_gap",
    "international_relative_gap",
    "best_relative_gap",
    "gross_log_absolute_gap",
    "international_log_absolute_gap",
    "catalog_gross_missing",
    # 국적·선종 대응관계
    "flag_text_exact",
    "type_text_exact",
    "metadata_pair_seen",
    "metadata_pair_rank",
    "metadata_pair_inverse_rank",
    "metadata_pair_probability",
    "metadata_pair_log_count",
    # catalog 후보 자체의 제원
    "catalog_log_gross",
    "catalog_log_deadweight",
    "catalog_log_size",
    "catalog_deadweight_gross_ratio",
    "catalog_size_deadweight_relative_gap",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="선박 후보 쌍 특징 및 랭커 학습 데이터 생성")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--assignment-path",
        type=Path,
        default=project_dir / "results" / "validation" / "validation_assignments.csv",
    )
    parser.add_argument(
        "--retrieval-script",
        type=Path,
        default=project_dir / "src" / "03_candidate_retrieval.py",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=project_dir / "results" / "pair_features",
    )
    parser.add_argument("--hard-negatives", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="0이면 전체 fit 행 사용. 빠른 점검은 1000처럼 지정한다.",
    )

    # 03번과 같은 후보 검색 설정이다.
    parser.add_argument("--name-candidates", type=int, default=120)
    parser.add_argument("--metadata-pairs", type=int, default=50)
    parser.add_argument("--per-pair-tonnage", type=int, default=15)
    parser.add_argument("--global-tonnage", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=600)
    return parser.parse_args()


def load_retrieval_module(path: Path) -> ModuleType:
    """숫자로 시작하는 03번 파일을 일반 모듈처럼 동적으로 불러온다."""
    if not path.exists():
        raise FileNotFoundError(f"후보 검색 파일을 찾을 수 없습니다: {path}")
    spec = importlib.util.spec_from_file_location("candidate_retrieval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"후보 검색 모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(missing)}")


def finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def relative_gap(left: float | None, right: float | None) -> float:
    """규모가 다른 선박도 비교할 수 있게 절대차를 큰 값으로 나눈다."""
    if left is None or right is None:
        return 10.0
    return min(abs(left - right) / max(abs(left), abs(right), 1.0), 10.0)


def log_absolute_gap(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return math.log1p(1_000_000.0)
    return math.log1p(abs(left - right))


def exact_number(left: float | None, right: float | None) -> float:
    return float(left is not None and right is not None and left == right)


def ngrams(text: str, n: int) -> set[str]:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def jaccard(left: str, right: str, n: int) -> float:
    left_set, right_set = ngrams(left, n), ngrams(right, n)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def prefix_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    same = 0
    while same < limit and left[same] == right[same]:
        same += 1
    return same / max(len(left), len(right))


def length_ratio(left: str, right: str) -> float:
    maximum = max(len(left), len(right))
    return min(len(left), len(right)) / maximum if maximum else 0.0


class PairFeatureBuilder:
    """한 입력 행과 여러 catalog 후보 사이의 특징을 계산한다."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        fit_train: pd.DataFrame,
        retriever: Any,
        retrieval_module: ModuleType,
    ) -> None:
        self.catalog = catalog.reset_index(drop=True).copy()
        self.retriever = retriever
        self.normalize_text = retrieval_module.normalize_text
        self.romanize_hangul = retrieval_module.romanize_hangul
        self.is_hangul_name = retrieval_module.is_hangul_name

        self.catalog_name = self.catalog["vessel_name"].map(self.normalize_text).tolist()
        self.catalog_roman = self.catalog["vessel_name"].map(self.romanize_hangul).tolist()
        self.catalog_flag = self.catalog["flag_name"].map(self.normalize_text).tolist()
        self.catalog_type = self.catalog["vessel_type"].map(self.normalize_text).tolist()
        self.catalog_gross = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.catalog_deadweight = pd.to_numeric(self.catalog["deadweight_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.catalog_size = pd.to_numeric(self.catalog["size"], errors="coerce").to_numpy(dtype=float)

        # 입력의 한글 국적/선종과 catalog의 영문 국적/선종 사이 대응관계를
        # 오직 unseen-fit 학습 부분에서 빈도로 학습한다.
        target_info = self.catalog[["vessel_id"]].copy()
        target_info["_catalog_flag"] = self.catalog_flag
        target_info["_catalog_type"] = self.catalog_type
        joined = fit_train.merge(
            target_info,
            left_on="target_vessel_id",
            right_on="vessel_id",
            how="left",
            validate="many_to_one",
        )
        if joined["vessel_id"].isna().any():
            raise ValueError("fit_train의 일부 target_vessel_id가 catalog에 없습니다.")
        joined["_query_flag"] = joined["flag_name"].map(self.normalize_text)
        joined["_query_type"] = joined["vessel_type"].map(self.normalize_text)

        counts: Counter[tuple[str, str, str, str]] = Counter(
            zip(
                joined["_query_flag"],
                joined["_query_type"],
                joined["_catalog_flag"],
                joined["_catalog_type"],
            )
        )
        totals: Counter[tuple[str, str]] = Counter(zip(joined["_query_flag"], joined["_query_type"]))
        pairs_by_query: defaultdict[tuple[str, str], list[tuple[tuple[str, str], int]]] = defaultdict(list)
        for (query_flag, query_type, catalog_flag, catalog_type), count in counts.items():
            pairs_by_query[(query_flag, query_type)].append(((catalog_flag, catalog_type), count))

        self.metadata_stats: dict[tuple[str, str], dict[tuple[str, str], tuple[int, int, float]]] = {}
        for query_key, values in pairs_by_query.items():
            values.sort(key=lambda item: (-item[1], item[0]))
            total = totals[query_key]
            number_of_pairs = len(values)
            # add-one smoothing으로 희소 조합의 확률 0/1 과신을 줄인다.
            self.metadata_stats[query_key] = {
                pair: (rank, count, (count + 1.0) / (total + number_of_pairs))
                for rank, (pair, count) in enumerate(values, start=1)
            }

    def transform_group(
        self,
        query: pd.Series,
        candidate_indices: list[int],
        retrieval_ranks: list[int] | None = None,
    ) -> np.ndarray:
        """한 입력 행에 속한 후보들을 FEATURE_NAMES 순서의 float32 행렬로 변환한다."""
        if retrieval_ranks is None:
            retrieval_ranks = list(range(1, len(candidate_indices) + 1))
        if len(candidate_indices) != len(retrieval_ranks):
            raise ValueError("후보 수와 원래 검색 순위 수가 다릅니다.")
        query_name = self.normalize_text(query.get("vessel_name"))
        query_roman = self.romanize_hangul(query.get("vessel_name"))
        query_flag = self.normalize_text(query.get("flag_name"))
        query_type = self.normalize_text(query.get("vessel_type"))
        query_gross = finite_positive(query.get("gross_tonnage"))
        query_international = finite_positive(query.get("international_gross_tonnage"))
        call_sign_present = bool(self.normalize_text(query.get("call_sign")))
        metadata = self.metadata_stats.get((query_flag, query_type), {})

        matrix = np.empty((len(candidate_indices), len(FEATURE_NAMES)), dtype=np.float32)
        for position, candidate_index in enumerate(candidate_indices):
            candidate_name = self.catalog_name[candidate_index]
            candidate_roman = self.catalog_roman[candidate_index]
            candidate_flag = self.catalog_flag[candidate_index]
            candidate_type = self.catalog_type[candidate_index]
            candidate_gross = finite_positive(self.catalog_gross[candidate_index])
            candidate_deadweight = finite_positive(self.catalog_deadweight[candidate_index])
            candidate_size = finite_positive(self.catalog_size[candidate_index])
            # 학습 배열 안의 위치가 아니라 03번 검색기가 반환한 원래 순위다.
            # 정답을 학습 목적으로 배열 첫 행에 두더라도 rank=1로 바뀌면 안 된다.
            rank = retrieval_ranks[position]

            pair_rank, pair_count, pair_probability = metadata.get(
                (candidate_flag, candidate_type),
                (0, 0, 0.0),
            )
            gross_gap = relative_gap(query_gross, candidate_gross)
            international_gap = relative_gap(query_international, candidate_gross)

            matrix[position] = np.asarray(
                [
                    rank,
                    rank / max(self.retriever.max_candidates, 1),
                    1.0 / rank,
                    math.log1p(rank),
                    float(self.is_hangul_name(query.get("vessel_name"))),
                    len(query_name),
                    len(query_roman),
                    float(call_sign_present),
                    float(query_gross is None),
                    float(query_international is None),
                    float(bool(query_name) and query_name == candidate_name),
                    float(bool(query_roman) and query_roman == candidate_roman),
                    fuzz.ratio(query_name, candidate_name) / 100.0,
                    fuzz.ratio(query_roman, candidate_roman) / 100.0,
                    fuzz.partial_ratio(query_roman, candidate_roman) / 100.0,
                    fuzz.WRatio(query_roman, candidate_roman) / 100.0,
                    jaccard(query_roman, candidate_roman, 2),
                    jaccard(query_roman, candidate_roman, 3),
                    prefix_ratio(query_roman, candidate_roman),
                    length_ratio(query_roman, candidate_roman),
                    exact_number(query_gross, candidate_gross),
                    exact_number(query_international, candidate_gross),
                    gross_gap,
                    international_gap,
                    min(gross_gap, international_gap),
                    log_absolute_gap(query_gross, candidate_gross),
                    log_absolute_gap(query_international, candidate_gross),
                    float(candidate_gross is None),
                    float(bool(query_flag) and query_flag == candidate_flag),
                    float(bool(query_type) and query_type == candidate_type),
                    float(pair_rank > 0),
                    pair_rank,
                    1.0 / pair_rank if pair_rank > 0 else 0.0,
                    pair_probability,
                    math.log1p(pair_count),
                    math.log1p(candidate_gross or 0.0),
                    math.log1p(candidate_deadweight or 0.0),
                    math.log1p(candidate_size or 0.0),
                    (candidate_deadweight / candidate_gross)
                    if candidate_deadweight is not None and candidate_gross is not None
                    else 0.0,
                    relative_gap(candidate_size, candidate_deadweight),
                ],
                dtype=np.float32,
            )
        return matrix


def select_training_candidates(
    candidates: list[int],
    true_index: int,
    hard_negatives: int,
) -> tuple[list[int], list[int], bool]:
    """검색 상위의 헷갈리는 오답을 고르고 정답 후보를 반드시 한 번 포함한다."""
    rank_by_index = {candidate_index: rank for rank, candidate_index in enumerate(candidates, start=1)}
    forced_positive = true_index not in rank_by_index
    negatives = [index for index in candidates if index != true_index][:hard_negatives]
    selected = [true_index, *negatives]
    original_ranks = [
        rank_by_index.get(candidate_index, len(candidates) + 1)
        for candidate_index in selected
    ]
    if len(set(selected)) != len(selected):
        raise AssertionError("한 학습 그룹 안에 중복 후보가 있습니다.")
    return selected, original_ranks, forced_positive


def build_training_matrix(
    fit_train: pd.DataFrame,
    candidate_lists: list[list[int]],
    retriever: Any,
    feature_builder: PairFeatureBuilder,
    hard_negatives: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_sizes: list[int] = []
    sample_ids: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    forced_count = 0

    for row_number, (_, query) in enumerate(fit_train.iterrows()):
        true_id = str(query["target_vessel_id"])
        true_index = retriever.id_to_index[true_id]
        selected, original_ranks, forced = select_training_candidates(
            candidate_lists[row_number],
            true_index,
            hard_negatives,
        )
        forced_count += int(forced)
        matrix = feature_builder.transform_group(query, selected, original_ranks)
        label = np.zeros(len(selected), dtype=np.int8)
        label[0] = 1

        matrices.append(matrix)
        labels.append(label)
        group_sizes.append(len(selected))
        sample_ids.append(str(query["sample_id"]))

        if len(preview_rows) < 200:
            for local_index, candidate_index in enumerate(selected):
                preview = {
                    "sample_id": str(query["sample_id"]),
                    "candidate_vessel_id": retriever.ids[candidate_index],
                    "label": int(label[local_index]),
                }
                preview.update(
                    {
                        feature_name: float(matrix[local_index, feature_index])
                        for feature_index, feature_name in enumerate(FEATURE_NAMES)
                    }
                )
                preview_rows.append(preview)
                if len(preview_rows) >= 200:
                    break

    X = np.vstack(matrices).astype(np.float32, copy=False)
    y = np.concatenate(labels)
    groups = np.asarray(group_sizes, dtype=np.int32)
    if X.shape[0] != y.shape[0] or X.shape[0] != int(groups.sum()):
        raise AssertionError("특징 행, 라벨, 그룹 크기가 일치하지 않습니다.")
    if not np.isfinite(X).all():
        raise AssertionError("특징 행렬에 NaN 또는 무한대가 있습니다.")
    if int(y.sum()) != len(groups):
        raise AssertionError("각 그룹에는 정답 후보가 정확히 하나 있어야 합니다.")

    artifact = {
        "X": X,
        "y": y,
        "group_sizes": groups,
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "feature_names": FEATURE_NAMES,
        "metadata": {
            "validation_split": "unseen_split fit only",
            "rows_are_grouped_by_sample": True,
            "one_positive_per_group": True,
            "hard_negatives_per_group": hard_negatives,
        },
    }
    diagnostics = {
        "forced_positive_groups": forced_count,
        "retrieval_hit_groups": len(groups) - forced_count,
    }
    return artifact, preview_rows, diagnostics


def main() -> None:
    args = parse_args()
    if args.hard_negatives < 1:
        raise ValueError("--hard-negatives는 1 이상이어야 합니다.")
    if args.max_train_rows < 0:
        raise ValueError("--max-train-rows는 0 이상이어야 합니다.")

    data_dir = args.data_dir.resolve()
    assignment_path = args.assignment_path.resolve()
    retrieval_script = args.retrieval_script.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    for path in [train_path, catalog_path, assignment_path, retrieval_script]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    assignments = pd.read_csv(assignment_path, encoding="utf-8-sig")
    check_columns(train, TRAIN_COLUMNS, "train.csv")
    check_columns(catalog, CATALOG_COLUMNS, "vessel_catalog.csv")
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
    valid_targets = set(marked.loc[marked["unseen_split"] == "valid", "target_vessel_id"].astype(str))
    if set(fit_train["target_vessel_id"].astype(str)) & valid_targets:
        raise AssertionError("미관측 검증 split에 target_vessel_id 누수가 있습니다.")

    if args.max_train_rows and args.max_train_rows < len(fit_train):
        # 고정 seed 표본이므로 반복 실행 결과가 같다. 빠른 코드 점검용 옵션이다.
        fit_train = (
            fit_train.sample(n=args.max_train_rows, random_state=args.seed)
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )

    retrieval = load_retrieval_module(retrieval_script)
    started = time.perf_counter()
    retriever = retrieval.CandidateRetriever(
        catalog=catalog,
        fit_train=fit_train,
        name_candidates=args.name_candidates,
        metadata_pairs=args.metadata_pairs,
        per_pair_tonnage=args.per_pair_tonnage,
        global_tonnage=args.global_tonnage,
        max_candidates=args.max_candidates,
    )
    feature_builder = PairFeatureBuilder(catalog, fit_train, retriever, retrieval)
    index_seconds = time.perf_counter() - started

    retrieval_started = time.perf_counter()
    candidate_lists = retriever.retrieve_many(fit_train, args.batch_size)
    retrieval_seconds = time.perf_counter() - retrieval_started

    feature_started = time.perf_counter()
    artifact, preview_rows, diagnostics = build_training_matrix(
        fit_train=fit_train,
        candidate_lists=candidate_lists,
        retriever=retriever,
        feature_builder=feature_builder,
        hard_negatives=args.hard_negatives,
    )
    feature_seconds = time.perf_counter() - feature_started

    artifact_path = result_dir / "train_pair_features.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    pd.DataFrame(preview_rows).to_csv(
        result_dir / "pair_feature_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    X = artifact["X"]
    y = artifact["y"]
    groups = artifact["group_sizes"]
    summary = {
        "purpose": "05_ranker_train.py에서 사용할 query-candidate pair feature 생성",
        "leakage_control": {
            "source_rows": "02_validation.py의 unseen_split == fit만 사용",
            "validation_labels_used": False,
            "target_overlap_with_unseen_validation": 0,
        },
        "settings": {
            "seed": args.seed,
            "hard_negatives": args.hard_negatives,
            "max_train_rows": args.max_train_rows,
            "retrieval": {
                "name_candidates": args.name_candidates,
                "metadata_pairs": args.metadata_pairs,
                "per_pair_tonnage": args.per_pair_tonnage,
                "global_tonnage": args.global_tonnage,
                "max_candidates": args.max_candidates,
            },
        },
        "dataset": {
            "groups": int(len(groups)),
            "pair_rows": int(len(y)),
            "features": int(X.shape[1]),
            "positive_rows": int(y.sum()),
            "negative_rows": int((y == 0).sum()),
            "mean_group_size": float(groups.mean()),
            "matrix_megabytes": float(X.nbytes / 1024**2),
            **diagnostics,
        },
        "feature_names": FEATURE_NAMES,
        "timing_seconds": {
            "index_and_metadata_build": index_seconds,
            "candidate_retrieval": retrieval_seconds,
            "feature_generation": feature_seconds,
            "total_before_save": index_seconds + retrieval_seconds + feature_seconds,
        },
        "output": {
            "artifact": str(artifact_path),
            "preview": str(result_dir / "pair_feature_sample.csv"),
        },
    }
    with (result_dir / "pair_feature_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("후보 쌍 특징 생성 완료")
    print(f"학습 그룹(입력 행): {len(groups):,}")
    print(f"후보 쌍 행: {len(y):,} / 특징: {X.shape[1]:,}개")
    print(f"정답/오답: {int(y.sum()):,} / {int((y == 0).sum()):,}")
    print(f"검색 누락으로 정답을 강제 추가한 그룹: {diagnostics['forced_positive_groups']:,}")
    print(f"특징 행렬 메모리: {X.nbytes / 1024**2:.1f} MB")
    print(
        f"인덱스 {index_seconds:.2f}초 / 검색 {retrieval_seconds:.2f}초 / "
        f"특징 {feature_seconds:.2f}초"
    )
    print(f"학습 파일: {artifact_path}")
    print(f"결과 저장 위치: {result_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
