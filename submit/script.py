from __future__ import annotations

"""항만 선박 매칭 대회 최종 추론 코드.

평가 서버 실행:
    python script.py

입력:
    data/train.csv
    data/vessel_catalog.csv
    data/test.csv
    data/sample_submission.csv  # 없으면 test 순서를 사용

출력:
    output/submission.csv

주의: 모든 test 행은 독립적으로 처리하며 다른 test 행이나 test 분포를 사용하지 않는다.
"""

import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

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
TRAIN_COLUMNS = TEST_COLUMNS | {"target_vessel_id"}
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

CHOSEONG = ["G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
JUNGSEONG = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "UI", "I"]
JONGSEONG = ["", "G", "GG", "GS", "N", "NJ", "NH", "D", "L", "LG", "LM", "LB", "LS", "LT", "LP", "LH", "M", "B", "BS", "S", "SS", "NG", "J", "CH", "K", "T", "P", "H"]

FEATURE_NAMES = [
    "retrieval_rank",
    "retrieval_rank_ratio",
    "retrieval_inverse_rank",
    "retrieval_log_rank",
    "query_is_hangul",
    "query_name_length",
    "query_roman_length",
    "query_call_sign_present",
    "query_gross_missing",
    "query_international_missing",
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
    "gross_exact",
    "international_exact",
    "gross_relative_gap",
    "international_relative_gap",
    "best_relative_gap",
    "gross_log_absolute_gap",
    "international_log_absolute_gap",
    "catalog_gross_missing",
    "flag_text_exact",
    "type_text_exact",
    "metadata_pair_seen",
    "metadata_pair_rank",
    "metadata_pair_inverse_rank",
    "metadata_pair_probability",
    "metadata_pair_log_count",
    "catalog_log_gross",
    "catalog_log_deadweight",
    "catalog_log_size",
    "catalog_deadweight_gross_ratio",
    "catalog_size_deadweight_relative_gap",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper().strip()
    return re.sub(r"[^0-9A-Z가-힣]", "", text)


def romanize_hangul(value: object) -> str:
    if pd.isna(value):
        return ""
    output: list[str] = []
    for char in unicodedata.normalize("NFKC", str(value)).upper():
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            output.extend([CHOSEONG[code // 588], JUNGSEONG[(code % 588) // 28], JONGSEONG[code % 28]])
        else:
            output.append(char)
    return re.sub(r"[^0-9A-Z]", "", "".join(output))


def is_hangul_name(value: object) -> bool:
    return bool(re.search(r"[가-힣]", str(value)))


def finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def relative_gap(left: float | None, right: float | None) -> float:
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
    limit, same = min(len(left), len(right)), 0
    while same < limit and left[same] == right[same]:
        same += 1
    return same / max(len(left), len(right))


def length_ratio(left: str, right: str) -> float:
    maximum = max(len(left), len(right))
    return min(len(left), len(right)) / maximum if maximum else 0.0


def check_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(missing)}")


class TonnageIndex:
    def __init__(self, gross_tonnage: np.ndarray, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        valid = np.isfinite(gross_tonnage[indices]) & (gross_tonnage[indices] > 0)
        indices = indices[valid]
        order = np.argsort(gross_tonnage[indices], kind="mergesort")
        self.indices = indices[order]
        self.values = gross_tonnage[self.indices]

    def nearest(self, value: object, count: int) -> list[int]:
        number = finite_positive(value)
        if number is None or len(self.values) == 0 or count <= 0:
            return []
        position = int(np.searchsorted(self.values, number))
        left, right, output = position - 1, position, []
        while len(output) < count and (left >= 0 or right < len(self.values)):
            if right >= len(self.values):
                choose_left = True
            elif left < 0:
                choose_left = False
            else:
                choose_left = abs(self.values[left] - number) <= abs(self.values[right] - number)
            output.append(int(self.indices[left if choose_left else right]))
            left -= int(choose_left)
            right += int(not choose_left)
        return output


class CandidateRetriever:
    def __init__(self, catalog: pd.DataFrame, train: pd.DataFrame, settings: dict[str, int]) -> None:
        self.catalog = catalog.reset_index(drop=True).copy()
        self.name_candidates = int(settings["name_candidates"])
        self.metadata_pairs = int(settings["metadata_pairs"])
        self.per_pair_tonnage = int(settings["per_pair_tonnage"])
        self.global_tonnage = int(settings["global_tonnage"])
        self.max_candidates = int(settings["max_candidates"])

        if not self.catalog["vessel_id"].is_unique:
            raise ValueError("catalog의 vessel_id는 고유해야 합니다.")
        self.ids = self.catalog["vessel_id"].astype(str).tolist()
        self.id_to_index = {vessel_id: index for index, vessel_id in enumerate(self.ids)}
        self.catalog["_name_norm"] = self.catalog["vessel_name"].map(normalize_text)
        self.catalog["_name_roman"] = self.catalog["vessel_name"].map(romanize_hangul)
        self.catalog["_flag_norm"] = self.catalog["flag_name"].map(normalize_text)
        self.catalog["_type_norm"] = self.catalog["vessel_type"].map(normalize_text)
        self.gross_tonnage = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)

        self.exact_name: dict[str, list[int]] = defaultdict(list)
        self.exact_tonnage: dict[float, list[int]] = defaultdict(list)
        for index, (name, tonnage) in enumerate(zip(self.catalog["_name_norm"], self.gross_tonnage)):
            if name:
                self.exact_name[name].append(index)
            if np.isfinite(tonnage) and tonnage > 0:
                self.exact_tonnage[float(tonnage)].append(index)

        all_indices = np.arange(len(self.catalog), dtype=np.int64)
        self.global_tonnage_index = TonnageIndex(self.gross_tonnage, all_indices)
        self.pair_tonnage_indices = {
            pair: TonnageIndex(self.gross_tonnage, group.index.to_numpy(dtype=np.int64))
            for pair, group in self.catalog.groupby(["_flag_norm", "_type_norm"], sort=False)
        }
        self.metadata_pair_map = self._learn_metadata_pair_map(train)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        catalog_matrix = self.vectorizer.fit_transform(self.catalog["_name_roman"])
        self.name_index = NearestNeighbors(
            n_neighbors=min(self.name_candidates, len(self.catalog)),
            metric="cosine",
            algorithm="brute",
            n_jobs=6,
        ).fit(catalog_matrix)

    def _learn_metadata_pair_map(self, train: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, str]]]:
        target_info = self.catalog[["vessel_id", "_flag_norm", "_type_norm"]]
        joined = train.merge(target_info, left_on="target_vessel_id", right_on="vessel_id", how="left", validate="many_to_one")
        if joined["vessel_id"].isna().any():
            raise ValueError("train의 일부 target_vessel_id가 catalog에 없습니다.")
        joined["_query_flag"] = joined["flag_name"].map(normalize_text)
        joined["_query_type"] = joined["vessel_type"].map(normalize_text)
        counts = (
            joined.groupby(["_query_flag", "_query_type", "_flag_norm", "_type_norm"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["_query_flag", "_query_type", "count"], ascending=[True, True, False])
        )
        mapping: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for key, group in counts.groupby(["_query_flag", "_query_type"], sort=False):
            mapping[key] = list(zip(group["_flag_norm"], group["_type_norm"]))
        return mapping

    def _name_neighbors(self, queries: pd.DataFrame, batch_size: int) -> list[np.ndarray]:
        names = queries["vessel_name"].map(romanize_hangul).tolist()
        output: list[np.ndarray] = []
        for start in range(0, len(names), batch_size):
            matrix = self.vectorizer.transform(names[start : start + batch_size])
            output.extend(self.name_index.kneighbors(matrix, return_distance=False))
        return output

    def retrieve_many(self, queries: pd.DataFrame, batch_size: int = 512) -> list[list[int]]:
        name_neighbors = self._name_neighbors(queries, batch_size)
        records = queries.to_dict("records")
        all_candidates: list[list[int]] = []
        for row_number, row in enumerate(records):
            candidates, used = [], set()

            def append(indices: list[int] | np.ndarray) -> None:
                for index in indices:
                    index = int(index)
                    if index not in used and len(candidates) < self.max_candidates:
                        candidates.append(index)
                        used.add(index)

            query_name = normalize_text(row.get("vessel_name"))
            query_roman = romanize_hangul(row.get("vessel_name"))
            if query_name:
                append(self.exact_name.get(query_name, []))
            if query_roman:
                append(name_neighbors[row_number])

            gross = row.get("gross_tonnage")
            international = row.get("international_gross_tonnage")
            for value in [gross, international]:
                number = finite_positive(value)
                if number is not None:
                    append(self.exact_tonnage.get(number, []))

            key = (normalize_text(row.get("flag_name")), normalize_text(row.get("vessel_type")))
            for pair in self.metadata_pair_map.get(key, [])[: self.metadata_pairs]:
                pair_index = self.pair_tonnage_indices.get(pair)
                if pair_index is not None:
                    append(pair_index.nearest(gross, self.per_pair_tonnage))
                    append(pair_index.nearest(international, self.per_pair_tonnage))
            append(self.global_tonnage_index.nearest(gross, self.global_tonnage))
            append(self.global_tonnage_index.nearest(international, self.global_tonnage))
            all_candidates.append(candidates)
        return all_candidates


class PairFeatureBuilder:
    def __init__(self, catalog: pd.DataFrame, train: pd.DataFrame, retriever: CandidateRetriever) -> None:
        self.catalog = catalog.reset_index(drop=True).copy()
        self.retriever = retriever
        self.catalog_name = self.catalog["vessel_name"].map(normalize_text).tolist()
        self.catalog_roman = self.catalog["vessel_name"].map(romanize_hangul).tolist()
        self.catalog_flag = self.catalog["flag_name"].map(normalize_text).tolist()
        self.catalog_type = self.catalog["vessel_type"].map(normalize_text).tolist()
        self.catalog_gross = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.catalog_deadweight = pd.to_numeric(self.catalog["deadweight_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.catalog_size = pd.to_numeric(self.catalog["size"], errors="coerce").to_numpy(dtype=float)

        target_info = self.catalog[["vessel_id"]].copy()
        target_info["_catalog_flag"] = self.catalog_flag
        target_info["_catalog_type"] = self.catalog_type
        joined = train.merge(target_info, left_on="target_vessel_id", right_on="vessel_id", how="left", validate="many_to_one")
        if joined["vessel_id"].isna().any():
            raise ValueError("train의 일부 target_vessel_id가 catalog에 없습니다.")
        joined["_query_flag"] = joined["flag_name"].map(normalize_text)
        joined["_query_type"] = joined["vessel_type"].map(normalize_text)

        counts: Counter[tuple[str, str, str, str]] = Counter(
            zip(joined["_query_flag"], joined["_query_type"], joined["_catalog_flag"], joined["_catalog_type"])
        )
        totals: Counter[tuple[str, str]] = Counter(zip(joined["_query_flag"], joined["_query_type"]))
        pairs_by_query: defaultdict[tuple[str, str], list[tuple[tuple[str, str], int]]] = defaultdict(list)
        for (query_flag, query_type, catalog_flag, catalog_type), count in counts.items():
            pairs_by_query[(query_flag, query_type)].append(((catalog_flag, catalog_type), count))
        self.metadata_stats: dict[tuple[str, str], dict[tuple[str, str], tuple[int, int, float]]] = {}
        for query_key, values in pairs_by_query.items():
            values.sort(key=lambda item: (-item[1], item[0]))
            total, number_of_pairs = totals[query_key], len(values)
            self.metadata_stats[query_key] = {
                pair: (rank, count, (count + 1.0) / (total + number_of_pairs))
                for rank, (pair, count) in enumerate(values, start=1)
            }

    def transform_group(self, query: pd.Series, candidate_indices: list[int]) -> np.ndarray:
        query_name = normalize_text(query.get("vessel_name"))
        query_roman = romanize_hangul(query.get("vessel_name"))
        query_flag = normalize_text(query.get("flag_name"))
        query_type = normalize_text(query.get("vessel_type"))
        query_gross = finite_positive(query.get("gross_tonnage"))
        query_international = finite_positive(query.get("international_gross_tonnage"))
        call_sign_present = bool(normalize_text(query.get("call_sign")))
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
            rank = position + 1
            pair_rank, pair_count, pair_probability = metadata.get((candidate_flag, candidate_type), (0, 0, 0.0))
            gross_gap = relative_gap(query_gross, candidate_gross)
            international_gap = relative_gap(query_international, candidate_gross)
            matrix[position] = np.asarray(
                [
                    rank,
                    rank / max(self.retriever.max_candidates, 1),
                    1.0 / rank,
                    math.log1p(rank),
                    float(is_hangul_name(query.get("vessel_name"))),
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
                    candidate_deadweight / candidate_gross if candidate_deadweight is not None and candidate_gross is not None else 0.0,
                    relative_gap(candidate_size, candidate_deadweight),
                ],
                dtype=np.float32,
            )
        return matrix


def make_unique_target_map(train: pd.DataFrame, column: str) -> dict[str, str]:
    frame = pd.DataFrame({"key": train[column].map(normalize_text), "target": train["target_vessel_id"].astype(str)})
    frame = frame[frame["key"] != ""]
    grouped = frame.groupby("key")["target"].agg(["first", "nunique"])
    return grouped.loc[grouped["nunique"] == 1, "first"].to_dict()


def direct_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    call_sign_map = make_unique_target_map(train, "call_sign")
    vessel_name_map = make_unique_target_map(train, "vessel_name")
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
    retriever: CandidateRetriever,
    feature_builder: PairFeatureBuilder,
    booster: lgb.Booster,
    prediction_batch_size: int = 64,
) -> tuple[dict[int, str], dict[int, str]]:
    predictions, routes = {}, {}
    unmatched = unmatched.reset_index().rename(columns={"index": "_original_index"})
    for start in range(0, len(unmatched), prediction_batch_size):
        end = min(start + prediction_batch_size, len(unmatched))
        matrices, matrix_rows, matrix_candidates = [], [], []
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
            scores = booster.predict(np.vstack(matrices).astype(np.float32, copy=False))
            offset = 0
            for original_index, candidates in zip(matrix_rows, matrix_candidates):
                size = len(candidates)
                best_index = candidates[int(np.argmax(scores[offset : offset + size]))]
                predictions[original_index] = retriever.ids[best_index]
                routes[original_index] = "hangul_ranker"
                offset += size
    return predictions, routes


def main() -> None:
    started = time.perf_counter()
    train_path = DATA_DIR / "train.csv"
    catalog_path = DATA_DIR / "vessel_catalog.csv"
    test_path = DATA_DIR / "test.csv"
    sample_path = DATA_DIR / "sample_submission.csv"
    model_path = MODEL_DIR / "ranker_model.txt"
    metadata_path = MODEL_DIR / "model_metadata.json"
    for path in [train_path, catalog_path, test_path, model_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"필수 파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    check_columns(train, TRAIN_COLUMNS, "train.csv")
    check_columns(catalog, CATALOG_COLUMNS, "vessel_catalog.csv")
    check_columns(test, TEST_COLUMNS, "test.csv")
    if "target_vessel_id" in test.columns:
        raise ValueError("test.csv에는 target_vessel_id가 없어야 합니다.")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv의 sample_id는 결측 없이 고유해야 합니다.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["feature_names"] != FEATURE_NAMES:
        raise ValueError("저장 모델과 script.py의 특징 순서가 다릅니다.")
    booster = lgb.Booster(model_file=str(model_path))
    if booster.num_feature() != len(FEATURE_NAMES):
        raise ValueError("저장 모델의 특징 수가 script.py와 다릅니다.")

    retriever = CandidateRetriever(catalog, train, metadata["retrieval_parameters"])
    feature_builder = PairFeatureBuilder(catalog, train, retriever)
    predictions, routes = direct_predictions(train, test)
    unmatched = test.loc[predictions.isna()].copy()
    if len(unmatched):
        candidate_lists = retriever.retrieve_many(unmatched, batch_size=512)
        unmatched_predictions, unmatched_routes = predict_unmatched(
            unmatched,
            candidate_lists,
            retriever,
            feature_builder,
            booster,
        )
        for index, prediction in unmatched_predictions.items():
            predictions.loc[index] = prediction
            routes.loc[index] = unmatched_routes[index]

    if predictions.isna().any():
        raise AssertionError(f"예측하지 못한 test 행: {int(predictions.isna().sum())}개")
    catalog_ids = set(catalog["vessel_id"].astype(str))
    if not predictions.astype(str).isin(catalog_ids).all():
        raise AssertionError("catalog에 없는 vessel_id를 예측했습니다.")

    predicted = pd.DataFrame({"sample_id": test["sample_id"].astype(str), "target_vessel_id": predictions.astype(str)})
    if sample_path.exists():
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        if not {"sample_id", "target_vessel_id"}.issubset(sample.columns):
            raise ValueError("sample_submission.csv 형식이 올바르지 않습니다.")
        if sample["sample_id"].isna().any() or not sample["sample_id"].is_unique:
            raise ValueError("sample_submission.csv의 sample_id는 결측 없이 고유해야 합니다.")
        if set(sample["sample_id"].astype(str)) != set(test["sample_id"].astype(str)):
            raise ValueError("test와 sample_submission의 sample_id 집합이 다릅니다.")
        submission = sample[["sample_id"]].astype({"sample_id": str}).merge(predicted, on="sample_id", how="left", validate="one_to_one")
    else:
        submission = predicted

    if submission["target_vessel_id"].isna().any() or len(submission) != len(test):
        raise AssertionError("submission.csv의 행 수 또는 예측값이 올바르지 않습니다.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"submission 생성 완료: {output_path}")
    print(f"행 수: {len(submission):,}")
    print(f"예측 경로: {routes.value_counts().to_dict()}")
    print(f"총 추론 시간: {time.perf_counter() - started:.2f}초")


if __name__ == "__main__":
    main()
