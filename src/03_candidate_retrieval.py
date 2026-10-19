from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


TRAIN_COLUMNS = {
    "sample_id",
    "vessel_name",
    "gross_tonnage",
    "international_gross_tonnage",
    "flag_name",
    "vessel_type",
    "target_vessel_id",
}
CATALOG_COLUMNS = {
    "vessel_id",
    "vessel_name",
    "gross_tonnage",
    "flag_name",
    "vessel_type",
}

# 한글 음절을 결정적으로 영문화하기 위한 초성·중성·종성 표기다.
# 외부 선박 데이터나 번역 사전이 아니라 유니코드 음절 분해 규칙만 사용한다.
CHOSEONG = ["G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
JUNGSEONG = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "UI", "I"]
JONGSEONG = ["", "G", "GG", "GS", "N", "NJ", "NH", "D", "L", "LG", "LM", "LB", "LS", "LT", "LP", "LH", "M", "B", "BS", "S", "SS", "NG", "J", "CH", "K", "T", "P", "H"]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="선박 catalog 후보 검색기 검증")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--assignment-path", type=Path, default=project_dir / "results" / "validation" / "validation_assignments.csv")
    parser.add_argument("--result-dir", type=Path, default=project_dir / "results" / "candidate_retrieval")
    parser.add_argument("--name-candidates", type=int, default=120)
    parser.add_argument("--metadata-pairs", type=int, default=50)
    parser.add_argument("--per-pair-tonnage", type=int, default=15)
    parser.add_argument("--global-tonnage", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
            output.extend(
                [
                    CHOSEONG[code // 588],
                    JUNGSEONG[(code % 588) // 28],
                    JONGSEONG[code % 28],
                ]
            )
        else:
            output.append(char)
    return re.sub(r"[^0-9A-Z]", "", "".join(output))


def positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def is_hangul_name(value: object) -> bool:
    return bool(re.search(r"[가-힣]", str(value)))


def check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {sorted(missing)}")


class TonnageIndex:
    """catalog 총톤수를 정렬해 특정 값과 가까운 선박을 빠르게 찾는다."""

    def __init__(self, gross_tonnage: np.ndarray, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        valid = np.isfinite(gross_tonnage[indices]) & (gross_tonnage[indices] > 0)
        indices = indices[valid]
        order = np.argsort(gross_tonnage[indices], kind="mergesort")
        self.indices = indices[order]
        self.values = gross_tonnage[self.indices]

    def nearest(self, value: object, count: int) -> list[int]:
        number = positive_float(value)
        if number is None or len(self.values) == 0 or count <= 0:
            return []
        position = int(np.searchsorted(self.values, number))
        left, right = position - 1, position
        output: list[int] = []
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
    """이름, 톤수, 학습된 메타데이터 대응관계로 catalog 후보를 생성한다."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        fit_train: pd.DataFrame,
        name_candidates: int,
        metadata_pairs: int,
        per_pair_tonnage: int,
        global_tonnage: int,
        max_candidates: int,
    ) -> None:
        self.catalog = catalog.reset_index(drop=True).copy()
        self.name_candidates = name_candidates
        self.metadata_pairs = metadata_pairs
        self.per_pair_tonnage = per_pair_tonnage
        self.global_tonnage = global_tonnage
        self.max_candidates = max_candidates

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
        self.metadata_pair_map = self._learn_metadata_pair_map(fit_train)

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=False,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        catalog_matrix = self.vectorizer.fit_transform(self.catalog["_name_roman"])
        neighbor_count = min(self.name_candidates, len(self.catalog))
        self.name_index = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(catalog_matrix)

    def _learn_metadata_pair_map(self, fit_train: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, str]]]:
        target_info = self.catalog[["vessel_id", "_flag_norm", "_type_norm"]]
        joined = fit_train.merge(
            target_info,
            left_on="target_vessel_id",
            right_on="vessel_id",
            how="left",
            validate="many_to_one",
        )
        if joined["vessel_id"].isna().any():
            raise ValueError("fit_train의 일부 target_vessel_id가 catalog에 없습니다.")
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
            indices = self.name_index.kneighbors(matrix, return_distance=False)
            output.extend(indices)
        return output

    def retrieve_many(self, queries: pd.DataFrame, batch_size: int = 512) -> list[list[int]]:
        name_neighbors = self._name_neighbors(queries, batch_size)
        records = queries.to_dict("records")
        all_candidates: list[list[int]] = []

        for row_number, row in enumerate(records):
            candidates: list[int] = []
            used: set[int] = set()

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
                number = positive_float(value)
                if number is not None:
                    append(self.exact_tonnage.get(number, []))

            metadata_key = (normalize_text(row.get("flag_name")), normalize_text(row.get("vessel_type")))
            catalog_pairs = self.metadata_pair_map.get(metadata_key, [])[: self.metadata_pairs]
            for pair in catalog_pairs:
                pair_index = self.pair_tonnage_indices.get(pair)
                if pair_index is not None:
                    append(pair_index.nearest(gross, self.per_pair_tonnage))
                    append(pair_index.nearest(international, self.per_pair_tonnage))

            append(self.global_tonnage_index.nearest(gross, self.global_tonnage))
            append(self.global_tonnage_index.nearest(international, self.global_tonnage))
            all_candidates.append(candidates)
        return all_candidates


def recall_metrics(truth: pd.DataFrame, candidate_lists: list[list[int]], retriever: CandidateRetriever) -> dict[str, object]:
    if len(truth) != len(candidate_lists):
        raise ValueError("검증 행 수와 후보 목록 수가 다릅니다.")

    true_indices = np.array([retriever.id_to_index[str(value)] for value in truth["target_vessel_id"]], dtype=np.int64)
    hangul = truth["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    candidate_counts = np.array([len(values) for values in candidate_lists], dtype=np.int64)
    cutoffs = [50, 100, 300, 600, retriever.max_candidates]
    recall_at: dict[str, dict[str, float]] = {}

    for cutoff in sorted(set(cutoffs)):
        hit = np.array([true_index in candidates[:cutoff] for true_index, candidates in zip(true_indices, candidate_lists)], dtype=bool)
        hangul_recall = float(hit[hangul].mean())
        non_hangul_recall = float(hit[~hangul].mean())
        recall_at[str(cutoff)] = {
            "overall": float(hit.mean()),
            "hangul": hangul_recall,
            "non_hangul": non_hangul_recall,
            "weighted": 0.7 * hangul_recall + 0.3 * non_hangul_recall,
            "misses": int((~hit).sum()),
        }

    return {
        "rows": int(len(truth)),
        "recall_at": recall_at,
        "candidate_count": {
            "mean": float(candidate_counts.mean()),
            "median": float(np.median(candidate_counts)),
            "p95": float(np.quantile(candidate_counts, 0.95)),
            "max": int(candidate_counts.max()),
        },
    }


def check_order_invariance(queries: pd.DataFrame, retriever: CandidateRetriever, batch_size: int, seed: int) -> bool:
    sample = queries.head(min(200, len(queries))).copy()
    original_lists = retriever.retrieve_many(sample, batch_size)
    original = {sample.iloc[index]["sample_id"]: retriever.ids[candidates[0]] for index, candidates in enumerate(original_lists)}
    shuffled = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled_lists = retriever.retrieve_many(shuffled, batch_size)
    reordered = {shuffled.iloc[index]["sample_id"]: retriever.ids[candidates[0]] for index, candidates in enumerate(shuffled_lists)}
    return original == reordered


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    assignment_path = args.assignment_path.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    catalog_path = data_dir / "vessel_catalog.csv"
    for path in [train_path, catalog_path, assignment_path]:
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    assignments = pd.read_csv(assignment_path, encoding="utf-8-sig")
    check_columns(train, TRAIN_COLUMNS, "train.csv")
    check_columns(catalog, CATALOG_COLUMNS, "vessel_catalog.csv")
    if not {"sample_id", "unseen_split"}.issubset(assignments.columns):
        raise ValueError("validation_assignments.csv에 sample_id와 unseen_split이 필요합니다.")

    marked = train.merge(assignments[["sample_id", "unseen_split"]], on="sample_id", how="left", validate="one_to_one")
    if marked["unseen_split"].isna().any():
        raise ValueError("일부 train 행에 unseen_split 배정이 없습니다. 02_validation.py를 다시 실행하세요.")
    fit_train = marked[marked["unseen_split"] == "fit"].drop(columns="unseen_split").reset_index(drop=True)
    valid = marked[marked["unseen_split"] == "valid"].drop(columns="unseen_split").reset_index(drop=True)
    if set(fit_train["target_vessel_id"]) & set(valid["target_vessel_id"]):
        raise AssertionError("미관측 검증에서 target_vessel_id 누수가 발견됐습니다.")

    started = time.perf_counter()
    retriever = CandidateRetriever(
        catalog=catalog,
        fit_train=fit_train,
        name_candidates=args.name_candidates,
        metadata_pairs=args.metadata_pairs,
        per_pair_tonnage=args.per_pair_tonnage,
        global_tonnage=args.global_tonnage,
        max_candidates=args.max_candidates,
    )
    index_seconds = time.perf_counter() - started
    retrieval_started = time.perf_counter()
    candidate_lists = retriever.retrieve_many(valid, args.batch_size)
    retrieval_seconds = time.perf_counter() - retrieval_started

    metrics = recall_metrics(valid, candidate_lists, retriever)
    order_invariant = check_order_invariance(valid, retriever, args.batch_size, args.seed)
    final_key = str(args.max_candidates)
    final_recall = metrics["recall_at"][final_key]

    true_indices = [retriever.id_to_index[str(value)] for value in valid["target_vessel_id"]]
    missed_mask = [true_index not in candidates for true_index, candidates in zip(true_indices, candidate_lists)]
    missed_columns = [
        "sample_id",
        "vessel_name",
        "gross_tonnage",
        "international_gross_tonnage",
        "flag_name",
        "vessel_type",
        "target_vessel_id",
    ]
    missed = valid.loc[missed_mask, missed_columns].copy()
    missed.insert(1, "is_hangul", missed["vessel_name"].map(is_hangul_name))
    missed.to_csv(result_dir / "missed_candidate_rows.csv", index=False, encoding="utf-8-sig")

    candidate_counts = pd.Series([len(values) for values in candidate_lists], name="candidate_count")
    candidate_counts.value_counts().sort_index().rename_axis("candidate_count").reset_index(name="row_count").to_csv(
        result_dir / "candidate_count_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "validation": "target_vessel_id-disjoint unseen-vessel split from 02_validation.py",
        "settings": {
            "name_candidates": args.name_candidates,
            "metadata_pairs": args.metadata_pairs,
            "per_pair_tonnage": args.per_pair_tonnage,
            "global_tonnage": args.global_tonnage,
            "max_candidates": args.max_candidates,
            "batch_size": args.batch_size,
        },
        "fit_rows": int(len(fit_train)),
        "valid_rows": int(len(valid)),
        "target_overlap": 0,
        "metrics": metrics,
        "order_invariant": order_invariant,
        "timing_seconds": {
            "index_build": index_seconds,
            "valid_retrieval": retrieval_seconds,
            "total_before_reports": index_seconds + retrieval_seconds,
        },
    }
    with (result_dir / "candidate_retrieval_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("미관측 선박 후보 검색 검증 완료")
    print(f"검증 행: {len(valid):,} / catalog 선박: {len(catalog):,}")
    print(f"Candidate Recall: {final_recall['overall']:.4f}")
    print(f"한글 Recall: {final_recall['hangul']:.4f}")
    print(f"비한글 Recall: {final_recall['non_hangul']:.4f}")
    print(f"가중 Recall: {final_recall['weighted']:.4f}")
    print(f"평균 후보 수: {metrics['candidate_count']['mean']:.1f}")
    print(f"누락 행: {final_recall['misses']:,}")
    print(f"행 순서 독립성: {order_invariant}")
    print(f"인덱스 생성: {index_seconds:.2f}초 / 검증 검색: {retrieval_seconds:.2f}초")
    print(f"결과 저장 위치: {result_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
