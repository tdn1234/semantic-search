from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from semantic_search import SemanticSearch, TextStrategy


DEFAULT_K_VALUES = (10, 5, 3, 1)


def load_evaluation(path: str | Path) -> list[dict[str, Any]]:
    evaluation_path = Path(path)

    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {evaluation_path}"
        )

    with evaluation_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("evaluation.json must contain a JSON array.")

    print(f"Loaded cases {cases} evaluation cases from {evaluation_path}.")
    # return
    if len(cases) < 20:
        raise ValueError(
            "The golden dataset must contain at least 20 queries."
        )

    for position, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"Evaluation case at index {position} must be an object."
            )

        query = case.get("query")
        relevant_skus = case.get("relevant_skus")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Evaluation case at index {position} has no valid query."
            )

        if not isinstance(relevant_skus, list) or not relevant_skus:
            raise ValueError(
                f"Evaluation case '{query}' must have relevant SKUs."
            )

        if not all(
            isinstance(sku, str) and sku.strip()
            for sku in relevant_skus
        ):
            raise ValueError(
                f"Evaluation case '{query}' contains an invalid SKU."
            )

    return cases


def recall_at_k(
    retrieved_skus: Iterable[str],
    relevant_skus: Iterable[str],
    k: int,
) -> float:
    relevant = set(relevant_skus)

    if not relevant:
        return 0.0

    retrieved_top_k = set(list(retrieved_skus)[:k])

    return len(retrieved_top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_skus: Iterable[str],
    relevant_skus: Iterable[str],
    k: int,
) -> float:
    relevant = set(relevant_skus)
    retrieved_top_k = set(list(retrieved_skus)[:k])

    return len(retrieved_top_k & relevant) / k


def validate_golden_skus(
    products: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    available_skus = {product["sku"] for product in products}

    unknown_skus = sorted(
        {
            sku
            for case in cases
            for sku in case["relevant_skus"]
            if sku not in available_skus
        }
    )

    if unknown_skus:
        raise ValueError(
            "Golden dataset references unknown SKUs: "
            + ", ".join(unknown_skus)
        )


def evaluate_strategy(
    products: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    strategy: TextStrategy,
    model_name: str,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    engine = SemanticSearch(
        model_name=model_name,
        strategy=strategy,
    )

    engine.index(products)

    maximum_k = max(k_values)
    per_query: list[dict[str, Any]] = []
    query_latencies_ms: list[float] = []

    metric_values: dict[str, dict[int, list[float]]] = {
        "recall": {k: [] for k in k_values},
        "precision": {k: [] for k in k_values},
    }

    for case in cases:
        started_at = time.perf_counter()

        results = engine.search(
            case["query"],
            top_k=maximum_k,
        )

        latency_ms = (
            time.perf_counter() - started_at
        ) * 1000

        query_latencies_ms.append(latency_ms)

        retrieved_skus = [
            result["product"]["sku"]
            for result in results
        ]

        query_metrics: dict[str, float] = {}

        for k in k_values:
            recall = recall_at_k(
                retrieved_skus,
                case["relevant_skus"],
                k,
            )

            precision = precision_at_k(
                retrieved_skus,
                case["relevant_skus"],
                k,
            )

            metric_values["recall"][k].append(recall)
            metric_values["precision"][k].append(precision)

            query_metrics[f"recall@{k}"] = recall
            query_metrics[f"precision@{k}"] = precision

        per_query.append(
            {
                "query": case["query"],
                "relevant_skus": case["relevant_skus"],
                "retrieved_skus": retrieved_skus,
                "latency_ms": round(latency_ms, 3),
                "metrics": query_metrics,
            }
        )

    aggregate_metrics: dict[str, float] = {}

    for k in k_values:
        aggregate_metrics[f"recall@{k}"] = statistics.mean(
            metric_values["recall"][k]
        )

        aggregate_metrics[f"precision@{k}"] = statistics.mean(
            metric_values["precision"][k]
        )

    return {
        "strategy": strategy.value,
        "product_count": len(products),
        "evaluation_query_count": len(cases),
        "embedding_dimensions": int(
            engine.embedding_matrix.shape[1]
        ),
        "index_latency_ms": round(
            engine.index_latency_ms or 0.0,
            3,
        ),
        "mean_query_latency_ms": round(
            statistics.mean(query_latencies_ms),
            3,
        ),
        "p95_query_latency_ms": round(
            percentile(query_latencies_ms, 95),
            3,
        ),
        "metrics": aggregate_metrics,
        "inspection": engine.inspect_product(0),
        "queries": per_query,
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        percentile_value / 100
    ) * (len(ordered) - 1)

    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index

    return (
        ordered[lower_index] * (1 - weight)
        + ordered[upper_index] * weight
    )


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def describe_best(
    results: list[dict[str, Any]],
    metric_key: str,
    label: str,
) -> str:
    """Describe the best strategy for a metric, reporting ties honestly."""
    best_value = max(result["metrics"][metric_key] for result in results)

    winners = [
        result["strategy"]
        for result in results
        if math.isclose(
            result["metrics"][metric_key],
            best_value,
            abs_tol=1e-9,
        )
    ]

    if len(winners) == len(results) and len(results) > 1:
        return (
            f"- {label}: all strategies tie at "
            f"{format_percentage(best_value)}."
        )

    names = ", ".join(f"`{name}`" for name in winners)
    suffix = " (tie)" if len(winners) > 1 else ""

    return (
        f"- Best {label}: {names}{suffix} with "
        f"{format_percentage(best_value)}."
    )


def build_markdown_report(
    results: list[dict[str, Any]],
    model_name: str,
) -> str:
    lines = [
        "# Day 2 Findings",
        "",
        f"- Model: `{model_name}`",
        f"- Product count: {results[0]['product_count']}",
        (
            "- Evaluation query count: "
            f"{results[0]['evaluation_query_count']}"
        ),
        (
            "- Embedding dimensions: "
            f"{results[0]['embedding_dimensions']}"
        ),
        "",
        "## Strategy comparison",
        "",
        (
            "| Strategy | Recall@1 | Recall@3 | Recall@5 | "
            "Recall@10 | Precision@1 | Precision@3 | "
            "Precision@5 | Precision@10 | Index ms | "
            "Mean query ms | P95 query ms |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|"
        ),
    ]

    for result in results:
        metrics = result["metrics"]

        lines.append(
            "| "
            + " | ".join(
                [
                    result["strategy"],
                    format_percentage(metrics["recall@1"]),
                    format_percentage(metrics["recall@3"]),
                    format_percentage(metrics["recall@5"]),
                    format_percentage(metrics["recall@10"]),
                    format_percentage(metrics["precision@1"]),
                    format_percentage(metrics["precision@3"]),
                    format_percentage(metrics["precision@5"]),
                    format_percentage(metrics["precision@10"]),
                    f"{result['index_latency_ms']:.3f}",
                    f"{result['mean_query_latency_ms']:.3f}",
                    f"{result['p95_query_latency_ms']:.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Concrete observations",
            "",
            describe_best(results, "recall@5", "Recall@5"),
            describe_best(results, "precision@3", "Precision@3"),
            describe_best(results, "recall@1", "Recall@1"),
            describe_best(results, "precision@1", "Precision@1"),
            (
                "- Name-only search has less descriptive information and "
                "may perform poorly for use-case queries."
            ),
            (
                "- Adding descriptions usually helps synonym and use-case "
                "queries, but noisy descriptions may reduce precision."
            ),
            (
                "- Category and brand can improve category or brand intent, "
                "but repeated catalog boilerplate may dominate embeddings."
            ),
            "",
            "## Intermediate representation inspected",
            "",
        ]
    )

    for result in results:
        inspection = result["inspection"]

        lines.extend(
            [
                f"### {result['strategy']}",
                "",
                (
                    f"- SKU: "
                    f"`{inspection['product']['sku']}`"
                ),
                (
                    f"- Searchable text: "
                    f"`{inspection['searchable_text']}`"
                ),
                (
                    f"- Vector shape: "
                    f"`{inspection['vector_shape']}`"
                ),
                (
                    "- First 10 vector values: "
                    f"`{inspection['first_10_vector_values']}`"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Why recall matters before reranking",
            "",
            (
                "A reranker can only reorder candidates returned by the "
                "retrieval stage. If the relevant product is absent from "
                "Top K, the reranker cannot recover it."
            ),
            "",
            (
                "For example, if the correct SKU is ranked 18th but the "
                "reranker receives only the first 10 candidates, the maximum "
                "possible final relevance for that query is already limited."
            ),
            "",
            (
                "Therefore, first-stage retrieval should normally prioritize "
                "high Recall@K. The reranker can then improve the ordering and "
                "final Precision@K."
            ),
            "",
            "## Reproducibility",
            "",
            "- Run: `python evaluate.py`",
            "- Run tests: `pytest -q`",
            "- Input catalog: `data/products.json`",
            "- Golden dataset: `data/evaluation.json`",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic product-search strategies."
    )

    parser.add_argument(
        "--products",
        default="data/products.json",
        help="Path to the product catalog.",
    )

    parser.add_argument(
        "--evaluation",
        default="data/evaluation.json",
        help="Path to the golden evaluation dataset.",
    )

    parser.add_argument(
        "--report",
        default="reports/day-2-findings.md",
        help="Markdown output file.",
    )

    parser.add_argument(
        "--json-output",
        default="reports/day-2-results.json",
        help="Detailed JSON output file.",
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()


    products = SemanticSearch.load_products(args.products)
    cases = load_evaluation(args.evaluation)

    # print(f"Loaded {len(products)} products from {products}.")
    print(f"Loaded {len(cases)} evaluation cases from {cases}.")
    # return

    # Validate the product schema before checking golden SKUs.
    validation_engine = SemanticSearch(
        model_name=args.model,
        strategy=TextStrategy.NAME,
    )
    validation_engine.validate_products(products)

    validate_golden_skus(products, cases)

    results = []

    for strategy in TextStrategy:
        print(f"Evaluating strategy: {strategy.value}")

        strategy_result = evaluate_strategy(
            products=products,
            cases=cases,
            strategy=strategy,
            model_name=args.model,
        )

        results.append(strategy_result)

    report_path = Path(args.report)
    json_output_path = Path(args.json_output)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        build_markdown_report(results, args.model),
        encoding="utf-8",
    )

    json_output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Evaluation completed.")
    print(f"Markdown report: {report_path}")
    print(f"Detailed results: {json_output_path}")

    for result in results:
        print()
        print(result["strategy"])

        for metric_name, metric_value in result["metrics"].items():
            print(
                f"  {metric_name}: "
                f"{format_percentage(metric_value)}"
            )


if __name__ == "__main__":
    main()