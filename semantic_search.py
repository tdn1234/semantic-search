from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sentence_transformers import SentenceTransformer

class TextStrategy(Enum):
    NAME = "name"
    FULL = "full"
    NAME_DESCRIPTION = "name_and_description"

class SemanticSearch:
    """
    Reusable in-memory semantic search engine.

    Offline/indexing stage:
        products -> searchable text -> embeddings -> NumPy matrix

    Online/query stage:
        query -> query embedding -> matrix-vector dot product -> Top K
    """

    REQUIRED_FIELDS = {
        "sku",
        "name",
        "description",
        "category",
        "brand",
        "color",
        "price",
        "in_stock",
    }

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        strategy: TextStrategy | str = TextStrategy.FULL,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.strategy = TextStrategy(strategy)
        self.model = model or SentenceTransformer(model_name)

        self.products: list[dict[str, Any]] = []
        self.searchable_texts: list[str] = []
        self.embedding_matrix: np.ndarray | None = None
        self.index_latency_ms: float | None = None

    @staticmethod
    def load_products(path: str | Path) -> list[dict[str, Any]]:
        product_path = Path(path)

        if not product_path.exists():
            raise FileNotFoundError(f"Product file not found: {product_path}")

        with product_path.open("r", encoding="utf-8") as file:
            products = json.load(file)

        if not isinstance(products, list):
            raise ValueError("products.json must contain a JSON array.")

        return products

    def validate_products(
        self,
        products: Sequence[dict[str, Any]],
    ) -> None:
        if not products:
            raise ValueError("At least one product is required.")

        seen_skus: set[str] = set()

        for position, product in enumerate(products):
            if not isinstance(product, dict):
                raise ValueError(
                    f"Product at index {position} must be a JSON object."
                )

            missing_fields = self.REQUIRED_FIELDS - product.keys()

            if missing_fields:
                fields = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Product at index {position} is missing fields: {fields}"
                )

            sku = product["sku"]

            if not isinstance(sku, str) or not sku.strip():
                raise ValueError(
                    f"Product at index {position} has an invalid SKU."
                )

            if sku in seen_skus:
                raise ValueError(f"Duplicate SKU found: {sku}")

            seen_skus.add(sku)

            if not isinstance(product["price"], (int, float)):
                raise ValueError(f"Product {sku} has an invalid price.")

            if product["price"] < 0:
                raise ValueError(f"Product {sku} has a negative price.")

            if not isinstance(product["in_stock"], bool):
                raise ValueError(
                    f"Product {sku} must have a Boolean in_stock value."
                )

    def build_searchable_text(self, product: dict[str, Any]) -> str:
        """
        Converts one structured product into text for the embedding model.
        """

        if self.strategy == TextStrategy.NAME:
            return product["name"].strip()

        if self.strategy == TextStrategy.NAME_DESCRIPTION:
            return (
                f"Name: {product['name']}. "
                f"Description: {product['description']}."
            ).strip()

        if self.strategy == TextStrategy.FULL:
            return (
                f"Name: {product['name']}. "
                f"Category: {product['category']}. "
                f"Brand: {product['brand']}. "
                f"Description: {product['description']}."
            ).strip()

        raise ValueError(f"Unsupported text strategy: {self.strategy}")

    def index(
        self,
        products: Sequence[dict[str, Any]],
    ) -> None:
        """
        Builds the product embedding matrix.

        normalize_embeddings=True makes every vector have unit length.
        The dot product then gives cosine similarity.
        """

        self.validate_products(products)

        started_at = time.perf_counter()

        self.products = [dict(product) for product in products]

        self.searchable_texts = [
            self.build_searchable_text(product)
            for product in self.products
        ]

        embeddings = self.model.encode(
            self.searchable_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        matrix = np.asarray(embeddings, dtype=np.float32)

        if matrix.ndim != 2:
            raise ValueError(
                "The embedding model must return a two-dimensional matrix."
            )

        if matrix.shape[0] != len(self.products):
            raise ValueError(
                "Embedding count does not match the product count."
            )

        self.embedding_matrix = matrix

        self.index_latency_ms = (
            time.perf_counter() - started_at
        ) * 1000

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if self.embedding_matrix is None:
            raise RuntimeError(
                "The search index has not been built. Call index() first."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        actual_k = min(top_k, len(self.products))

        query_embedding = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_embedding[0],
            dtype=np.float32,
        )

        # Shape:
        #
        # embedding_matrix: (number_of_products, embedding_dimensions)
        # query_vector:     (embedding_dimensions,)
        # scores:           (number_of_products,)
        scores = self.embedding_matrix @ query_vector

        # argpartition avoids sorting every score when only Top K is needed.
        candidate_indices = np.argpartition(
            scores,
            -actual_k,
        )[-actual_k:]

        # Sort the selected candidates by descending similarity.
        sorted_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        results: list[dict[str, Any]] = []

        for rank, product_index in enumerate(sorted_indices, start=1):
            product = self.products[int(product_index)]

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[product_index]),
                    "searchable_text": self.searchable_texts[
                        int(product_index)
                    ],
                    "product": product,
                }
            )

        return results

    def inspect_product(self, index: int = 0) -> dict[str, Any]:
        if self.embedding_matrix is None:
            raise RuntimeError(
                "The search index has not been built. Call index() first."
            )

        if index < 0 or index >= len(self.products):
            raise IndexError("Product index is outside the index.")

        vector = self.embedding_matrix[index]

        return {
            "product": self.products[index],
            "searchable_text": self.searchable_texts[index],
            "vector_shape": list(vector.shape),
            "first_10_vector_values": [
                round(float(value), 6)
                for value in vector[:10]
            ],
        }
