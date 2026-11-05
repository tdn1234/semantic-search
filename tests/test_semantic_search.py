import numpy as np
import pytest

from semantic_search import SemanticSearch, TextStrategy


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []

        for text in texts:
            value = text.lower()

            if "running" in value or "road" in value:
                vector = np.array([1.0, 0.0, 0.0])
            elif "laptop" in value or "computer" in value:
                vector = np.array([0.0, 1.0, 0.0])
            else:
                vector = np.array([0.0, 0.0, 1.0])

            vectors.append(vector)

        return np.vstack(vectors).astype(np.float32)


@pytest.fixture
def products():
    return [
        {
            "sku": "SHOE-001",
            "name": "Road Running Shoes",
            "description": "Lightweight footwear for daily road running",
            "category": "Shoes > Running",
            "brand": "Velocity",
            "color": "Black",
            "price": 89.99,
            "in_stock": True,
        },
        {
            "sku": "BAG-001",
            "name": "Office Backpack",
            "description": "Backpack with padded laptop compartment",
            "category": "Bags > Laptop Bags",
            "brand": "UrbanCarry",
            "color": "Grey",
            "price": 69.99,
            "in_stock": True,
        },
    ]


def test_index_and_search_returns_expected_product(products):
    engine = SemanticSearch(
        strategy=TextStrategy.FULL,
        model=FakeEmbeddingModel(),
    )

    engine.index(products)
    results = engine.search("running on the road", top_k=1)

    assert len(results) == 1
    assert results[0]["product"]["sku"] == "SHOE-001"
    assert results[0]["rank"] == 1


def test_embedding_matrix_has_one_row_per_product(products):
    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    engine.index(products)

    assert engine.embedding_matrix.shape == (2, 3)


def test_search_before_index_fails():
    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    with pytest.raises(RuntimeError, match="index"):
        engine.search("running shoes")


def test_empty_query_fails(products):
    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    engine.index(products)

    with pytest.raises(ValueError, match="must not be empty"):
        engine.search("   ")


def test_invalid_top_k_fails(products):
    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    engine.index(products)

    with pytest.raises(ValueError, match="greater than zero"):
        engine.search("running shoes", top_k=0)


def test_missing_product_field_fails(products):
    invalid_products = [dict(products[0])]
    del invalid_products[0]["brand"]

    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    with pytest.raises(ValueError, match="brand"):
        engine.index(invalid_products)


def test_duplicate_sku_fails(products):
    duplicate_products = [
        products[0],
        dict(products[0]),
    ]

    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    with pytest.raises(ValueError, match="Duplicate SKU"):
        engine.index(duplicate_products)


def test_top_k_is_clamped_to_catalog_size(products):
    engine = SemanticSearch(
        model=FakeEmbeddingModel(),
    )

    engine.index(products)
    results = engine.search("running shoes", top_k=10)

    assert len(results) == 2


def test_build_name_only_text(products):
    engine = SemanticSearch(
        strategy=TextStrategy.NAME,
        model=FakeEmbeddingModel(),
    )

    text = engine.build_searchable_text(products[0])

    assert text == "Road Running Shoes"


def test_build_full_text_contains_expected_fields(products):
    engine = SemanticSearch(
        strategy=TextStrategy.FULL,
        model=FakeEmbeddingModel(),
    )

    text = engine.build_searchable_text(products[0])

    assert "Road Running Shoes" in text
    assert "Shoes > Running" in text
    assert "Velocity" in text
    assert "Lightweight footwear" in text