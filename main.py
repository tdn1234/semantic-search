from sentence_transformers import SentenceTransformer, util
import numpy as np

def main():
    print("Starting the product similarity search...")
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    products_text = [
        f"{product['name']}"
        for product in products
    ]

    
    product_embeddings = model.encode(products_text, normalize_embeddings=True)
    
    query = "running shoes for wet conditions"
    query_embedding = model.encode(query, normalize_embeddings=True)
    result = []
    for product, embedding in zip(products, product_embeddings):
        score = cosine_similarity(embedding, query_embedding)
        result.append({"product": product, "score": score})

    result.sort(key=lambda item:item["score"], reverse=True)
    for res in result:
        product = res["product"]
        print (
           f"{res['score']:.4f} | "
           f"{product['sku']} | "
           f"{product['name']} "     
        )
    # print(result)


def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))   

products = [
    {
        "sku": "RUN-001",
        "name": "Trail Runner GTX",
        "description": "Waterproof trail running shoes for wet and muddy conditions."
    },
    {
        "sku": "RUN-002",
        "name": "Road Runner Pro",
        "description": "Lightweight running shoes designed for long distance road running."
    },
    {
        "sku": "HIKE-001",
        "name": "Mountain Trek",
        "description": "Waterproof hiking boots with strong ankle support."
    },
    {
        "sku": "FORMAL-001",
        "name": "Classic Oxford",
        "description": "Black leather formal shoes for business and office wear."
    },
    {
        "sku": "CASUAL-001",
        "name": "City Walker",
        "description": "Comfortable casual shoes for everyday walking."
    },
    {
        "sku": "RUN-003",
        "name": "Storm Runner",
        "description": "Water-resistant running shoes with excellent grip for rainy weather."
    },
    {
        "sku": "SANDAL-001",
        "name": "Summer Comfort",
        "description": "Lightweight breathable sandals for hot summer weather."
    },
    {
        "sku": "HIKE-002",
        "name": "Light Hiker",
        "description": "Lightweight outdoor shoes suitable for hiking and travelling."
    },
    {
        "sku": "RUN-004",
        "name": "Speed Racer",
        "description": "Responsive road running shoes designed for racing and speed."
    },
    {
        "sku": "CASUAL-002",
        "name": "Leather Sneaker",
        "description": "Premium leather sneakers for casual everyday use."
    }
]


if __name__ == "__main__":
    main()