from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from usearch.index import Index, Matches, MetricKind
import numpy as np
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

data = [
    """
The Importance of Proper Training for Dogs
Effective training is essential for the well-being of both dogs and their owners. Properly trained dogs are more obedient, better-behaved, and less likely to engage in destructive or dangerous behaviors, making for a more harmonious and enjoyable relationship.
""",
    """
The Diversity of Dog Breeds
With over 300 recognized dog breeds worldwide, the diversity of canine companions is truly astounding. From the tiny Chihuahua to the massive Great Dane, each breed has its own unique characteristics, temperament, and physical attributes, catering to a wide range of preferences and lifestyles.
""",
    """
The Therapeutic Benefits of Dogs
Dogs have long been recognized for their ability to provide emotional and therapeutic support to humans. From service dogs assisting individuals with disabilities to therapy dogs visiting hospitals and nursing homes, these furry companions can have a profound impact on the well-being of those they interact with.
""",
    """
The Purr-fect Companions: Exploring the Unique Personalities of Cats
Cats have long been beloved companions, captivating us with their independent spirits, playful antics, and soothing purrs. From the regal Persian to the mischievous tabby, each feline friend brings a distinct personality to the household. Delve into the fascinating world of cat behavior and discover the nuances that make these furry companions so endearing.
    """,
    """
The Curious Case of Catnip: Unraveling the Allure of this Feline Favorite
Catnip has long been a source of fascination for cat owners, as it elicits a unique response in felines. Discover the science behind this plant's ability to captivate our furry friends and learn how to incorporate catnip into your cat's enrichment routine.
    """,
    """
Feline Grooming: Keeping Your Cat's Coat Healthy and Shiny
Maintaining a cat's coat can be a delicate balance, requiring the right grooming techniques and products. Explore the importance of regular brushing, bathing, and nail trimming, and learn how to address common grooming challenges, such as hairballs and matted fur.
    """,
    """
The Purr-fect Nap: Understanding the Sleeping Habits of Cats
Cats are renowned for their love of sleep, often spending up to 16 hours a day in a state of slumber. Delve into the science behind feline sleep patterns, and discover how you can create the perfect environment for your cat to indulge in their favorite pastime.
    """,
    """
Feline Enrichment: Keeping Your Cat Mentally Stimulated
Cats are intelligent creatures that require mental stimulation to thrive. Discover the importance of providing your cat with enrichment activities, such as puzzle feeders, interactive toys, and environmental changes, to keep them engaged and content.
    """,
    """
The Importance of Responsible Dog Ownership
Owning a dog comes with a significant responsibility to provide proper care, training, and a safe environment. Responsible dog ownership not only ensures the well-being of the animal but also contributes to the overall safety and harmony of the community.
""",
    """
The Enduring Bond Between Humans and Dogs
The relationship between humans and dogs is one of the most enduring and profound in the animal kingdom. This deep connection, forged over centuries of coexistence, has led to the dog being referred to as "man's best friend" – a testament to the unwavering loyalty and companionship these animals provide.
""",
    """
The Importance of Adopting Rescue Dogs
Adopting a dog from a shelter or rescue organization is a compassionate and impactful way to provide a loving home for an animal in need. By choosing to adopt, individuals not only save a life but also contribute to the reduction of pet overpopulation and the euthanasia of healthy, adoptable animals.
""",
]
key_ids = np.arange(len(data))

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# https://www.mixedbread.ai/blog/binary-mrl
# the mixed bread model can be cut to 512 from 1024 and still roughly the same performance 94-95%.
embeds = model.encode(
    data, normalize_embeddings=True, batch_size=128, show_progress_bar=True
)[..., :512]

# decrease to 512
binary_embeds = quantize_embeddings(embeds, "binary")

binary_index = Index(
    ndim=64,
    metric=MetricKind.Hamming,
    dtype="i8",
)
binary_index.add(key_ids, binary_embeds)
binary_index.save("binary_index.usearch")

int8_embeds = quantize_embeddings(embeds, "int8")
int8_index = Index(
    ndim=512,
    # the metric here doesn't matter since we're just using this as a disk store.
    metric=MetricKind.IP,
    dtype="i8",
)
int8_index.add(key_ids, int8_embeds)
int8_index.save("int8_index.usearch")
del int8_index
int8_view = Index.restore("int8_index.usearch", view=True)


def search(query: str, top_k: int = 3):
    start_time = time.time()

    # Encode the query
    encode_start = time.time()
    e = model.encode(query)[..., :512]
    encode_end = time.time()
    logging.info(f"Encoding time: {encode_end - encode_start:.4f} seconds")

    # Quantize embeddings
    quantize_start = time.time()
    be = quantize_embeddings(e.reshape(1, -1), "binary")
    quantize_end = time.time()
    logging.info(f"Quantization time: {quantize_end - quantize_start:.4f} seconds")

    # Search in binary index
    search_start = time.time()
    binary_matches: Matches = binary_index.search(be, top_k)
    search_end = time.time()
    logging.info(f"Binary index search time: {search_end - search_start:.4f} seconds")

    # Retrieve embeddings from int8 index
    retrieve_start = time.time()
    embeds = int8_view[binary_matches.keys].astype(np.float32)
    retrieve_end = time.time()
    logging.info(
        f"Embedding retrieval time: {retrieve_end - retrieve_start:.4f} seconds"
    )

    # Compute final scores
    score_start = time.time()
    scores = e @ embeds.T
    inds = np.argsort(-scores)[:top_k]
    top_k_indices = binary_matches.keys[inds]
    top_k_scores = scores[inds]
    score_end = time.time()
    logging.info(f"Scoring time: {score_end - score_start:.4f} seconds")

    total_time = time.time() - start_time
    logging.info(f"Total search time: {total_time:.4f} seconds")

    print(f"top k scores {top_k_scores}")
    print(f"top k indices {top_k_indices}")
    return {
        "data": [data[i] for i in top_k_indices],
        "scores": top_k_scores,
        "indices": top_k_indices,
    }


search("what are cats sleeping habits", top_k=3)
