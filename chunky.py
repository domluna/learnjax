import spacy
from datasets import load_dataset
import tqdm
import os
import logging
from multiprocessing import Pool
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")


def split_text_into_chunks(text, chunk_size=1024):
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        if len(current_chunk) + len(sentence_text) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence_text
        else:
            current_chunk += " " + sentence_text
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def process_article(article):
    return split_text_into_chunks(article, chunk_size=1024)


def main():
    dataset = load_dataset("scientific_papers", "pubmed")
    articles = dataset["validation"]["article"][1000:2000]
    output_file = "chunks_output"

    with Pool(processes=4) as pool:
        # Use imap_unordered for potentially better performance in an unordered task
        result_iterator = pool.imap_unordered(process_article, articles)
        all_chunks = list(tqdm.tqdm(result_iterator, total=len(articles)))

    all_chunks = [chunk for article_chunks in all_chunks for chunk in article_chunks]
    pickle.dump(all_chunks, open(output_file, "wb"))
    logging.info(f"Saved {len(all_chunks)} chunks to '{output_file}'.")


if __name__ == "__main__":
    main()
