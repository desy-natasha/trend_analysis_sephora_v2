import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunking(text, model, percentile=25, min_sentences=2):
    """Split a single review into semantically coherent chunks."""
    sentences = sent_tokenize(text)

    # too short to bother splitting
    if len(sentences) <= min_sentences:
        return [text], [0]

    embeddings = model.encode(sentences)

    chunks        = []
    boundaries    = []
    chunk_start   = 0
    chunk_embeds  = [embeddings[0]]

    for i in range(1, len(embeddings)):
        # centroid of current chunk
        centroid = np.mean(chunk_embeds, axis=0)
        sim = cosine_similarity(centroid, embeddings[i])

        # calculate similarity between current sentence and chunk centroid, compare to threshold
        if sim < np.percentile(
            [cosine_similarity(np.mean(embeddings[max(0,j-3):j], axis=0), embeddings[j])
             for j in range(1, len(embeddings))],
            percentile
        ):
            # split if current chunk is large enough
            if (i - chunk_start) >= min_sentences:
                chunks.append(" ".join(sentences[chunk_start:i]))
                boundaries.append(chunk_start)
                chunk_start  = i
                chunk_embeds = [embeddings[i]]
                continue

        chunk_embeds.append(embeddings[i])

    # last chunk
    chunks.append(" ".join(sentences[chunk_start:]))
    boundaries.append(chunk_start)

    return (chunks if chunks else [text]), (boundaries if boundaries else [0])

def apply_boundaries_to_text(text, boundaries):
    """Apply sentence-level boundaries to two text columns to achieve identical chunk splits. """
    
    sentences = sent_tokenize(text)

    # derive end indices from boundaries
    chunks = []
    split_pairs = list(zip(boundaries, boundaries[1:] + [len(sentences)]))

    for start, end in split_pairs:
        # parallel text may have fewer sentences due to emoji expansion
        # ensure the boundaries don't exceed the actual sentence count to avoid index errors
        start = min(start, len(sentences))
        end   = min(end,   len(sentences))
        chunk = sentences[start:end]
        chunks.append(" ".join(chunk) if chunk else "")

    return chunks if chunks else [text]

def clean_chunk(chunk):
    """Remove leading punctuation from a chunk."""
    return re.sub(r'^[!?.,;:/\s]+', '', chunk).strip()

def is_valid_chunk(chunk):
    """Reject chunks that are too short to be meaningful."""
    return len(chunk.strip()) >= 5

def chunk_reviews(reviews_df, model, text_col="cleaned_review_text",
                  parallel_col="cleaned_demojize_review_text",
                  percentile=25, min_sentences=2):
    """Apply semantic chunking across all reviews."""
    records       = []
    mismatch_log  = []

    for idx, row in reviews_df.iterrows():
        primary_text  = row[text_col]
        parallel_text = row[parallel_col]

        n_primary  = len(sent_tokenize(primary_text))
        n_parallel = len(sent_tokenize(parallel_text))
        if n_primary != n_parallel:
            mismatch_log.append({
                "review_id" : idx,
                "n_primary" : n_primary,
                "n_parallel": n_parallel
            })

        chunks, boundaries = semantic_chunking(
            text      = primary_text,
            model     = model,
            percentile    = percentile,
            min_sentences = min_sentences
        )

        parallel_chunks = apply_boundaries_to_text(parallel_text, boundaries)

        n = min(len(chunks), len(parallel_chunks))

        valid_pairs = []
        for i in range(n):
            chunk          = clean_chunk(chunks[i])
            parallel_chunk = clean_chunk(parallel_chunks[i])
            if is_valid_chunk(chunk):
                valid_pairs.append((chunk, parallel_chunk))

        # n_chunks now reflects actual valid chunks, not raw count
        n_valid = len(valid_pairs)

        for chunk_pos, (chunk, parallel_chunk) in enumerate(valid_pairs):
            chunk = clean_chunk(chunks[chunk_pos])
            if not is_valid_chunk(chunk): 
                continue
            
            records.append({
                "review_id"             : row.get("review_id"),
                "chunk_pos"             : chunk_pos,   # position within the review
                "n_chunks"              : n_valid,     # total chunks this review was split into
                "review_chunk"          : chunk,
                "review_chunk_demojize" : parallel_chunk,
                
                "product_id"            : row.get("product_id"),
                "rating"                : row.get("rating"),
                "title"                 : row.get("cleaned_review_title"),
                "timestamp"             : row.get("timestamp"),
                "nickname"              : row.get("reviewer_nickname")
            })

    if mismatch_log:
        mismatch_df = pd.DataFrame(mismatch_log)
        mismatch_rate = len(mismatch_df) / len(reviews_df) * 100
        print(f"Sentence count mismatches : {len(mismatch_df)} / {len(reviews_df)} "
              f"({mismatch_rate:.1f}%)")
    else:
        print("No sentence count mismatches detected.")

    chunks_df = pd.DataFrame(records)

    return chunks_df