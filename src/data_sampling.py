import pandas as pd
import numpy as np
import re
import anthropic
import asyncio
import nest_asyncio
from collections import defaultdict
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split

### CONSTANTS ###

# Keywords for targeted resampling to easily capture relevant samples
aspects_keywords = {
    'Acne & Blemish Control': [
        'pores', 'acne', 'pimple', 'blackheads', 'breakout', 'breakouts', 'pimples', 'scars', 'zit',
        'break', 'broke', 'break outs', 'breaking',        
        'strips', 'nose', 'pore strips', 'pore', 'nose strip', 'nose strips', 'pore strip',
        'spot', 'spot treatment', 'treatment', 'sulfur spot', 'sulfur smell'
    ], 
    'Anti-Aging & Skin Renewal': [
        'retinol', 'renewal serum', 'texture', 
        'lines', 'fine lines', 'wrinkles', 'forehead', 'lines wrinkles',
        'neck', 'anti aging', 'anti-aging', 'aging', 'ginseng',
        'wrinkle smoothing', 'wrinkle', 'power infusing',
        'tightening', 'tighten', 'tightens', 'skin tightening', 'tightening effect', 'tightening skin'
    ],  
    'Cleansing & Exfoliation': [
        'cleanser', 'cleansing', 'balm', 'cleansing balm', 'wash', 'face wash', 'clean', 'cleanses'
        'remover', 'remove', 'removes',
        'resurfacing', 'resurfacing pads', 'exfoliating pads', 'exfoliating', 'exfoliant', 'scrub', 'exfoliates'
        'wipes', 'wipe', 'makeup wipes', 'removing',
        'cotton', 'cotton pads', 'facial cotton', 'cotton pad'
    ], 
    'Consistency & Texture': [
        'apply', 'easy apply', 'easy use', 'application', 'applying', 'application easy', 
        'lightweight', 'heavy', 'light', 'light weight', 'feel heavy', 'super lightweight', 
        'smell', 'smells', 'scent', 'fragrance',
        'sticky', 'residue', 'glides', 'smooth', 'feel sticky',
        'pilling', 'pilled',
        'gentle', 'gentle skin', 'gentle use', 'gentle sensitive', 
        'thicker',
        'vegan', 'cruelty', 'cruelty free', 'crueltyfree', 'parabens', 'animals', 'vegan cruelty', 'vegan crueltyfree', 'ingredients',
        'formula', 'new formula', 'old formula', 'changed formula', 'original formula'
    ],
    'Hydration & Moisturization': [
        'face mask', 'clay',
        'toner', 'toners','toner pads',
        'cream', 'moisturizer', 'hydrated', 'dry', 'moisture', 'drying'
        'argan', 'argan oil', 'greasy',
        'butter', 'body butter', 'hand cream', 
        'hydrating', 'hydration', 'soft skin', 'smooth skin',
        'moisturizer summer',
        'absorbs', 'absorb', 'absorbed',
        'balm', 'lip balm'
    ],
    'Packaging & Size': [
        'packaging',
        'pump', 'pumps', 'bottle', 'dispenser',
        'container', 'half',
        'mini', 'mini size', 'mini version', 'version',
        'tube', 'tubes', 'squeeze', 'small tube',
        'dropper', 'drop', 'drops', 
        'refillable', 'refill', 'refills', 'recyclable', 'sustainable',
        'travel', 'travel size', 'traveling',
        'purse', 'bag', 'carry'
    ],
    'Skin Sensitivity': [
        'redness', 'sensitive', 'peel', 'sensitive skin', 'eczema', 'red', 'rosacea',
        'psoriasis', 'irritation', 'allergies', 'burn', 'rash'
    ],
    'Skin Tone & Pigmentation': [
        'circles', 'dark circles', 'dark', 'area', 'eye area', 'eye bags'
        'dark spots', 'hyperpigmentation', 'spot', 'dark spot',
        'cellulite', 'stretch', 'stretch marks', 'marks', 'thighs',
         'brighter', 'luminous', 'brighten'
    ],
    'Sun Protection': [
        'sunscreen', 'spf', 'sunscreens', 'sun', 'protection', 'sunshine'
        'leave white', 'leaves white', 'white', 'white cast'
    ], 
    'Wear & Longevity': [
        'reapply', 'wear', 'reapplying', 'hours',
        'long way', 'goes long', 'little goes', 'way little', 'bit goes', 'way jar',
        'last', 'go through', 'wore off', 'small amount', 'comfortable'
    ], 
}

# Negative keywords for targeted sampling
negative_keywords = {
    "Acne & Blemish Control"      : ["broke out", "breakout", "clogged", "pimple", "worse", "didn't help", "irritated"],
    "Anti-Aging & Skin Renewal"   : ["no difference", "didn't work", "no results", "disappointed", "no improvement"],
    "Cleansing & Exfoliation"     : ["didn't remove", "left residue", "didn't clean", "irritating", "harsh"],
    "Consistency & Texture"       : ["sticky", "greasy", "heavy", "pilling", "doesn't absorb", "thick"],
    "Hydration & Moisturization"  : ["dry", "didn't moisturize", "no hydration", "tight", "flaky"],
    "Packaging & Size"            : ["broke", "leaked", "pump broke", "too small", "wasteful", "hard to open"],
    "Skin Sensitivity"            : ["reaction", "rash", "burns", "stings", "irritated", "redness", "broke out"],
    "Skin Tone & Pigmentation"    : ["no difference", "still dark", "didn't fade", "no brightening", "worse"],
    "Sun Protection"              : ["white cast", "greasy", "pilled", "burned", "didn't protect"],
    "Wear & Longevity"            : ["faded", "didn't last", "wore off", "reapply", "short lasting"],
}

# Neutral keywords for targeted sampling
NEUTRAL_KEYWORDS = [
    "okay", "ok", "alright", "decent", "average", "not bad", "so-so",
    "mixed", "sometimes", "kind of", "sort of", "somewhat", "mediocre",
    "could be better", "not great", "nothing special", "does the job",
    "fine", "fair", "acceptable", "adequate", "moderate", "middle",
    "hit or miss", "inconsistent", "works sometimes", "on the fence"
]

# Language for back-translation augmentation
PIVOT_LANGUAGES = ["fr", "de", "es"]
CHAIN_PAIRS = [("fr", "de"), ("de", "es"), ("es", "fr"), ("fr", "es"), ("es", "de"), ("de", "fr")]


### INITIAL SAMPLING FUNCTION ###

def sample_for_manually_labelled_set(df, text_col, n_per_aspect=400, seed=42):
    """Initial sampling for manual labelling with stratification by aspect and year"""

    df = df[(df['year'] >= 2016) & (df['year'] <= 2026)].copy()
    df = df[df[text_col].str.split().str.len() >= 5]
    df = df.drop_duplicates(subset=text_col)

    n_years = 2026 - 2016 + 1
    n_per_year = max(1, n_per_aspect // n_years)

    sampled_parts = []
    for aspect, aspect_group in df.groupby("aspect"):
        aspect_samples = []
        for year, year_group in aspect_group.groupby("year"):
            n = min(n_per_year, len(year_group))
            aspect_samples.append(year_group.sample(n=n, random_state=seed))
        
        aspect_df = pd.concat(aspect_samples)
        if len(aspect_df) > n_per_aspect:
            aspect_df = aspect_df.sample(n=n_per_aspect, random_state=seed)
        sampled_parts.append(aspect_df)

    sampled = pd.concat(sampled_parts).reset_index(drop=True)
    output_path = "absa_labelling_sample.xlsx"
    sampled.to_excel(output_path, index=False)

    print(f"Total samples : {len(sampled)}")
    print(f"\nSamples per aspect per year:")
    print(sampled.groupby(["aspect", "year"]).size().unstack(fill_value=0).to_string())

    return sampled


### HELPER FUNCTIONS FOR RESAMPLING ###

def _remaining_targets(labelled_df, targets_per_label,
                               aspect_col="aspect", label_col="label"):
    """Calculate how many samples are still needed per aspect/label"""

    counts = (labelled_df.groupby([aspect_col, label_col])
              .size()
              .reset_index(name="current"))

    rows = []
    remaining_targets = {}

    for aspect in labelled_df[aspect_col].unique():
        aspect_remaining = {}
        for label, target in targets_per_label.items():
            current = counts.loc[
                (counts[aspect_col] == aspect) & (counts[label_col] == label),
                "current"].sum() 

            remaining = max(0, target - current)
            rows.append({
                "aspect"   : aspect,
                "label"    : label,
                "current"  : int(current),
                "target"   : target,
                "remaining": remaining,
            })
            if remaining > 0:
                aspect_remaining[label] = remaining

        if aspect_remaining:
            remaining_targets[aspect] = aspect_remaining

    return remaining_targets

def _get_pool(df, aspect, text_col, year_col, year_start, year_end,
              exclude_chunks, aspect_col="aspect", low_rating=False, positive_only=False):
    """Function to build a clean pool for any aspect"""

    pool = df[
        (df[aspect_col] == aspect) &
        (df[year_col]   >= year_start) &
        (df[year_col]   <= year_end)
    ].copy()
    pool = pool[pool[text_col].str.split().str.len() >= 5]
    pool = pool.drop_duplicates(subset=text_col)
    if exclude_chunks:
        pool = pool[~pool[text_col].isin(exclude_chunks)]
    if low_rating:
        pool = pool[pool['rating'] <= 2]
    if positive_only:
        pool = pool[pool['rating'] >= 4]

    return pool


def _keyword_filter(pool, text_col, keywords, n_needed):
    """Function to filter pool by keywords. If not enough samples with keywords, return original pool for random sampling"""

    if not keywords:
        return pool
    pattern      = "|".join([re.escape(kw) for kw in keywords])
    keyword_pool = pool[pool[text_col].str.lower().str.contains(pattern, na=False)]
    return keyword_pool if len(keyword_pool) >= n_needed else pool


def _make_sample(pool, n, aspect, label, note, seed):
    """Function to sample n rows and add label columns."""
    n = min(n, len(pool))
    cols_to_keep = ["review_id", "aspect", "year", "main_topic", "rating", "cleaned_review_text", "cleaned_demojize_review_text", "cleaned_review_title"] 
    sampled = pool.sample(n=n, random_state=seed)[cols_to_keep].copy()
    
    sampled["label"]  = label
    sampled["note"]   = note

    return sampled

def _backtranslate(labelled_df, aspect, label, n_needed,
                   text_col, title_col):
    """Back-translation augmentation"""

    source_rows = labelled_df[
        (labelled_df["aspect"] == aspect) &
        (labelled_df["label"]  == label)
    ][[text_col, title_col, "review_id", "year", "main_topic", "rating"]].values.tolist()

    if not source_rows:
        return []

    augmented  = []
    seen_texts = set(row[0] for row in source_rows)

    max_single_hop = len(source_rows) * len(PIVOT_LANGUAGES)
    use_chaining   = n_needed > max_single_hop

    for i in range(n_needed):
        original_text, original_title, original_review_id, original_year, original_topic, original_rating = (
            source_rows[i % len(source_rows)]
        )
        try:
            if not use_chaining:
                pivot      = PIVOT_LANGUAGES[i % len(PIVOT_LANGUAGES)]
                back_text  = GoogleTranslator(source="en", target=pivot).translate(original_text)
                back_text  = GoogleTranslator(source=pivot, target="en").translate(back_text)
                back_title = (GoogleTranslator(source="en", target=pivot).translate(original_title)
                              if original_title else None)
                back_title = (GoogleTranslator(source=pivot, target="en").translate(back_title)
                              if back_title else None)
                note       = f"augmented_bt_{pivot}_from: {original_text[:50]}"
            else:
                lang1, lang2 = CHAIN_PAIRS[i % len(CHAIN_PAIRS)]
                back_text    = GoogleTranslator(source="en",  target=lang1).translate(original_text)
                back_text    = GoogleTranslator(source=lang1, target=lang2).translate(back_text)
                back_text    = GoogleTranslator(source=lang2, target="en").translate(back_text)
                if original_title:
                    back_title = GoogleTranslator(source="en",  target=lang1).translate(original_title)
                    back_title = GoogleTranslator(source=lang1, target=lang2).translate(back_title)
                    back_title = GoogleTranslator(source=lang2, target="en").translate(back_title)
                else:
                    back_title = None
                note = f"augmented_bt_{lang1}_{lang2}_from: {original_text[:50]}"

            if back_text and back_text not in seen_texts:
                seen_texts.add(back_text)
                augmented.append({
                    text_col    : back_text,
                    title_col   : back_title,
                    "review_id" : original_review_id,
                    "year"      : original_year,
                    "main_topic": original_topic,
                    "rating"    : original_rating,
                    "aspect"    : aspect,
                    "label"     : label,
                    "note"      : note,
                })

        except Exception as e:
            print(f"  Translation failed: {e}")

    return augmented

def _paraphrase_with_llm(labelled_df, aspect, label, n_needed,
                                  text_col, title_col):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    source_rows = labelled_df[
        (labelled_df["aspect"] == aspect) &
        (labelled_df["label"]  == label)
    ]

    results = []
    for i in range(n_needed):
        row      = source_rows.iloc[i % len(source_rows)]
        original = row[text_col]

        inputs  = tokenizer(
            f"paraphrase: {original}",
            return_tensors = "pt",
            truncation     = True,
            max_length     = 256,
        )
        outputs = model.generate(
            **inputs,
            max_length        = 256,
            num_beams         = 5,
            num_return_sequences = 1,
            early_stopping    = True,
        )
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            text_col    : paraphrased,
            title_col   : row[title_col],
            "review_id" : row["review_id"],
            "year"      : row["year"],
            "main_topic": row["main_topic"],
            "rating"    : row["rating"],
            "aspect"    : aspect,
            "label"     : label,
            "note"      : f"lllm_paraphrase_from: {original[:50]}",
        })

    return results

### LLM PARAPHRASING ###

# async def _paraphrase_single(client, row, aspect, label, text_col, title_col, idx):
#     """Paraphrase one sample through the Anthropic API"""
    
#     original = row[text_col]
#     prompt   = f"""Paraphrase the following {label} product review about {aspect}.
#     Keep the same sentiment ({label}) and meaning, but use different wording.
#     Return only the paraphrased review, nothing else.
    
#     Original: {original}"""

#     response = await client.messages.create(
#         model      = "claude-sonnet-4-20250514",
#         max_tokens = 300,
#         messages   = [{"role": "user", "content": prompt}],
#     )
#     paraphrased = response.content[0].text.strip()

#     return {
#         text_col    : paraphrased,
#         title_col   : row[title_col],
#         "review_id" : row["review_id"],
#         "year"      : row["year"],
#         "main_topic": row["main_topic"],
#         "rating"    : row["rating"],
#         "aspect"    : aspect,
#         "label"     : label,
#         "note"      : f"llm_paraphrase_from: {original[:50]}",
#     }

# async def _paraphrase_batch_async(labelled_df, aspect, label, n_needed,
#                                    text_col, title_col, batch_size=10):
#     """Run LLM paraphrasing in parallel batches"""

#     source_rows = labelled_df[(labelled_df["aspect"] == aspect) &
#                               (labelled_df["label"]  == label)]

#     if source_rows.empty:
#         return []

#     client  = anthropic.AsyncAnthropic()
#     tasks   = [
#         _paraphrase_single(
#             client,
#             source_rows.iloc[i % len(source_rows)],
#             aspect, label, text_col, title_col, i
#         )
#         for i in range(n_needed)]

#     results = []
#     for i in range(0, len(tasks), batch_size):
#         batch         = tasks[i : i + batch_size]
#         batch_results = await asyncio.gather(*batch, return_exceptions=True)
#         for r in batch_results:
#             if isinstance(r, Exception):
#                 print(f"  Paraphrase failed: {r}")
#             else:
#                 results.append(r)
#         await asyncio.sleep(1) 

#     return results

# def _paraphrase_with_llm(labelled_df, aspect, label, n_needed,
#                           text_col, title_col, batch_size=10):
#     """Sync wrapper around the async paraphrase batch — safe to call from notebooks"""
#     try:
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
            
#             nest_asyncio.apply()
#         return loop.run_until_complete(
#             _paraphrase_batch_async(labelled_df, aspect, label, n_needed,
#                                      text_col, title_col, batch_size)
#         )
#     except RuntimeError:
#         return asyncio.run(
#             _paraphrase_batch_async(labelled_df, aspect, label, n_needed,
#                                      text_col, title_col, batch_size)
#         )


### RESAMPLING FUNCTIONS FOR UNDERREPRESENTED CLASSES ###

def resample_positive(df, labelled_df, aspect, target, text_col, aspect_col,
                        label_col, year_col, year_start, year_end,
                        existing_chunks, aspects_keywords, seed):
    """Resampling for positive class by targeted upsampling"""

    aspect_pos    = labelled_df[
        (labelled_df[aspect_col] == aspect) &
        (labelled_df[label_col]  == "positive")
    ].copy()
    current_count = len(aspect_pos)
    diff          = target - current_count

    if diff == 0:
        return aspect_pos, pd.DataFrame()

    if diff < 0:
        downsampled = aspect_pos.sample(n=target, random_state=seed)
        return downsampled, pd.DataFrame()

    pool = _get_pool(df, aspect, text_col, year_col, year_start, year_end,
                     existing_chunks, aspect_col, positive_only=True)
    pool = _keyword_filter(pool, text_col, aspects_keywords.get(aspect, []), diff)
    sampled = _make_sample(pool, diff, aspect, "positive", "resample_positive", seed)

    return aspect_pos, sampled


def resample_not_mentioned(df, aspect, n_needed, text_col, year_col,
                             year_start, year_end, existing_chunks, seed):
    """Resampling for not_mentioned by randomly sampling from other aspects."""
    
    pool = df[
        (df["aspect"] != aspect) &
        (df["aspect"].notna()) &
        (df[year_col]  >= year_start) &
        (df[year_col]  <= year_end)
    ].copy()
    pool = pool[pool[text_col].str.split().str.len() >= 5]
    pool = pool.drop_duplicates(subset=text_col)
    if existing_chunks:
        pool = pool[~pool[text_col].isin(existing_chunks)]
    sampled = _make_sample(pool, n_needed, aspect, "not_mentioned", f"resample_not_mentioned | aspect: {aspect}", seed)

    return sampled

def resample_with_augmentation(labelled_df, df, aspect, label, n_needed,
                               text_col, year_col, year_start, year_end,
                               existing_chunks, seed,
                               title_col="cleaned_review_title",
                               batch_size=10):
    """
    Resampling for negatives and neutrals with three-stage augmentation pipeline:
      1. Back-translation      
      2. LLM paraphrasing   
      3. Keyword/rating pool   
    """

    # 1. Back-translation
    augmented = _backtranslate(labelled_df, aspect, label, n_needed, text_col, title_col)
    remaining  = n_needed - len(augmented)
    print(f"  [{aspect} | {label}] BT produced {len(augmented)}, need {remaining} more")  # <-- add this
    
    # 2. LLM paraphrasing
    if remaining > 0:
        paraphrased = _paraphrase_with_llm(labelled_df, aspect, label, remaining,
                                           text_col, title_col)
        augmented  += paraphrased
        remaining   = n_needed - len(augmented)
        print(f"  [{aspect} | {label}] LLM paraphrasing produced {len(paraphrased)}, need {remaining} more")

    augment_df = pd.DataFrame(augmented) if augmented else pd.DataFrame()

    # 3. Keyword/rating fallback (manual labelling)
    if remaining <= 0:
        return augment_df, pd.DataFrame()

    pool = _get_pool(df, aspect, text_col, year_col, year_start, year_end, existing_chunks)

    if label == "neutral":
        rated_pool    = pool[pool["rating"] == 3].copy() if "rating" in pool.columns else pool
        base_pool     = rated_pool if len(rated_pool) > 0 else pool
        pattern       = "|".join(re.escape(kw) for kw in NEUTRAL_KEYWORDS)
        keyword_pool  = base_pool[base_pool[text_col].str.lower().str.contains(pattern, na=False)]
        fallback_pool = keyword_pool if len(keyword_pool) >= remaining else base_pool
        note          = "fallback_neutral_rated_kw"

    elif label == "negative":
        pool          = _get_pool(df, aspect, text_col, year_col, year_start, year_end,
                                  existing_chunks, low_rating=True)
        fallback_pool = _keyword_filter(pool, text_col,
                                        negative_keywords.get(aspect, []), remaining)
        note          = "fallback_negative_kw"
    else:
        fallback_pool = pool
        note          = f"fallback_{label}"

    fallback_df = _make_sample(fallback_pool, remaining, aspect, label, note, seed)
    print(f"  [{aspect} | {label}] Fallback pool added {len(fallback_df)} for manual labelling")

    return augment_df, fallback_df


### MAIN RESAMPLING LOOP ###

def run_targeted_resampling(df, labelled_df, targets,
                             aspects_keywords = aspects_keywords,
                             text_col    = "cleaned_review_text",
                             aspect_col  = "aspect",
                             label_col   = "sentiment_labels",
                             year_col    = "year",
                             year_start  = 2016,
                             year_end    = 2026,
                             seed        = 42,
                             batch_size  = 10):
    """Main resampling loop for all classes"""

    existing_chunks = labelled_df[text_col].tolist()
    updated_pos_parts  = []  
    resample_parts     = []  
    augment_parts      = [] 

    non_positive_df = labelled_df[labelled_df[label_col] != "positive"].copy()

    for aspect, class_targets in targets.items():

        for label, n_needed in class_targets.items():
            if n_needed == 0:
                continue

            if label == "positive":
                current_count = len(labelled_df[
                    (labelled_df[aspect_col] == aspect) &
                    (labelled_df[label_col]  == "positive")
                ])
                full_target = current_count + n_needed

                kept_pos, new_pos = resample_positive(
                    df, labelled_df, aspect,
                    target         = full_target,
                    text_col       = text_col,
                    aspect_col     = aspect_col,
                    label_col      = label_col,
                    year_col       = year_col,
                    year_start     = year_start,
                    year_end       = year_end,
                    existing_chunks= existing_chunks,
                    aspects_keywords=aspects_keywords,
                    seed           = seed
                )
                updated_pos_parts.append(kept_pos)
                if not new_pos.empty:
                    resample_parts.append(new_pos)

            elif label == "not_mentioned":
                result = resample_not_mentioned(
                    df, aspect, n_needed,
                    text_col       = text_col,
                    year_col       = year_col,
                    year_start     = year_start,
                    year_end       = year_end,
                    existing_chunks= existing_chunks,
                    seed           = seed
                )
                resample_parts.append(result)

            elif label in ("negative", "neutral"):
                auto_df, manual_df = resample_with_augmentation(
                    labelled_df, df, aspect, label, n_needed,
                    text_col        = text_col,
                    year_col        = year_col,
                    year_start      = year_start,
                    year_end        = year_end,
                    existing_chunks = existing_chunks,
                    seed            = seed,
                    batch_size      = batch_size
                )
                if not auto_df.empty:
                    augment_parts.append(auto_df)
                if not manual_df.empty:
                    resample_parts.append(manual_df)

    # Assemble outputs
    resampled_pos_df = (pd.concat(updated_pos_parts).reset_index(drop=True)
                        if updated_pos_parts else pd.DataFrame())
    resample_df      = (pd.concat(resample_parts).reset_index(drop=True)
                        if resample_parts else pd.DataFrame())
    augment_df       = (pd.concat(augment_parts).reset_index(drop=True)
                        if augment_parts else pd.DataFrame())
    updated_labelled_df = pd.concat([non_positive_df, resampled_pos_df]).reset_index(drop=True)

    # Save
    if not resample_df.empty:
        resample_df.to_excel("resampled_samples.xlsx", index=False)
        print(f"\nSaved {len(resample_df)} samples needing manual labelling")

    if not augment_df.empty:
        augment_df.to_excel("augmented_samples.xlsx", index=False)
        print(f"Saved {len(augment_df)} samples with augmentation")

    return updated_labelled_df, resample_df, augment_df


### FINAL TRIMMING AND SPLITTING ###

def trim_labelled_set(labelled_df, targets,
                      aspect_col="aspect", label_col="label",
                      seed=42):
    """Trim labelled dataframe to required number of samples per class, to handle dominant classes that exceed targets"""
    
    trimmed_parts = []

    for aspect, aspect_group in labelled_df.groupby(aspect_col):
        label_parts = []

        for label, label_group in aspect_group.groupby(label_col):
            n_available = len(label_group)
            n_target    = targets.get(label, None)

            if n_target is None:
                label_parts.append(label_group)

            elif n_available <= n_target:
                label_parts.append(label_group)

            else:
                label_parts.append(label_group.sample(n=n_target, random_state=seed))

        trimmed_parts.append(pd.concat(label_parts))

    trimmed_df = pd.concat(trimmed_parts).reset_index(drop=True)

    # Exclude any labels not in targets
    trimmed_df = trimmed_df[trimmed_df[label_col].isin(targets.keys())].reset_index(drop=True)

    summary = (
        trimmed_df.groupby([aspect_col, label_col])
        .size()
        .unstack(fill_value=0)
    )
    summary["TOTAL"] = summary.sum(axis=1)

    return trimmed_df

def split_labelled_set(labelled_df, train_size=0.7, val_size=0.15, test_size=0.15,
                       aspect_col="aspect", label_col="label", seed=42):
    
    """Stratified train/val/test split by combination of aspect and sentiment label"""

    stratify_col = labelled_df[aspect_col] + " | " + labelled_df[label_col]

    train_df, temp_df = train_test_split(
        labelled_df,
        test_size    = val_size + test_size,
        stratify     = stratify_col,
        random_state = seed
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size    = test_size / (val_size + test_size),
        stratify     = temp_df[aspect_col] + " | " + temp_df[label_col],
        random_state = seed
    )

    # Print num of rows in each set
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)