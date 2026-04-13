import pandas as pd
import numpy as np
import re
from collections import defaultdict
from deep_translator import GoogleTranslator

## Keywords for targeted resampling to easily capture relevant samples

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

### Initial sampling function

def sample_for_manually_labelled_set(df, text_col, n_per_aspect=400, seed=42):
    """Initial sampling with even distribution across aspects and years."""
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


### Resampling functions for underrepresented classes

def _get_pool(df, aspect, text_col, year_col, year_start, year_end,
              exclude_chunks, aspect_col="aspect", low_rating=False):
    """Function to build a clean pool for any aspect."""
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

    return pool


def _keyword_filter(pool, text_col, keywords, n_needed):
    """Function to filter pool by keywords. If not enough samples with keywords, return original pool for random sampling."""
    if not keywords:
        return pool
    pattern      = "|".join([re.escape(kw) for kw in keywords])
    keyword_pool = pool[pool[text_col].str.lower().str.contains(pattern, na=False)]
    return keyword_pool if len(keyword_pool) >= n_needed else pool


def _make_sample(pool, n, text_col, aspect, label, note, seed):
    """Function to sample n rows and add label columns."""
    n = min(n, len(pool))
    cols_to_keep = ["review_id", "aspect", "year", "main_topic", "rating", "cleaned_review_text", "cleaned_demojize_review_text", "cleaned_review_title"] 
    sampled = pool.sample(n=n, random_state=seed)[cols_to_keep].copy()
    
    # sampled = pool.sample(n=n, random_state=seed)[[text_col]].copy()
    sampled["label"]  = label
    sampled["note"]   = note
    return sampled

def resample_positive(df, labelled_df, aspect, target, text_col, aspect_col,
                        label_col, year_col, year_start, year_end,
                        existing_chunks, aspects_keywords, seed):
    """Resampling for positive class by targeted upsampling."""

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
                     existing_chunks, aspect_col)
    pool = _keyword_filter(pool, text_col, aspects_keywords.get(aspect, []), diff)
    sampled = _make_sample(pool, diff, text_col, aspect, "", "resample_positive", seed)

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
    sampled = _make_sample(pool, n_needed, text_col, aspect, "", "resample_not_mentioned", seed)

    return sampled

def resample_negative(df, aspect, n_needed, text_col, year_col,
                        year_start, year_end, existing_chunks, seed):
    """Resampling for negative text using keyword targeting."""

    pool = _get_pool(df, aspect, text_col, year_col, year_start, year_end, existing_chunks, low_rating=True)
    pool = _keyword_filter(pool, text_col, negative_keywords.get(aspect, []), n_needed)
    sampled = _make_sample(pool, n_needed, text_col, aspect, "", "resample_negative", seed)

    return sampled

PIVOT_LANGUAGES = ["fr", "de", "es"]
CHAIN_PAIRS = [("fr", "de"), ("de", "es"), ("es", "fr"), ("fr", "es"), ("es", "de"), ("de", "fr")]

def augment_neutral(labelled_df, df, aspect, n_needed, text_col,
                      year_col, year_start, year_end, existing_chunks, seed):
    """Resampling for neutral text using augmentation through back-translation."""

    existing_neutral = labelled_df[
        (labelled_df["aspect"]           == aspect) &
        (labelled_df["sentiment_labels"] == "neutral")
    ][text_col].tolist()

    augmented = []
    seen_texts = set(existing_neutral)

    if existing_neutral:
        max_single_hop = len(existing_neutral) * len(PIVOT_LANGUAGES)
        use_chaining   = n_needed > max_single_hop

        for i in range(n_needed):
            original = existing_neutral[i % len(existing_neutral)]

            try:
                if not use_chaining:
                    pivot    = PIVOT_LANGUAGES[i % len(PIVOT_LANGUAGES)]
                    mid = GoogleTranslator(source="en", target=pivot).translate(original)
                    back   = GoogleTranslator(source=pivot, target="en").translate(mid)
                else:
                    lang1, lang2 = CHAIN_PAIRS[i % len(CHAIN_PAIRS)]
                    mid1  = GoogleTranslator(source="en",   target=lang1).translate(original)
                    mid2  = GoogleTranslator(source=lang1,  target=lang2).translate(mid1)
                    back  = GoogleTranslator(source=lang2,  target="en").translate(mid2)

                if back and back not in seen_texts:  # skip true duplicates
                    seen_texts.add(back)
                    note = (f"augmented_bt_{lang1}_{lang2}_from: {original[:50]}"
                            if use_chaining else
                            f"augmented_bt_{pivot}_from: {original[:50]}")
                    augmented.append({
                        text_col : back,
                        "aspect" : aspect,
                        "label"  : "neutral",
                        "note"   : note
                    })

            except Exception as e:
                print(f"  Translation failed: {e}")

    if len(augmented) >= n_needed:
        return pd.DataFrame(augmented), pd.DataFrame()

    # Fallback — general pool
    remaining = n_needed - len(augmented)
    pool      = _get_pool(df, aspect, text_col, year_col, year_start, year_end, existing_chunks)
    fallback  = _make_sample(pool, remaining, text_col, aspect, "", "manual_label_neutral", seed)

    return pd.DataFrame(augmented), fallback


def run_targeted_resampling(df, labelled_df, targets,
                             aspects_keywords = aspects_keywords,
                             text_col    = "cleaned_review_text",
                             aspect_col  = "aspect",
                             label_col   = "sentiment_labels",
                             year_col    = "year",
                             year_start  = 2016,
                             year_end    = 2026,
                             seed        = 42):
    """Main resampling loop for all classes."""
    existing_chunks = labelled_df[text_col].tolist()

    # Separate outputs
    updated_pos_parts  = []  
    resample_parts     = []  
    augment_parts      = [] 

    # Keep non-positive rows untouched
    non_positive_df = labelled_df[labelled_df[label_col] != "positive"].copy()

    for aspect, class_targets in targets.items():

        for label, n_needed in class_targets.items():
            if n_needed == 0:
                continue

            if label == "positive":
                kept_pos, new_pos = resample_positive(
                    df, labelled_df, aspect,
                    target         = n_needed,
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

            elif label == "negative":
                result = resample_negative(
                    df, aspect, n_needed,
                    text_col       = text_col,
                    year_col       = year_col,
                    year_start     = year_start,
                    year_end       = year_end,
                    existing_chunks= existing_chunks,
                    seed           = seed
                )
                resample_parts.append(result)

            elif label == "neutral":
                auto_df, manual_df = augment_neutral(
                    labelled_df, df, aspect, n_needed,
                    text_col       = text_col,
                    year_col       = year_col,
                    year_start     = year_start,
                    year_end       = year_end,
                    existing_chunks= existing_chunks,
                    seed           = seed
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
        resample_df.to_excel("resampled_for_labelling.xlsx", index=False)
        print(f"\nSaved {len(resample_df)} samples needing manual labelling")

    if not augment_df.empty:
        augment_df.to_excel("augmented_neutral.xlsx", index=False)
        print(f"Saved {len(augment_df)} neutral samples with augmentation")

    return updated_labelled_df, resample_df, augment_df
