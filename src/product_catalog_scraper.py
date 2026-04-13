import json
import time
import random
import requests
import pandas as pd
from tqdm.notebook import tqdm
import re

### Config
OUTPUT_FILE       = "sephora_skincare_products.csv"
RESULTS_PER_PAGE  = 60    

CNSTRC_API_KEY    = "u7PNVQx-prod-en-us"
CATEGORY          = "cat150006"  
BASE_URL          = "https://sephora.cnstrc.com/browse/group_id/{category}"

HEADERS = {
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://www.sephora.com",
    "Referer":         "https://www.sephora.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

def fetch_page(session, page, per_page = RESULTS_PER_PAGE):
    """Params to call Constructor.io API for a each page number."""

    url = BASE_URL.format(category=CATEGORY)
    params = {
        "key"                             : CNSTRC_API_KEY,
        "page"                            : page,
        "num_results_per_page"            : per_page,
        "_dt"                             : int(time.time() * 1000),
        "c"                               : "ciojs-client-2.74.0",
        "i"                               : "be9612a0-b2fe-4d71-935d-3f99f9794b47",
        "s"                               : "1",
    }

    resp = session.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()

    return resp.json()

def parse_facet(facets, name):
    """Extract values list for a named facet, returned as comma-separated string."""
    
    for f in facets:
        if f.get("name") == name:
            return ", ".join(str(v) for v in f.get("values", []))
    return ""

def parse_response(data):
    """Extract product fields from a single Constructor.io API response."""

    results = []

    for item in data.get("response", {}).get("results", []):
        meta = item.get("data", {})
        url_path = meta.get("url", "")
        sku      = meta.get("currentSku", {})
        facets   = meta.get("facets", [])

        # Extract P-number to retrieve product ID
        id_match   = re.search(r"(P\d+)", url_path)
        product_id = id_match.group(1) if id_match else meta.get("id", "")

        results.append({
            "product_id":   product_id,
            "product_name": item.get("value", ""),
            "brand":        meta.get("brandName", ""),
            "price":        sku.get("listPriceFloat", ""),
            "rating":       meta.get("rating", ""),
            "num_reviews":  meta.get("totalReviews", ""),
            "skin_type":    parse_facet(facets, "skinType"),
            "skin_concern": parse_facet(facets, "skinConcerns"),
            "product_link": "https://www.sephora.com" + url_path.split("?")[0],
        })
    return results


def scrape_sephora_api(max_products, per_page = RESULTS_PER_PAGE,):
    """ Main function to scrape all products listed under skincare category through Constructor.io API"""

    session  = requests.Session()
    all_rows = []
    seen_ids = set()
    page     = 1
    pages_needed = -(-max_products // per_page)

    pbar = tqdm(total=max_products, desc="Products fetched", unit=" products")

    while page <= pages_needed:
        try:
            data = fetch_page(session, page=page, per_page=per_page)
        except requests.HTTPError as e:
            print(f"HTTP error on page {page}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error on page {page}: {e}")
            break

        rows = parse_response(data)
        if not rows:
            print(f"Page {page} returned 0 results — end of catalog.")
            break

        # Prevent duplicates by product_id
        new_rows = [r for r in rows if r["product_id"] not in seen_ids]
        for r in new_rows:
            seen_ids.add(r["product_id"])

        all_rows.extend(new_rows)
        pbar.update(len(new_rows))

        if len(all_rows) >= max_products:
            break

        page += 1
        
        time.sleep(random.uniform(0.8, 1.8))

    pbar.close()

    df = pd.DataFrame(all_rows[:max_products]).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProducts saved to  \t: {OUTPUT_FILE}")
    print(f"Total unique products \t: {len(df)}")
    
    return df
