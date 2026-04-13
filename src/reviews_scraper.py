import json
import time
import random
import os
import sys
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

### Config
PRODUCTS_FILE   = "sephora_skincare_products.csv"
REVIEWS_FILE    = "sephora_reviews.csv"
PROGRESS_FILE   = "scrape_progress.json"
ERROR_LOG_FILE  = "scrape_errors.log"

BV_URL              = "https://api.bazaarvoice.com/data/reviews.json"
BV_PASSKEY          = "calXm2DyQVjcCy9agq85vmTJv5ELuuBCF2sdg4BnJzJus"
BV_REVIEWS_PER_PAGE = 100

def scrape_reviews(product_id, session):
    """Fetch all reviews for a single Sephora product_id through the Bazaarvoice API."""

    base_params = [
        ("Filter",     "contentlocale:en*"),
        ("Filter",     f"ProductId:{product_id}"),
        ("Sort",       "SubmissionTime:desc"),
        ("Limit",      BV_REVIEWS_PER_PAGE),
        ("Include",    "Products,Comments"),
        ("Stats",      "Reviews"),
        ("passkey",    BV_PASSKEY),
        ("apiversion", "5.4"),
        ("Locale",     "en_US"),
    ]

    reviews = []
    offset  = 0
    total   = None

    while True:
        params = base_params + [("Offset", offset)]

        try:
            resp = session.get(BV_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {resp.status_code} at offset {offset} — {e}"
            return reviews, error
        except requests.exceptions.ConnectionError as e:
            error = f"Connection error at offset {offset} — {e}"
            return reviews, error
        except requests.exceptions.Timeout:
            error = f"Timed out at offset {offset}"
            return reviews, error
        except Exception as e:
            error = f"Unexpected error at offset {offset} — {type(e).__name__}: {e}"
            return reviews, error

        try:
            data = resp.json()
        except Exception as e:
            error = f"Failed to parse JSON at offset {offset} — {e}"
            return reviews, error

        if total is None:
            total = data.get("TotalResults", 0)
            # product has no reviews
            if total == 0:
                return reviews, None

        batch = data.get("Results", [])
        if not batch:
            break

        for r in batch:
            reviews.append({
                "product_id":        product_id,
                "review_id":         r.get("Id"),
                "rating":            r.get("Rating"),
                "review_title":      r.get("Title"),
                "review_text":       r.get("ReviewText"),
                "timestamp":         r.get("SubmissionTime"),
                "reviewer_nickname": r.get("UserNickname"),
            })

        offset += len(batch)
        if offset >= total:
            break

        time.sleep(0.2) 

    return reviews, None

def load_progress():
    """Load the set of completed product IDs from the progress file."""

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_progress(completed):
    """Save the set of completed product IDs to the progress file."""

    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": list(completed)}, f)


def append_reviews_to_csv(rows):
    """Append a batch of reviews to the output CSV. Creates the file with headers on first call."""

    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(REVIEWS_FILE)
    df.to_csv(REVIEWS_FILE, mode="a", index=False, header=write_header)


def log_error(product_id, product_name, error):
    """Append a line to the error log file after processing a product."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}]  {product_id:<12}  {product_name:<50}  {error}\n"
    with open(ERROR_LOG_FILE, "a") as f:
        f.write(line)

def scrape_all_reviews(products_file=PRODUCTS_FILE):
    """Main function to scrape reviews for all products."""

    if not os.path.exists(products_file):
        sys.exit(1)

    products_df = pd.read_csv(products_file)

    name_lookup = (
        products_df.set_index("product_id")["product_name"].to_dict()
        if "product_name" in products_df.columns else {}
    )

    product_ids = products_df["product_id"].dropna().unique().tolist()

    completed   = load_progress()
    remaining   = [p for p in product_ids if p not in completed]
    error_count = 0

    session = requests.Session()

    for pid in tqdm(remaining, desc="Scraping reviews", unit=" product"):
        product_name = name_lookup.get(pid, "unknown")

        reviews, error = scrape_reviews(pid, session=session)

        if error:
            error_count += 1
            tqdm.write(f"ERROR — {pid} ({product_name}): {error}")
            log_error(pid, product_name, error)
            append_reviews_to_csv(reviews)
        else:
            append_reviews_to_csv(reviews)
            completed.add(pid)
            save_progress(completed)

        # Polite delay between products
        time.sleep(random.uniform(1.0, 2.5))

    # Final summary
    print(f"\n{'All done.' if error_count == 0 else 'Done with errors.'}")
    print(f"Reviews saved to  : {REVIEWS_FILE}")

    if error_count:
        print(f"Failed products : {error_count}  (see {ERROR_LOG_FILE})")
        print(f"Re-run the script to retry failed products automatically.")

    if os.path.exists(REVIEWS_FILE):
        reviews_df = pd.read_csv(REVIEWS_FILE)
        print(f"Total reviews     : {len(reviews_df):,}")
        print(f"Unique products   : {reviews_df['product_id'].nunique()}")

    return reviews_df if os.path.exists(REVIEWS_FILE) else None
