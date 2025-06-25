
import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset
import time

import pymongo
from bson.binary import Binary, BinaryVectorDtype
from voyageai import Client

api_key = ''

vo = Client(api_key=api_key)

connection_str = ''

client = pymongo.MongoClient(connection_str) # mongodb cluster URI
db = client['vector-test']
coll = db['large_amazon_dataset'] # coll = db['2048d_amazon_dataset'] if doing medium dataset load 


def generate_bson_vector(vector, vector_dtype):
    return Binary.from_vector(vector, vector_dtype)

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# medium dataset categories
categories = [
  'All_Beauty',
  'Amazon_Fashion',
  'Appliances',
  'Arts_Crafts_and_Sewing',
  'Automotive',
  'Baby_Products',
  'Beauty_and_Personal_Care',
  'CDs_and_Vinyl',
  'Cell_Phones_and_Accessories',
  'Gift_Cards',
  'Grocery_and_Gourmet_Food',
  'Handmade_Products',
  'Health_and_Household',
  'Health_and_Personal_Care',
  'Industrial_and_Scientific',
  'Magazine_Subscriptions',
  'Movies_and_TV',
  'Musical_Instruments',
  'Office_Products'
]


# large dataset categories (all)

categories = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care", 
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store", 
    "Magazine_Subscriptions", 
    "Movies_and_TV", 
    "Musical_Instruments", 
    "Office_Products",
    "Patio_Lawn_and_Garden", 
    "Pet_Supplies", 
    "Software",
    "Sports_and_Outdoors", 
    "Subscription_Boxes", 
    "Tools_and_Home_Improvement", 
    "Toys_and_Games",
    "Video_Games", 
    "Unknown", 
]

# Create a mapping from category to index
category_start_id: Dict[str, int] = {
    cat: i * 10_000_000 for i, cat in enumerate(categories)
}


def strip_brackets(text):
    if isinstance(text, str):
        return " ".join(re.findall(r"\[([^][]+)\]", text))
    elif isinstance(text, list):
        return " ".join(strip_brackets(t) for t in text if t)


def clean_data(category: str) -> Optional[pd.DataFrame]:
    logging.info(f"Processing category: {category}")
    prices = np.random.randint(1, 100, 1000)

    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full"
    )
    cols_to_keep = ["title", "description", "average_rating", "price"]
    ds = ds.map(
        lambda entry: process_entry(entry, prices),
        batched=False,
        remove_columns=[c for c in ds.column_names if c not in cols_to_keep],
    )

    logging.info(f"{len(ds)} records before processing")
    ds = ds.filter(lambda x: x["description"] is not None)
    logging.info(f"{len(ds)} records after removing descriptions of None")
    df = ds.to_pandas()
    del ds
    df["id"] = df.index + category_start_id[category]
    df["category"] = category
    df.drop_duplicates(subset=["title", "description"], inplace=True)
    df.dropna(subset=["title", "description"], inplace=True)
    df["price"] = (df["price"] * 100).astype(int)
    return df


def process_entry(entry: Dict, prices: np.ndarray) -> Dict:
    price_str = entry["price"]
    try:
        price = float(re.sub(r"[^\d.]", "", price_str))
    except ValueError:
        price = np.random.choice(prices)
    description = strip_brackets(entry["description"]) if entry["description"] else None
    return {
        "title": entry["title"],
        "description": description,
        "average_rating": entry["average_rating"],
        "price": price,
    }

def voyage_embed_with_retry(batch):

    max_retries = 5
    for attempt in range(max_retries):
        try:
            batch_embs = vo.embed(batch, model="voyage-3.5", input_type="document", output_dimension=2048)
            break  # Success, exit the loop
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts: {e}")
                raise  # Re-raise the exception
            else:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    return batch_embs


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding embeddings...")
    documents = "Item: Title: " + df["title"] + " Description: " + df["description"]
    documents = documents.tolist()
    batch_size = 200
    embs = [] 
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing chunk {i//batch_size + 1}: {len(batch)} items")
        batch_embs = voyage_embed_with_retry(batch)
        embs.extend(batch_embs.embeddings)
    embeddings = [generate_bson_vector(embedding, BinaryVectorDtype.FLOAT32) for embedding in embs] # comment out and replace embs with embeddingsfor medium dataset
    df["embedding"] = embeddings
    del batch_embs
    del embs
    del embeddings
    logging.info("Embeddings added.")
    return df


def split_dataframe(df, max_size=100000):
    """
    Split a DataFrame into chunks where each chunk has at most max_size rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to split
    max_size : int, default 1000000
        Maximum number of rows in each chunk
    
    Returns:
    --------
    list of pandas.DataFrame
        List containing the DataFrame chunks
    """
    # Calculate number of chunks needed
    num_chunks = (len(df) + max_size - 1) // max_size
    
    # Split the DataFrame
    df_chunks = []
    for i in range(num_chunks):
        start_idx = i * max_size
        end_idx = min((i + 1) * max_size, len(df))
        df_chunks.append(df.iloc[start_idx:end_idx].copy())
    
    return df_chunks



def process_category(category: str) -> Optional[pd.DataFrame]:
    df = clean_data(category)

    if df is not None:
        processed_df_chunks = []
        max_chunk_size = 100000
        total_rows = len(df)
        if total_rows > max_chunk_size:
            df_chunks = split_dataframe(df)
            num_chunks = (total_rows + max_chunk_size - 1) // max_chunk_size
            for i, df_chunk in enumerate(df_chunks):
                start_idx = i * max_chunk_size
                end_idx = min((i + 1) * max_chunk_size, total_rows)
                print(f"Processing chunk {i+1}/{num_chunks} (rows {start_idx:,} to {end_idx:,})")
                df_chunk = add_embeddings(df_chunk)
                logging.info(f"writing df for {category} to mongo") 
                coll.insert_many(df_chunk.to_dict('records'))
        else:
            print(f"Processing df")
            df_processed = add_embeddings(df)
            logging.info(f"writing df for {category} to mongo") 
            coll.insert_many(df_processed.to_dict('records'))
      

if __name__ == "__main__":
    for category in categories:
        logging.info(f"creating embeddings df for {category}") 
        df_processed = process_category(category)
        if df_processed is None:
            continue
