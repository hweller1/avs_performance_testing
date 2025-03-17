
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from typing import List, Set, Tuple, Union

import pandas as pd
import numpy as np
import pymongo
from sentence_transformers import SentenceTransformer

from transformers import AutoModel
from bson.binary import Binary, BinaryVectorDtype


searches = [
   "handmade ceramic mugs for sale",
   "best acrylic paint brushes set",
   "polymer clay starter kit",
   "scrapbooking paper 12x12",
   "watercolor palette professional",
   "knitting needles bamboo set",
   "crafting scissors bulk pack",
   "crochet hooks ergonomic grip",
   "glue gun sticks mini size",
   "jewelry making supplies kit",
   "leather crafting tools beginner",
   "origami paper patterns pack",
   "calligraphy pens set black",
   "canvas panels bulk 8x10",
   "washi tape decorative pack",
   "yarn wool merino organic",
   "beading supplies wholesale",
   "resin molds jewelry silicone",
   "embroidery kit beginners",
   "chalk paint furniture diy",
   "ribbon spools assorted colors",
   "stamp pad ink waterproof",
   "mod podge gloss finish",
   "feltmaking wool roving",
   "sticker paper printable sheets",
   "drawing pencils professional set",
   "macrame cord natural cotton",
   "glass beads assorted colors",
   "sewing machine needles universal",
   "paper quilling strips multicolor",
   "paint markers permanent craft",
   "wire wrapping tools jewelry",
   "gouache paint set artist",
   "wood burning kit beginner",
   "adhesive vinyl sheets craft",
   "art easel tabletop wooden",
   "brush pen markers calligraphy",
   "craft storage organizer boxes",
   "decoupage paper vintage",
   "embossing powder heat tool",
   "fabric paint textile permanent",
   "gemstone beads natural",
   "hot glue gun cordless",
   "ink pad stamping black",
   "jute twine thick natural",
   "kraft paper rolls brown",
   "latch hook kit beginner",
   "metal stamping tools set",
   "needle felting starter kit",
   "oil pastels professional grade"
]


def generate_random_query_vectors(dims) -> List:
	embeddings = np.random.uniform(
		low=-0.5,
		high=0.5,
		size=(10000, dims)
	)
	# Convert to Python list and ensure float precision
	embeddings_list = embeddings.tolist()
	return embeddings_list


def save_embeddings(
	embeddings: List[List[float]],
	output_file: str = "random_embeddings.json"
) -> None:
	"""
	Save embeddings to JSON file.
	
	Args:
		embeddings: List of embedding vectors
		output_file: Path to output JSON file
	"""
	# Create dictionary with metadata
	data = {
		"num_samples": len(embeddings),
		"dimensions": len(embeddings[0]),
		"embeddings": embeddings
	}
	
	# Save to JSON file
	with open(output_file, 'w') as f:
		json.dump(data, f)

def load_embeddings(input_file: str = "random_embeddings.json") -> List[List[float]]:
    """
    Load embeddings from JSON file.
    
    Args:
        input_file: Path to input JSON file
    
    Returns:
        List of embedding vectors
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    embeddings = data["embeddings"]
    print(f"Loaded {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    return embeddings


def compute_overlap(exact_result_set: List, approx_result_set: Set) -> float:
	# each result set is a list of titles, order not considered
	return len(set(exact_result_set).intersection(approx_result_set)) / len(set(exact_result_set))


MODEL_NAME: str = "Snowflake/snowflake-arctic-embed-xs"
CACHE_PATH: str = "./model_cache"

# model = SentenceTransformer(model_name_or_path=MODEL_NAME, cache_folder=CACHE_PATH)
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)



embeddings = model.encode(searches, truncate_dim=512,  normalize_embeddings=True)


random_embeddings = generate_random_query_vectors(512)
# save_embeddings(generate_random_query_vectors())
# random_embeddings = load_embeddings()

#import pdb; pdb.set_trace() # -0.5, 0.5

# for elem in embeddings: print(f"min: {min(elem)} max: {max(elem)}")

mongo_uri = ''



client = pymongo.MongoClient(mongo_uri, readPreference='nearest')

db = client['vector-test']

collections = [db['amazon_ecommerce_jina_512_1m']]
# collections = [db['amazon_ecommerce_2']]


indexes = ['jina_512_1m']
# indexes = ['medium_vector_index']

# k_values = [5, 10, 100]
k_values = [5]
mult = [200, 1000]
# mult = [2000]
# mult = [100]
# mult = [5, 10, 20, 40, 100, 200, 1000, 2000]
# random_query_tests = [100, 1000, 2000, 5000, 10000]



plot_results = {}
# num_exact_results  = 100


def run_query(embedding, k, candidates, filter_clause, exact_title_list=[], random=False) -> Union[Set, Tuple[float, float]]:
	# import pdb; pdb.set_trace()
	if not isinstance(embedding, list):
		embedding = embedding.tolist()
	if exact_title_list == [] and not random:
		if filter_clause is None:
			query = [
						{
						"$vectorSearch": {
							"index":f'{indexes[i_0]}',
							"path": "embedding_512",
							"queryVector": embedding,
							"limit": k,
							"exact": True
						},
						},
						{
						"$project": {"embedding_512": 0}
						}
					]
		else:	
			query = [
						{
						"$vectorSearch": {
							"index":f'{indexes[i_0]}',
							"path": "embedding_512",
							"queryVector": embedding,
							"limit": k,
							"exact": True,
							"filter": filter_clause
						},
						},
						{
						"$project": {"embedding_512": 0}
						}
					]
			# import pdb; pdb.set_trace()
		results = []
		x = coll.aggregate(query)

		for result_index in range(k):
			try:
				results.append(x.next()['title'])
			except:
				break
		return results

	if filter_clause is None:
		query = [
					{
					"$vectorSearch": {
						"index":f'{indexes[i_0]}',
						"path": "embedding_512",
						"queryVector": embedding,
						"limit": k,
						"numCandidates": candidates,
					},
					},
					{
					"$project": {"embedding_512": 0}
					}
				]
	else:	
		query = [
					{
					"$vectorSearch": {
						"index":f'{indexes[i_0]}',
						"path": "embedding_512",
						"queryVector": embedding,
						"limit": k,
						"numCandidates": candidates,
						"filter": filter_clause
					},
					},
					{
					"$project": {"embedding_512": 0}
					}
				]
	start = time.time()
	x = coll.aggregate(query)
	results = []
	for result_index in range(k):
		try:
			results.append(x.next()['title'])
		except:
			break
	end = time.time()
	results = set(results)
	if not random:
		overlap = compute_overlap(exact_title_list[:k], results)
	else:
		overlap = None

	return overlap, end - start


# filename = "exact_results_data_jina_512_1m.json"

filename = "exact_results_data_jina.json"
# filename = "small_exact_results_data.json"
# filename = "medium_exact_results_data.json"


def write_json(data, filename=filename):
   with open(filename, 'w') as f:
	   json.dump(data, f, indent=4)

def read_json(filename=filename):
   with open(filename) as f:
	   return json.load(f)

# Or parse JSON string
# tests = ['filtered']
tests = ['concurrency_10_random']
# tests = ['no_filter', 'filtered', 'concurrency_10', 'concurrency_100']
for i_0, coll in enumerate(collections):
	for test in tests:
		print(f"running exact test 1M {test}\n")
		if 'concurrency' in test:
			concurrency = int(test.split('_')[1])
		else:
			concurrency = 1
		if "random" in test:
			break 

		original_concurrency = None
		for i_1, k in enumerate([k_values[-1]]): 
			times = []
			recalls = []
			futures = []
			for i, embedding in enumerate(embeddings):
				try:
					exact_title_list = read_json() 
				except: 
					exact_title_list = {}
				if 'filtered' in test:
					filter_type = "compound"
					filter_clause = {"$and": [
					 {'price': {'$lte': 1000}},
					 {'category': {'$eq': 'Pet Supplies'}}
					 ]
					 }

					if f"{test}.{i}" not in exact_title_list.keys():

						print(f"producing exact results for test {test}.{i}")
						exact_title_list[f"{test}.{i}"] = run_query(embedding=embedding,
														 k=k,
														 candidates=None,
														 filter_clause=filter_clause)
						write_json(exact_title_list, filename)
					# filter_type = 'Random'
					# filter_clause = None
					# if f"Random.{i}" not in exact_title_list.keys():
					# 	print(f"producing exact results for test Random.{i}")
					# 	exact_title_list[f"Random.{i}"] = run_query(embedding=embedding,
					# 								   k=k,
					# 								   candidates=None,
					# 								   filter_clause=filter_clause)
					# 	write_json(exact_title_list, filename)

				
				else:
					filter_type = 'Unfiltered'
					filter_clause = None
					if f"Unfiltered.{i}" not in exact_title_list.keys():
						print(f"producing exact results for test Unfiltered.{i}")
						exact_title_list[f"Unfiltered.{i}"] = run_query(embedding=embedding,
													   k=k,
													   candidates=None,
													   filter_clause=filter_clause)
						write_json(exact_title_list, filename)

	print("finished producing exact result sets")


	for test in tests:
		if 'filtered' in test:
			filter_type = "compound"
			filter_clause = {"$and": [{'price': {'$lte': 1000}}, {'category': {'$eq': 'Pet Supplies'}}]}
		else:
			filter_type = 'Unfiltered'
			filter_clause = None

		print(f"running approx test 1M {test}\n")

		if 'concurrency' in test:
			concurrency = int(test.split('_')[1])
		else:
			concurrency = 1
		
		if "random" in test:
			test_embeddings = random_embeddings
			query_test_sizes = [50, 100, 1000, 2000, 5000, 10000]
		else:
			test_embeddings = embeddings
			query_test_sizes = [50]

		original_concurrency = None
		for i_1, k in enumerate(k_values): 
			for multiplier in mult:
				candidates = k * multiplier
				# if k == 100:
				# 	candidates = 4000
				if candidates > 10000:
					candidates = 10000
				times = []
				recalls = []
				futures = []
				print(f"running for limit, numCandidates of {k}, {candidates}\n")
				# import pdb; pdb.set_trace()
				if original_concurrency is not None:
					concurrency = original_concurrency

				elif concurrency > candidates:
					original_concurrency = copy(concurrency)
					concurrency = candidates

				executor = ThreadPoolExecutor(max_workers=concurrency)
				for query_test_size in query_test_sizes:
					print(f"running query test size of {query_test_size}\n")
					for i, embedding in enumerate(test_embeddings[:query_test_size]):

						if concurrency == 1:
							if "random" in test:
								overlap, query_time = run_query(embedding=embedding,
															k=k,
															candidates=candidates,
															filter_clause=filter_clause, 
															random=True
															)
							else:
								if filter_clause is None:
									exact_titles = exact_title_list[f"Unfiltered.{i}"][:query_test_size]
								else:
									exact_titles = exact_title_list[f"{test}.{i}"]
								# import pdb; pdb.set_trace()
								overlap, query_time = run_query(embedding=embedding,
																k=k,
																candidates=candidates,
																filter_clause=filter_clause, 
																exact_title_list=exact_titles
																)

							recalls.append(overlap)
							times.append(query_time)
						else:
							if "random" in test:
								f = executor.submit(run_query, embedding, k, candidates, filter_clause, [], random=ThreadPoolExecutor)
							else:
								f = executor.submit(run_query, embedding, k, candidates, filter_clause, exact_title_list[f"Unfiltered.{i}"])
							futures.append(f)
					start_time = time.time()
					if concurrency != 1:
						for f in as_completed(futures):
							overlap, query_time = f.result()
							recalls.append(overlap)
							times.append(query_time)
						total_time = time.time() - start_time 
						qps = query_test_size / total_time
					else:
						qps = query_test_size / np.sum(times)

					plot_results[f"1M.k{k}.candidates{candidates}.{test}.concurrency{concurrency}.test_queries{query_test_size}"] = {
					   "Limit": k,
					   "numCandidates": candidates,
					   "Concurrent Requests": concurrency,
					   "Num Test Queries": query_test_size,
					   "Filter Type": filter_type,
						"Recall": np.mean(recalls) * 100 if "random" not in test else None, 
						"Mean Latency": np.mean(times) * 1000,
						"p99 Latency": np.percentile(times, 99) * 1000,
						"QPS": qps}


flattened_list = [(outer_key, *inner_dict.values()) for outer_key, inner_dict in plot_results.items()]
# import pdb; pdb.set_trace()
df = pd.DataFrame(flattened_list)
df.columns = ["Test Case", "Limit", "numCandidates", "Concurrent Requests", "Num Test Queries", "Filter Type", "Recall", "Mean Latency (ms)", "p99 Latency", "QPS"]
df.to_csv(f'results/3_2025_runs/1M_m20_amazon_ecommerce_all_runs_jina512_random_5.csv')



