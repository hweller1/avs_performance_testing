
import json
import random
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from typing import List, Set, Tuple, Union

import pandas as pd
import numpy as np
import pymongo
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

from voyageai import Client
api_key = ''
vo = Client(api_key=api_key)


def read_csv_to_list(file_path):
    """
    Reads a CSV file and returns its content as a list of lists.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of lists representing the CSV data.
    """
    data_list = []
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_list.append(row)
    return data_list


def compute_overlap(exact_result_set: List, approx_result_set: Set) -> float:
	# each result set is a list of titles, order not considered
	return len(set(exact_result_set).intersection(approx_result_set)) / len(set(exact_result_set))

mongo_uri = ''

client = pymongo.MongoClient(mongo_uri, readPreference='nearest')

db = client['vector-test']

collections = [db['large_amazon_dataset']]

indexes = ['large_vector_index']

k_values = [5, 10, 100]
mult = [2, 5, 10, 20, 40, 100, 200, 400, 1000, 2000]

dims = [2048]


for dim in dims:
	embeddings = vo.embed(searches, model="voyage-3.5", input_type="document", output_dimension={dim}).embeddings
	with open(f'{dim}_query_embeddings.csv', 'w', newline='') as file:
	    writer = csv.writer(file)
	    writer.writerows(embeddings)
	embeddings_dict[dim] = embeddings

# uncomment to read query embeddings

# for dim in dims: 
# 	csv_file_path = f'{dim}_query_embeddings.csv'
# 	embeddings = read_csv_to_list(csv_file_path)
# 	embeddings_dict[dim] = [[float(x) for x in string_list] for string_list in embeddings]


plot_results = {}


def run_query(embedding, path, k, candidates, filter_clause, exact_title_list=[], random=False) -> Union[Set, Tuple[float, float]]:
	if not isinstance(embedding, list):
		embedding = embedding.tolist()
	if exact_title_list == [] and not random:
		if filter_clause is None:
			query = [
						{
						"$vectorSearch": {
							"index":f'{indexes[i_0]}',
							"path": path,
							"queryVector": embedding,
							"limit": k,
							"exact": True
						},
						},
						{
						"$project": {path: 0}
						}
					]
		else:	
			query = [
						{
						"$vectorSearch": {
							"index":f'{indexes[i_0]}',
							"path": path,
							"queryVector": embedding,
							"limit": k,
							"exact": True,
							"filter": filter_clause
						},
						},
						{
						"$project": {path: 0}
						}
					]
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
						"path": path,
						"queryVector": embedding,
						"limit": k,
						"numCandidates": candidates,
					},
					},
					{
					"$project": {path: 0}
					}
				]
	else:	
		query = [
					{
					"$vectorSearch": {
						"index":f'{indexes[i_0]}',
						"path": path,
						"queryVector": embedding,
						"limit": k,
						"numCandidates": candidates,
						"filter": filter_clause
					},
					},
					{
					"$project": {path: 0}
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

filename = "exact_results_data_voyage_2048_17m.json"


def write_json(data, filename=filename):
   with open(filename, 'w') as f:
	   json.dump(data, f, indent=4)

def read_json(filename=filename):
   with open(filename) as f:
	   return json.load(f)

# test configuration
tests = ['no_filter', 'filtered', 'concurrency_10', 'concurrency_100']
for i_0, coll in enumerate(collections):
	for test in tests:
		print(f"running exact test 15.3 {test}\n")
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
			for i, embedding in enumerate(embeddings_dict[2048]):
				try:
					exact_title_list = read_json() 
				except: 
					exact_title_list = {}
				if 'filtered' in test:
					filter_type = "compound"
					filter_clause = {"$and": [
					 {'price': {'$lte': 1000}},
					 {'category': {'$eq': 'Pet_Supplies'}}
					 ]
					 }

					if f"{test}.{i}" not in exact_title_list.keys():

						print(f"producing exact results for test {test}.{i}")
						exact_title_list[f"{test}.{i}"] = run_query(embedding=embedding,
														 path='embedding',
														 k=k,
														 candidates=None,
														 filter_clause=filter_clause)
						write_json(exact_title_list, filename)

				
				else:
					filter_type = 'Unfiltered'
					filter_clause = None
					if f"Unfiltered.{i}" not in exact_title_list.keys():
						print(f"producing exact results for test Unfiltered.{i}")
						exact_title_list[f"Unfiltered.{i}"] = run_query(embedding=embedding,
																		path='embedding',
																		k=k,
																		candidates=None,
																		filter_clause=filter_clause)
						write_json(exact_title_list, filename)

	print("finished producing exact result sets")


	for test in tests:
		for dimension in dims:
			if dimension == 2048:
				path = 'embedding'
			else:
				path = f"{dimension}_embedding"
			if 'filtered' in test:
				filter_type = "compound"
				filter_clause = {"$and": [{'price': {'$lte': 1000}}, {'category': {'$eq': 'Pet_Supplies'}}]}
			else:
				filter_type = 'Unfiltered'
				filter_clause = None


			embeddings = embeddings_dict[dimension]
			print(f"running approx test 5.4M {test} at dim {dimension}\n")

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
					if candidates > 10000:
						continue
					times = []
					recalls = []
					futures = []
					print(f"running for limit, numCandidates of {k}, {candidates}\n")
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
																path=path,
																k=k,
																candidates=candidates,
																filter_clause=filter_clause, 
																random=True
																)
								else:
									if filter_clause is None:
										exact_titles = exact_title_list[f"Unfiltered.{i}"][:query_test_size]
									else:
										exact_titles = exact_title_list[f"{test}.{i}"][:query_test_size]
										# import pdb; pdb.set_trace()
									try: 
										overlap, query_time = run_query(embedding=embedding,
																		path=path,
																		k=k,
																		candidates=candidates,
																		filter_clause=filter_clause, 
																		exact_title_list=exact_titles
																		)
									except:
										import pdb; pdb.set_trace() 

								recalls.append(overlap)
								times.append(query_time)
							else:
								if "random" in test:
									f = executor.submit(run_query, embedding, path, k, candidates, filter_clause, [], random=ThreadPoolExecutor)
								else:
									f = executor.submit(run_query, embedding, path, k, candidates, filter_clause, exact_title_list[f"Unfiltered.{i}"])
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

						plot_results[f"15.3M.{dimension}.k{k}.candidates{candidates}.{test}.concurrency{concurrency}.test_queries{query_test_size}"] = {
						   "Dimensions": dimension, 
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
df = pd.DataFrame(flattened_list)
df.columns = ["Test Case", "Dimensions", "Limit", "numCandidates", "Concurrent Requests", "Num Test Queries", "Filter Type", "Recall", "Mean Latency (ms)", "p99 Latency", "QPS"]
df.to_csv(f'results/5_2025_runs/15.3M_s30hc_m20_amazon_ecommerce_all_runs_voyage_binary_all_runs_sharded.csv')



