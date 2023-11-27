
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from typing import Set

import numpy as np
import pymongo
from sentence_transformers import SentenceTransformer

from companies import names


def compute_overlap(exact_result_set: Set, approx_result_set: Set) -> float:
	# each result set is a list of urls, order not considered
	return len(exact_result_set.intersection(approx_result_set)) / len(exact_result_set)


embeddings_list = []

sentences = [f"What is {x}" for x in names]


model = SentenceTransformer('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
embeddings = model.encode(sentences)

# import pdb; pdb.set_trace()

client = pymongo.MongoClient("") # mongodb cluster URI

db = client['vector-test']

# collections = [db['sphere10mm']]

# num_vecs = ['100k', '1M']
# num_vecs = ['10M']
num_vecs = ['1M']
# num_vecs = ['100k']


# collections = [db['sphere100k'], db['sphere1mm']]
# collections = [db['sphere10mm']]
collections = [db['sphere1mm']]
# collections = [db['sphere100k']]


indexes = ['1M_sphere_index']

# indexes = ['10M_sphere_index']
# indexes = ['100k_sphere_index', '1M_sphere_index']

# indexes = ['100k_sphere_index']

unfiltered_exact_search_results = ['1M_exact_search_results.json']

# unfiltered_exact_search_results = ['100k_exact_search_results.json']

# unfiltered_exact_search_results = ['10M_exact_search_results.json']
# unfiltered_exact_search_results = ['100k_exact_search_results.json', '1M_exact_search_results.json']

filtered_exact_search_results = {'filtered_low_card':'1M_lowcard_exact_search_results.json',
								 'filtered_high_card': '1M_highcard_exact_search_results.json',
								 'filtered_multicard': '1M_multi_lowcard_exact_search_results.json'}

k_values = [1, 10, 100]
candidate_values = [64, 100, 1000]
plot_results = {}
num_exact_results = 100



def run_query(embedding, k, candidates, filter_clause, exact_url_list):
	if filter_clause is None:
		query = [
					{
			        "$vectorSearch": {
			            "index":f'{indexes[i_0]}',
			            "path": "vector",
			            "queryVector": embedding.tolist(),
			            "limit": k,
			            "numCandidates": candidates,
			        },
			        },
			        {
		    		"$project": {"vector": 0}
			        }
			    ]
	else:	
		query = [
					{
			        "$vectorSearch": {
			            "index":f'{indexes[i_0]}',
			            "path": "vector",
			            "queryVector": embedding.tolist(),
			            "limit": k,
			            "numCandidates": candidates,
			            "filter": filter_clause
			        },
			        },
			        {
		    		"$project": {"vector": 0}
			        }
			    ]

	start = time.time()
	x = coll.aggregate(query)
	results = []
	for result_index in range(k):
		try:
			results.append(x.next()['url'])
		except:
			break
	end = time.time()
	# import pdb; pdb.set_trace()
	print(f"query {i} with project took {end - start} seconds\n")
	# print(f"results are {results}\n")
	# print(f"exact results are {exact_url_list}\n")

	results = set(results)
	overlap = compute_overlap(exact_url_list, results)


	return overlap, end - start



for i_0, coll in enumerate(collections):
	if coll == db['sphere1mm']:
		tests = ['no_filter', 'filtered_low_card', 'filtered_high_card', 'filtered_multicard', 'concurrency_10', 'concurrency_100']
		# tests = ['no_filter']
	else:
		tests = ['no_filter', 'concurrency_10', 'concurrency_100']


	for test in tests:
		print(f"running test {num_vecs[i_0]} {test}\n")
		if 'filtered' in test:
			exact_results = json.load(open(f'exact_results/{filtered_exact_search_results[test]}'))
			if "low" in test:
				filter_clause = {'low_card': {'$eq': 1}}
			elif "high" in test:
				filter_clause = {'high_card': {'$eq': 1}}
			else:
				filter_clause = {"$and": [
					{'low_card': {'$eq': 1}},
					{'low_card_1': {'$eq': 1}},
					{'low_card_2': {'$eq': 1}}]
					}
		else:
			exact_results = json.load(open(f'exact_results/{unfiltered_exact_search_results[i_0]}'))
			filter_clause = None
		if 'concurrency' in test:
			concurrency = int(test.split('_')[1])
		else:
			concurrency = 1

		original_concurrency = None
		for i_1, k in enumerate(k_values): 
			times = []
			recalls = []
			futures = []

			candidates = candidate_values[i_1]

			if original_concurrency is not None:
				concurrency = original_concurrency

			elif concurrency > candidates:
				original_concurrency = copy(concurrency)
				concurrency = candidates
			# import pdb; pdb.set_trace()


			executor = ThreadPoolExecutor(max_workers=concurrency)

			for i, embedding in enumerate(embeddings):

				if i > num_exact_results:
					break
				# import pdb; pdb.set_trace()
				
				exact_url_list = set(exact_results[f"{sentences[i]}.{k}"])

				if concurrency == 1:

					overlap, query_time = run_query(embedding=embedding,
											        k=k,
											        candidates=candidates,
											        filter_clause=filter_clause,
											        exact_url_list=exact_url_list)

					recalls.append(overlap)
					times.append(query_time)
				else:
					f = executor.submit(run_query, embedding, k, candidates, filter_clause, exact_url_list)
					futures.append(f)
				# import pdb; pdb.set_Trace()
			start_time = time.time()
			if concurrency != 1:

				for f in as_completed(futures):
					overlap, query_time = f.result()
					recalls.append(overlap)
					times.append(query_time)
				# import pdb; pdb.set_trace()
				total_time = time.time() - start_time 
				qps = num_exact_results / total_time
			else:
				qps = num_exact_results / np.sum(times)

			# import pdb; pdb.set_trace()
			plot_results[f"{num_vecs[i_0]}.k{k}.{test}.concurrency{concurrency}"] = {"Recall": np.mean(recalls) * 100, "Mean Latency": np.mean(times), "p99 Latency": np.percentile(times, 99), "QPS": qps}

with open(f"results/performance_test_results_1M_2_shards_id_shard_key.json", "w") as outfile:
	json.dump(plot_results, outfile)

# with open(f"results/performance_test_results_1M_2_shards_id_shard_key.json", "w") as outfile:
# 	json.dump(plot_results, outfile)



