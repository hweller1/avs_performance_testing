
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from typing import Set

import pandas as pd
import numpy as np
import pymongo
import cohere 

co = cohere.Client("")


from companies import names


def compute_overlap(exact_result_set: Set, approx_result_set: Set) -> float:
	# each result set is a list of urls, order not considered
	return len(exact_result_set.intersection(approx_result_set)) / len(exact_result_set)


sentences = [f"What is {x}" for x in names]


embeddings = co.embed(texts=sentences, model="embed-multilingual-v3.0", input_type="search_document",).embeddings
# import pdb; pdb.set_trace()
# doc_emb = np.asarray(doc_emb)

# import pdb; pdb.set_trace()

mongo_uri = ""

client = pymongo.MongoClient(mongo_uri)

db = client['sample_vectors']

# db = client['test']

# collections = [db['sphere10mm']]

# num_vecs = ['100k', '1M']
num_vecs = ['10M']
# num_vecs = ['1M']
# num_vecs = ['100k']


# collections = [db['sphere100k'], db['sphere1mm']]
# collections = [db['sphere10m']]
collections = [db['cohere_wikipedia_multilingual_v3']]
# collections = [db['sphere100k']]


# indexes = ['1M_sphere_index']

# indexes = ['10M_sphere_index']
# indexes = ['100k_sphere_index', '1M_sphere_index']

indexes = ['vector_index']

# unfiltered_exact_search_results = ['1M_exact_search_results.json']

# unfiltered_exact_search_results = ['100k_exact_search_results.json']

# unfiltered_exact_search_results = ['10M_exact_search_results.json']
# unfiltered_exact_search_results = ['100k_exact_search_results.json', '1M_exact_search_results.json']

# filtered_exact_search_results = {'filtered_low_card':'1M_lowcard_exact_search_results.json',
# 								 'filtered_high_card': '1M_highcard_exact_search_results.json',
# 								 'filtered_multicard': '1M_multi_lowcard_exact_search_results.json'}

# filtered_exact_search_results = {'filtered_low_card':'1M_lowcard_exact_search_results.json',
# 								 'filtered_high_card': '1M_highcard_exact_search_results.json',
# 								 'filtered_multicard': '1M_multi_lowcard_exact_search_results.json'}



# k_values = [1, 10, 100]
# candidate_values = [64, 100, 1000]

k_values = [10, 100]

# multipliers = [10] + list(range(60, 100, 10))

multipliers = [1] + list(range(10, 100, 10))


# multipliers = [10, 20, 100]

# candidate_values = [100, 2000]
num_exact_results = len(sentences)



def build_query(indexes, embedding, candidates, filter_clause, exact=False):

	if filter_clause is None:
		if exact:
			query = [
					{
			        "$vectorSearch": {
			            "index":f'{indexes[i_0]}',
			            "path": "vector",
			            "queryVector": embedding,
			            "limit": k,
			            "exact": True,
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
				            "queryVector": embedding,
				            "limit": k,
				            "numCandidates": candidates,
				        },
				        },
				        {
			    		"$project": {"vector": 0}
				        }
				    ]
	else:	
		if exact:
			query = [
						{
				        "$vectorSearch": {
				            "index":f'{indexes[i_0]}',
				            "path": "vector",
				            "queryVector": embedding,
				            "limit": k,
				            "exact": True,
				            "filter": filter_clause
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
				            "queryVector": embedding,
				            "limit": k,
				            "numCandidates": candidates,
				            "filter": filter_clause
				        },
				        },
				        {
			    		"$project": {"vector": 0}
				        }
				    ]
	return query 


def run_query(embedding, k, candidates, filter_clause, exact_result_dict, query_index):
	if filter_clause:
		filter_key = list(filter_clause.keys())[0]
	else:
		filter_key = None
	if (query_index, k, filter_key) not in exact_result_dict.keys():
		query = build_query(indexes, embedding, candidates, filter_clause, True)
		start = time.time()
		# print("running exact search queries")
		x = coll.aggregate(query)
		results = []
		for result_index in range(k):
			try:
				results.append(x.next()['title'])
			except:
				break
		end = time.time()
	# print(f"query {i}: {sentences[i]}\n")
	# print(f"results are {results}\n")
	# print(f"exact results are {exact_url_list}\n")

		results = set(results)
		exact_result_dict[(query_index, k, filter_key)] = results
		# import pdb; pdb.set_trace()


	query = build_query(indexes, embedding, candidates, filter_clause)
	start = time.time()
	x = coll.aggregate(query)
	results = []
	# print("running ann queries")
	for result_index in range(k):
		try:
			# print(f"running query {result_index}/{k}")
			results.append(x.next()['title'])
		except:
			break
	end = time.time()
	# import pdb; pdb.set_trace()
	# print(f"query {i}: {sentences[i]}\n")
	# print(f"results are {results}\n")
	# print(f"exact results are {exact_url_list}\n")
	results = set(results)
	overlap = compute_overlap(exact_result_dict[(query_index, k, filter_key)], results)

	return overlap, end - start


for mult in multipliers:
	plot_results = {}

	print(f"running mult {mult}\n")
	for i_0, coll in enumerate(collections):
		if coll == db['sphere1mm']:
			# tests = [[]'filtered_low_card', 'filtered_high_card', 'filtered_multicard', 'concurrency_10', 'concurrency_100']
			# tests = ['no_filter', 'filtered_low_card', 'filtered_high_card', 'filtered_multicard', 'concurrency_10', 'concurrency_100']
			tests = ['no_filter']
		else:
			# tests = ['no_filter', 'concurrency_10', 'concurrency_100']
			# tests = ['concurrency_100']
			# tests = ['no_filter', 'filtered_low_card', 'filtered_high_card', 'concurrency_10', 'concurrency_100']
			tests = ['no_filter', 'concurrency_10', 'concurrency_100']



		for test in tests:
			print(f"running test {num_vecs[i_0]} {test}\n")
			if 'filtered' in test:
				exact_results = None
				# exact_results = json.load(open(f'exact_results/{filtered_exact_search_results[test]}'))
				if "low" in test:
					filter_type = 'Low Cardinality'
					filter_clause = {'lowCardinalityCategoryInt': {'$eq': 1}}
				elif "high" in test:
					filter_type = 'High Cardinality'
					filter_clause = {'highCardinalityCategoryInt': {'$eq': 1}}
			else:
				filter_type = 'Unfiltered'
				exact_results = None
				# exact_results = json.load(open(f'exact_results/{unfiltered_exact_search_results[i_0]}'))
				filter_clause = None
			if 'concurrency' in test:
				concurrency = int(test.split('_')[1])
			else:
				concurrency = 1

			original_concurrency = None
			for i_1, k in enumerate(k_values): 
				print(f"running k = {k}")
				times = []
				recalls = []
				futures = []

				candidates = k * mult

				if original_concurrency is not None:
					concurrency = original_concurrency

				elif concurrency > candidates:
					original_concurrency = copy(concurrency)
					concurrency = candidates
				# import pdb; pdb.set_trace()


				executor = ThreadPoolExecutor(max_workers=concurrency)
				num_embeddings = len(embeddings)
				for i, embedding in enumerate(embeddings):
					print(f"running query {i}/{num_embeddings}")
					exact_result_dict = {}
					
					if concurrency == 1:

						overlap, query_time = run_query(embedding=embedding,
												        k=k,
												        candidates=candidates,
												        filter_clause=filter_clause,
												        exact_result_dict=exact_result_dict, query_index=i)

						recalls.append(overlap)
						times.append(query_time)
					else:
						f = executor.submit(run_query, embedding, k, candidates, filter_clause, exact_result_dict, i)
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
				plot_results[f"{num_vecs[i_0]}.k{k}.{test}.concurrency{concurrency}"] = {
			 	   "Limit, numCandidates": (k, candidates),
			 	   "Concurrent Requests": concurrency,
			 	   "Filter Type": filter_type,
					"Recall": np.mean(recalls) * 100,
				 	"Mean Latency": np.mean(times) * 1000,
			 	   # "p99 Latency": np.percentile(times, 99),
			 	    "QPS": qps}
# import pdb; pdb.set_trace()


	flattened_list = [(outer_key, *inner_dict.values()) for outer_key, inner_dict in plot_results.items()]
	df = pd.DataFrame(flattened_list)
	# import pdb; pdb.set_trace()
	df.columns = ["Test Case", "Limit, numCandidates", "Concurrent Requests", "Filter Type", "Recall", "Mean Latency (ms)", "QPS"]
	df.to_csv(f'results/12_2024_runs/cohere_unfiltered_{mult}x_mult.csv')

# with open(f"results/performance_test_results_.json", "w") as outfile:
# 	json.dump(plot_results, outfile)

# with open(f"results/performance_test_results_1M_2_shards_id_shard_key.json", "w") as outfile:
# 	json.dump(plot_results, outfile)



