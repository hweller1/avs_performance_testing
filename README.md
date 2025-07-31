# avs_performance_testing

Run Atlas Vector Search under various conditions to assess performance. 

Configurable parameters include:
- Filtering
- numCandidates
- Limit
- Request concurrency

Future Improvements:
- Factor out test cases into a config file that can be passed to run.py via a CLI
- 100M vector performance testing

## Official Benchmark

All test results provided in the [Atlas Vector Search Benchmark](https://www.mongodb.com/docs/atlas/atlas-vector-search/benchmark/) were produced using the scripts titled run_amazon_ecommerce_voyage_15m.py and run_amazon_ecommerce_voyage_multidim.py.

These run scripts use [this](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset embedded with voyage-3-large assessing scalar and binary quantized indexes produced using save_voyage_embeddings. The multidimensional script issues queries against an index with 4 different dimensionalities of that embedding model, produced by building a view that slices the original embeddings (detailed [here](https://www.mongodb.com/docs/atlas/atlas-vector-search/benchmark/overview/#vector-dimensionality)).


## Original scripts

Original run.py script use [sphere dataset](https://ai.meta.com/blog/introducing-sphere-meta-ais-web-scale-corpus-for-better-knowledge-intensive-nlp/).

Cohere run script uses [cohere wikipedia dataset](https://cohere.com/blog/embedding-archives-wikipedia).

Jina/amazon script uses [this](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset embedded with jina-embeddings-v3(https://huggingface.co/jinaai/jina-embeddings-v3) and an index that is binary quantized, using saved exact results at 1M and 17M vectors. This also uses a range filter instead of a point filter as the sphere dataset tests used.
