# avs_performance_testing

Run Atlas Vector Search under various conditions to assess performance. 


Original run.py script use [sphere dataset](https://ai.meta.com/blog/introducing-sphere-meta-ais-web-scale-corpus-for-better-knowledge-intensive-nlp/).

Cohere run script uses [cohere wikipedia dataset](https://cohere.com/blog/embedding-archives-wikipedia).

Jina/amazon script uses [this](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset embedded with jina-embeddings-v3(https://huggingface.co/jinaai/jina-embeddings-v3) and an index that is binary quantized, using saved exact results at 1M and 17M vectors. This also uses a range filter instead of a point filter as the sphere dataset tests ued.

Configurable parameters include:
- Filtering
- numCandidates
- Limit
- Size of random query test set (if 'random' test case is configured)
- Request concurrency

Future Improvements:
- Factor out test cases into a config file that can be passed to run.py via a CLI
- 100M vector performance testing
