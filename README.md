# avs_performance_testing

Run Atlas Vector Search under various conditions to assess performance against the [sphere dataset](https://ai.meta.com/blog/introducing-sphere-meta-ais-web-scale-corpus-for-better-knowledge-intensive-nlp/)


Future Improvements (please suggest any you think would be good to perform in a PR):
- Factor out test cases into a config file that can be passed to run.py via a CLI
- Assess accuracy of single shard candidates = 200 against 2 shards candidates = 100
- 100M vector performance testing
- 768 vs 1536 dim performance testing (to assess wall time as dimensionality increases)
- More configurations of Limit, numCandidates 
	- 10,100; 25,250; 50,500; 100,1000; 200,2000
	- different ratio than 1:10
- Experiment with different types of filters
	- range filtering
	- different types
- Experiment with what concurrency looks like with more search nodes in a replica set 


12/10/24 UPDATE

I've added another file which allows for running different test configs against the [cohere wikipedia dataset](https://cohere.com/blog/embedding-archives-wikipedia)
