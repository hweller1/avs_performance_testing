# avs_performance_testing

Implements performance tests according to spec [here](https://docs.google.com/document/d/12TWR07_xx0VFkRL4Ie-MXcfEoHYf8qtf8ChLFiytjdk/edit#heading=h.chhxb36ff1ko).

Results are collated [here](https://docs.google.com/document/d/1dsulEFhYwj3MufTG3YM-ga94tBAxk8o8ID2iksOmYa4/edit)


Future Improvements (please suggest any you think would be good to perform in a PR):
- Factor out test cases into a config file that can be passed to run.py via a CLI
- Assess accuracy of single shard candidates = 200 against 2 shards candidates = 100
- 100M vector performance testing
- 768 vs 1536 dim performance testing (to assess wall time as dimensionality increases)
- More configurations of Limit, numCandidates 
	- 10,100; 25,250; 50,500; 100,1000; 200,2000
	- different ratio than 1:10
