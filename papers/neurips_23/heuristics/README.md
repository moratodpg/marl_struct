# Expert-based heuristic policies

To find the best expert-based heuristic policy, one must conduct a search over the possible inspection intervals and number of components to be inspected.

This is done via the [run_heuristics](run_heuristics.py) script.

This script takes as input via its first lines the parameters of the environments and the parameters of the heuristic search.

Execute the script `download_heuristic_logs.sh` to retrieve the logs of the experiments conducted in the paper.

**Reproduce the results:** 
1. Run again the policy search to identify the optimised heuristics.
 The policy search can be executed by indicating `search = True` in `run_heuristics.py`. 
2. To re-run the policy evaluation corresponding to the optimised heuristics.
 In this case, specify `search = False` in `run_heuristics.py`, and the best heuristic parameters in the eval function.
 These parameters of the stored policies, namely the inspection interval and number of components to inspect, are stored in the `heur_search` folder of the logs.

For example, to compute the return resulting from the uncorrelated 4-out-of-5 environment:

* Check the optimized heuristics, providing 'insp_interv': 10, 'insp_comp': 5.
 ```
 with np.load("../heur_search/results_struct_uc/heuristics_5_4ucref_2023_04_15_130930.npz", allow_pickle=True) as data:
     print(data["opt_heur"])
 ```
 
* The seed was set up as 0 by default
* Execute run_heuristics.py
