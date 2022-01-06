# Deep Learning Bayesian Network

A Bayesian network structure learning package based on deep learning. Still developing ...  


## Here you can use:
* Greedy hill climb
* Simulated Annealing
* Dynamic program: shortest path perspective
* PC algorithm
* Genetic algorithm 

## Easily use

All the algorithms have been packed in ```dlbn.estimators```, like HC(Hill Climb). You only need to initialize the estimator with ```pd.Dataframe``` or ```np.ndarray```, then run it!
```Estimator.run()```will return you an enhanced ```DAG```instance, it inherited all attributions of ```nx.Digraph```, besides it can draw, calculate score criteria. 
```python
# import you favorite Estimator, like HC, SA, DP
from dlbn.estimators import DP
import pandas as pd
data = pd.read_excel(r"your data path")
# use DataFrame initialize the estimator
dp = DP(data)
# run it
dag = dp.run()
# dag instance stores all feature about the result. 
dag.show()
```



## Acknowledgement
Any question, feel free to email me!  
Author: Huang Hao    
School: Tianjin University, Priceless Lab  





