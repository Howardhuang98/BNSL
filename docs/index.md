# Deep Learning Bayesian Network

A Bayesian network structure learning package based on deep learning. Still developing ...  


## Here you can use:
* Hill climb
* Simulated annealing
* Dynamic program: shortest path perspective
* PC algorithm
* Genetic algorithm
* K2 algorithm

## Easily use
### Observed data
Directly use `pd.DataFrame` as observed data.
### DAG
`dlbn.graph.DAG` class is an enhanced `nx.Digraph`, thus you can initialize a DAG instance as you do it with `networkx`. Please check networkx's document to learn the basic operation of DAG.   
> networkx.org    

Some enhanced methods are added:  

`dlbn.graph.DAG.summary` print a summary of the DAG.  
`dlbn,graph.DAG.show` draw the DAG.  
`dlbn.graph.DAG.score` return the score of the total DAG, e.g. BIC score.
`dlbn.graph.DAG.read` read a .csv or .xlsx file.  
more methods please check source code.  


### Run your favorite estimator
All the algorithms have been packed in ```dlbn.estimators```, like HC(Hill Climb). You only need to initialize the estimator with ```pd.Dataframe``` or ```np.ndarray```, then run it!
```Estimator.run()```will return you an enhanced ```DAG```instance, it inherited all attributions of ```nx.Digraph```, besides it can draw, calculate score criteria.

```python
# import you favorite Estimator, like HC, SA, DP
from bnsl.estimators import DP
import pandas as pd
data = pd.read_excel(r"your data path")
# use DataFrame initialize the estimator
dp = DP(data)
# run it
dag = dp.run()
# dag instance stores all feature about the old_result. 
dag.show()
```



## Acknowledgement
  





