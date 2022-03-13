Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ git clone https://github.com/Howardhuang98/BNSL-doc.git

Load DataFrame
-----------------
use pandas

Choose your favorite estimator
------------------------------
**Hill climb search estimator**. It greedily search possible operations based on hill climb strategy. It is the most common used
score-based algorithm.

.. autoclass:: bnsl.estimators.HC

**Dynamic planning estimator**. It exhaustively searches every state graph, then uses shortest path algorithm to find optimal order,
finally recover the optimal DAG from order.

.. autoclass:: bnsl.estimators.DP

.. autoclass:: bnsl.estimators.GA

.. autoclass:: bnsl.estimators.PC







