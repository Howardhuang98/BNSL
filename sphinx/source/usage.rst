Usage
=====

.. _installation:

Installation
------------

To use BNSL, first download it using git:

.. code-block:: console

   (.venv) $ git clone https://github.com/Howardhuang98/BNSL.git

Load Data
-----------------
`Pandas <https://pandas.pydata.org/>`_ is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, we use pandas to manipulate data. 

.. tip:: In the whole library, we use **pd.DataFrame** as data input, don't change it into any other format. 

.. code-block:: python

    import pandas as pd
    data = pd.read_csv(r"./datasets/asian/Asian.csv")


Choose your favorite estimator
------------------------------
BNSL provides several up-to-dated Bayesian network structure learning estimator, e.g., Hill climb, genetic algorithm... Let's take **Hill climb estimator** for example. it greedily search possible operations based on hill climb strategy. It is the most common used
score-based algorithm.

.. code-block:: python

    from bnsl.estimators import HC
    hc = HC(data)

Congratulations, you have initialized an estimator.

Run the estimator
------------------

Every estiamtor has function ``run()``, use it straightly with default configuration.  

.. code-block:: python

    hc.run()
    print("Structure learning Done!")
  

Visualize and store dag result
-------------------------------

After running, the estimator will save the ``DAG`` instance in ``hc.result``. Then you can use ``show()`` to visualize the dag, and ``to_csv()`` to story the dag.  

.. code-block:: python

    dag = hc.result
    dag.show()
    dag.to_csv(path=r"./asian_dag.csv")







