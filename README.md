# XAI-eXirt-vs-Trust
This repository contains the code developed in the article: Exploring Reliability in XAI: Do Model Explanations Depend on Data or Algorithms?

# Results of applying the pipeline to 4 different datasets

Link: [summary_menu.html](summary_menu.html)

# Installation

Creating an Anaconda environment with a specific Python version:

```
conda create -n env_xai python=3.10.14

```

Activating the environment:

```
conda activate env_xai

```

Installing dependencies:

``` conda install --yes --file requirements_conda.txt

pip install -r requirements_pip.txt

```

Running the main pipeline:

```
python execute_main.py

```

Running the rankings:

```
python execute_ranks_comparisons.py

```

Running statistical analyses:

```
python execute_statistical_test.py
```