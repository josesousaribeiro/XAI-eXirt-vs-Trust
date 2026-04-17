# XAI-eXirt-vs-Trust
Este repositório possui o código desenvolvido no artigo: Exploring Reliability in XAI: Do Model Explanations Depend on Data or Algorithms?


# Instalação

Criação de um ambiente anaconda com versão específica do python:

```
conda create -n env_xai python=3.10.14
```


Ativação do ambiente:

```
conda activate env_xai
```

Instalação de dependências 
``` 
conda install --yes --file requirements_conda.txt

pip install -r requirements_pip.txt
```

Execução pipeline principal:

```
python execute_main.py

```

Execução da criação de ranques:
```
python execute_ranks_comparations.py
```

Execução de análises estatísticas:
```
python execute_statistical_test.py
```

