# mlex_dimension_reduction
Dimension reduction algorithms for the MLExchange platform. 
  - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  - [UMAP](https://umap-learn.readthedocs.io/en/latest/)

## Getting started
To get started, you will need:
  - [Docker](https://docs.docker.com/get-docker/)

## Running
First, build the dimension reduction image in terminal:
`cd mlex_dimension_reduction`
`make build_docker`

Once built, you can run the following examples:
`make UMAP_example`
which is equivalend to first `make run_docker` then `python umap_run.py example_umap.yaml`.

These examples utilize the information stored in the folder /data. The computed latent vectors will be saved in data/output.

## Developer Setup
If you are developing this library, there are a few things to note.

1. Install development dependencies:

```
pip install .
pip install ".[dev]"
```

2. Install pre-commit
This step will setup the pre-commit package. After this, commits will get run against flake8, black, isort.

```
pre-commit install
```

3. (Optional) If you want to check what pre-commit would do before commiting, you can run:

```
pre-commit run --all-files
```

4. To run test cases:

```
python -m pytest
```

## Copyright
MLExchange Copyright (c) 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
