# QUIC-FL evaluation

This sub-project contains QUIC-FL's TensorFlow implementation  (`./quic_fl,py`).

## Setup

### Install requirements:

```setup
pip install -r requirements.txt
```

### Initialize git submodule

Run the following to make sure that the remote Google's [federated research repo](https://github.com/google-research/federated)
is cloned as a submodule:

```setup
git submodule update --init --recursive
```



### Update PYTHONPATH

Add `tf/google_research_federated` to `PYTHONPATH`.

## Training

In order to reproduce the paper's results, execute `trainer.py` (the current working directory should be the repo's root).

You can view the documentation for every command line parameter using `trainer.py --help`.

* `--root_output_dir` and `--experiment_name` flags determine where the outputs will be stored. 

* `--compressor` can be one of `quic-fl`, `eden`, `kashin`, `hadamard`, or `qsgd`, as described in the paper.

* `--num_bits` determines the number of integer bits to use.

* The rest of the parameters can be found in `cli_params_shakespeare.txt`.

You can monitor the progress using TensorBoard:

```setup
tensorboard --logdir <root_output_dir>/logdir
```

## Results

Execute `plots.ipynb` using [Jupyter](https://jupyter.org/) to re-create figures from the paper.