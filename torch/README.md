# QUIC-FL evaluation

This sub-project contains QUIC-FL's PyTorch implementation (`./quic_fl/quic_fl,py`).

## Setup

### Install requirements:

```setup
pip install -r requirements.txt
```

## Running NMSE experiments 

Run `nmse_test.py` with the following parameters to calculate the NMSE:

* `--alg` can be one of `quic-fl`, `eden`, `kashin`, `kashin-tf`, `hadamard`, `countsketch`, or `qsgd`, as described in the paper.

* `--nbits` determines the number of integer bits to use.

* `--d` determines the dimension of the clients' vectors.

* `--nclients` determines the number of clients/senders (for vNMSE set nclient=1).

* `--ntrials` determines the number of trials.

* `--seed` determines the seed for the PRNGs.

* `--rotation_seed` determines the rotation seed for Hadamard, Kashin and QUIC-FL.

