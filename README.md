# QUIC-FL: **Q**uick **U**nb**i**ased **C**ompression for **F**ederated **L**earning

This repository contains the official QUIC-FL implementation as presented in 'Accelerating Federated Learning with Quick Distributed Mean Estimation' (ICML 2024).

QUIC-FL is a lossy unbiased compression technique for distributed mean estimation that achieves the optimal $O(1/n)$ Normalized Mean Squared Error (NMSE) guarantee 
and has an asymptotically faster encoding and decoding complexity than existing methods.

QUIC-FL is particularly well-suited for Federated Learning (FL) settings, where the server iteratively aggregates model updates from multiple clients.

## Folder structure 

The `torch` and `tf` folders contain QUIC-FL's implementation in PyTorch and TensorFlow, respectively.

## Citation

If you find this useful, please cite us:

```bibtex
@inproceedings{ben-basat2024accelerating,
  title={Accelerating Federated Learning with Quick Distributed Mean Estimation},
  author={Ran Ben-Basat and Shay Vargaftik and Amit Portnoy and Gil Einziger and Yaniv Ben-Itzhak and Michael Mitzenmacher},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=gWEwIlZrbQ}
}
```
