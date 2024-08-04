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
@InProceedings{pmlr-v235-ben-basat24a,
  title = 	 {Accelerating Federated Learning with Quick Distributed Mean Estimation},
  author =       {Ben-Basat, Ran and Vargaftik, Shay and Portnoy, Amit and Einziger, Gil and Ben-Itzhak, Yaniv and Mitzenmacher, Michael},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {3410--3442},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/ben-basat24a.html}
}
```
