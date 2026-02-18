# Active Deep Decoding with Importance Sampling

Efficient active deep decoding of linear codes using importance sampling and weighted belief propagation.

This repository provides a reproducible TensorFlow implementation of the framework proposed in:

> **Efficient Active Deep Decoding of Linear Codes Using Importance Sampling**
> IEEE Communications Letters, 2025.

The code implements active weighted belief propagation (WBP) training combined with importance sampling (IS) to improve decoding performance in the high-SNR regime.

---

## Overview

This repository contains:

* Weighted belief propagation (WBP) decoder
* Active training procedure
* Importance sampling–based noise generation
* Monte-Carlo BER/FER simulation framework
* Utilities for ALIST → parity-check matrix conversion

The implementation supports arbitrary linear block codes defined by a parity-check matrix.

---

## Repository Structure

```
codes/              # ALIST / parity-check matrix files
Alist2bin.py        # Convert ALIST to binary PCM
BPSio.py            # LDPC / Weighted BP decoder layer
Gnoise.py           # Gaussian + Importance Sampling noise generation
train.py            # Training script
sim.py              # BER/FER simulation script
LICENSE
README.md
```

---

## Requirements

Python 3.8+ recommended.

Main dependencies:

```
tensorflow >= 2.8
numpy
scipy
matplotlib
```

Install with:

```bash
pip install tensorflow numpy scipy matplotlib
```

---

## Training

The training script implements active learning with importance sampling.

Example:

```bash
python train.py
```

Key configurable parameters inside `train.py`:

* `min_snr`, `max_snr`
* `batch_size`
* `M` (importance sampling grid size)
* `num_itr` (number of BP iterations)

Models are saved in the `models/` directory.

---

## Simulation (BER / FER)

To evaluate a trained model:

```bash
python sim.py
```

This runs Monte-Carlo simulation over an SNR range and outputs:

* BER curves
* FER curves
* CSV result files
* Plots saved under `figs/`

The simulation compares:

* Active weighted BP (trained model)
* Standard BP decoding

---

## Using Custom Codes

To use a custom code:

1. Place the ALIST file in `codes/`
2. Load it using:

```python
from Alist2bin import Alist2bin
pcm = Alist2bin("codes/your_code.alist")
```

Or load `.mat` files as shown in `sim.py`.

---

## Reproducibility Notes

* All simulations assume BPSK modulation.
* All-zero codeword transmission is used (standard in coding simulations).
* Random seeds can be added for deterministic behavior.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ActiveDeepDecoding2025,
  title={Efficient Active Deep Decoding of Linear Codes Using Importance Sampling},
  journal={IEEE Communications Letters},
  year={2025}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions, suggestions, or collaboration inquiries, please open a GitHub issue.

For direct communication, please email:
hnksm@connect.ust.hk



