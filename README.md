# Cayley-Spinor Networks & QuatScan

Experimental research codebase, interactive HTML5 demo, and 4-page LaTeX paper exploring **wavy unitary orbits**, **quantum-mechanical atom neurons**, **trig-free rational Cayley $SU(2)$ spinors**, and **non-abelian quaternion sequence memory (`QuatScan`)**.

## Key Results (Verified)

| Experiment | Metric | Result |
| :--- | :--- | :--- |
| **8--16--4 Complex Net** | Test MSE | `wavy` ($n=3$, 1 param/edge): **`0.152465`** vs `float` (`0.162708`) vs `u1` (`0.262644`) |
| **16-Node Wavy Graph Pruning** | Test MSE | `100%` edges: `0.069346` $\rightarrow$ **`25%` edges (75% pruned): `0.069590`** (`+0.35%`) |
| **32-Node Holographic Recall** | Unseen Phase `rot_MSE` | **`harmonic_orbital`: `0.1590`** (`1,248`p) vs `real_tanh`: `0.7206` (`4,160`p) |
| **1-Shot Zero-Backprop Recall** | Test MSE / Phase | `u1_1shot` (`1.088 ms` storage): **`0.2790` / `0.8605`** vs `real_1shot_hop`: `0.7579` |
| **1-Atom QM Neuron on XOR** | Final MSE | `eigh`, `cayley`, `su2_pauli` (13p, `6.47 ms` / 10k vec): **`0.00000000`** |
| **2.5-Turn 2-Spirals (193p)** | Test Acc (600 steps) | `su2_spinor_net`: **`70.5%`**, `cayley_su2_fast`: **`68.0%`** vs `real_tanh_mlp`: `62.0%` |
| **`QuatScan` Order Task (`L=16`)** | Test Acc (no pos enc) | **`su2_quat_memory` (65p, $\sigma_{\text{init}}=0.1$): `100.0%`** vs `u1_commutative`: `53.5%` |

## Repository Structure

- `cayley_spinor_paper.tex` / `cayley_spinor_paper.pdf` — Full 4-page paper with derivations, tables, and 7 figures (`figures/`).
- `su2_seq_memory.py` — **QuatScan**: non-commutative $SU(2)$ quaternion sequence memory vs commutative $U(1)$ on `pos(A) < pos(B)` with zero positional encodings.
- `su2_spinor_net.py` — 16-neuron $SU(2)$ Pauli spinor network vs parameter-matched `tanh` MLP on 2.5-turn 2-spirals.
- `harmonic_orbital_memory.py` — 32-node per-neuron 2-harmonic Bohr-wave resonant graph with $U(1)$ gauge-equivariance test.
- `wavy_net_demo.html` — Live interactive HTML5/Canvas 16-node Wavy Complex Graph (polar orbit, graph edges, live MSE, lobe & pruning sliders).
- `dataset.txt` / `gen_append.py` — Clean ASCII facts & conversations dataset (`2,562` lines).

## Quickstart

```bash
pip install -r requirements.txt

# 1. Non-abelian sequence memory (100% vs 53.5%)
python su2_seq_memory.py

# 2. SU(2) spinor network on 2-spirals
python su2_spinor_net.py

# 3. 32-node holographic recall + unseen phase-shift test
python harmonic_orbital_memory.py

# 4. Compile paper (fast 2-pass)
pdflatex --enable-installer=false -interaction=nonstopmode cayley_spinor_paper.tex
pdflatex --enable-installer=false -interaction=nonstopmode cayley_spinor_paper.tex
```

Open `wavy_net_demo.html` in any browser to watch the 16-node wavy-orbit graph train live.

## Author & License

- **Author:** [Shubham Barmase (shu195)](https://github.com/shu195)
- **License:** [MIT](LICENSE)

## Citation

```bibtex
@misc{barmase2026cayleyspinor,
  author       = {Shubham Barmase},
  title        = {From Wavy Unitary Orbits to Trig-Free Cayley-Spinor Networks and Non-Abelian Sequence Memory},
  year         = {2026},
  howpublished = {\url{https://github.com/shu195/cayley-spinor-networks}}
}
```
