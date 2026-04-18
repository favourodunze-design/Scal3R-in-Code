# Scal3R Unpacked: Global Context, Fast Weights, and 3D Reconstruction

> A hands-on PyTorch walkthrough of **Scal3R** — the 2026 paper that reconstructs kilometer-scale 3D scenes from raw RGB video using test-time training as a memory mechanism.
>
> *Xie et al., "Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction", arXiv:2604.08542*

---

## What this is

Most papers on 3D reconstruction stop at the math. This notebook goes further — every core idea in Scal3R is translated into **runnable, annotated PyTorch code** with visualizations that make the concepts tangible.

If you've read the paper and wanted to see the moving parts, this is that.

---

## What's inside

| Section | What it covers |
|---------|---------------|
| **1 — The Problem** | Why O(N²) attention collapses on long sequences. Memory and compute plots across frame counts. |
| **2 — TTT as Memory** | Minimal implementation of the fast-weight update/apply loop. Watch the weight norm evolve token by token. |
| **3 — GCM Module** | Full `GlobalContextMemory` class with `AdaptiveMemoryUnit` (gated MLP, SiLU ∘ gate formulation). State-size verified against the paper's formula. |
| **4 — Gate Activation** | Heatmap showing how gate α responds to mid-sequence appearance changes across chunks. |
| **5 — GCS Simulation** | CPU simulation of Global Context Synchronization. Pose drift with and without cross-chunk gradient sharing. |
| **6 — Chunking Pipeline** | Visual breakdown of the overlapping window scheme (chunk=60, overlap=30) used at inference. |
| **7 — Benchmark Results** | Tables 1 & 2 reproduced as charts — KITTI, Oxford Spires, ETH3D across ATE, RTE, RRE, Chamfer Distance, F1. |
| **8 — Ablation Studies** | Table 3 reproduced — state size vs. accuracy, and the effect of removing GCM vs. GCS. |
| **9 — Runtime Scaling** | Linear fit to Table 5. FPS stability and RPE flatness as sequence length grows to 990 frames. |
| **10 — End-to-End Sketch** | A structurally faithful `Scal3RPipeline` class tying encoder → GCM → pose/depth heads into one forward pass. |

---

## Key concepts

**Test-Time Training (TTT) as memory**
Rather than compressing history into a fixed-size vector (like an RNN), Scal3R stores context in the *weights* of a small neural network that gets updated online via a self-supervised objective:

```
update:  W ← W − η ∇_W L(f_W(k), v)
apply:   o = f_W(q)
```

**Global Context Memory (GCM)**
A gated MLP module plugged into VGGT after layers 4, 11, 17, and 24. Its Adaptive Memory Units are updated chunk-by-chunk, giving the model a compact but expressive memory of the entire sequence seen so far.

**Global Context Synchronization (GCS)**
When running across multiple GPUs, each device computes its local AMU gradient. These are all-reduced and applied globally — so every chunk benefits from what every other chunk has seen.

---

## Requirements

```bash
pip install torch numpy matplotlib scipy
```

No GPU required — everything runs on CPU. A GPU will speed up Section 10's pipeline sketch but isn't necessary.

---

## Running it

Open `scal3r_concepts.ipynb` in Jupyter Lab, VS Code, or Google Colab and run cells top to bottom. Each section is self-contained after the setup cell (Section 0).

```bash
jupyter lab scal3r_concepts.ipynb
```

Or upload directly to [Google Colab](https://colab.research.google.com).

---

## A note on the implementation

This notebook is an **educational reconstruction**, not the official codebase. The goal is clarity over performance:

- VGGT's full transformer is replaced with a small stand-in encoder
- GCM and AMU are faithful to the paper's equations and architecture
- The chunking, GCS logic, and evaluation metrics are accurate
- Training datasets and multi-GPU setup are not reproduced

For the real thing: [zju3dv.github.io/scal3r](https://zju3dv.github.io/scal3r)

---

## Paper reference

```bibtex
@article{xie2026scal3r,
  title   = {Scal3R: Scalable Test-Time Training for Large-Scale 3D Reconstruction},
  author  = {Xie, Tao and Yang, Peishan and Jin, Yudong and Cai, Yingfeng and
             Yin, Wei and Ren, Weiqiang and Zhang, Qian and Hua, Wei and
             Peng, Sida and Guo, Xiaoyang and Zhou, Xiaowei},
  journal = {arXiv preprint arXiv:2604.08542},
  year    = {2026}
}
```

---

## License

MIT — do whatever you want with the code. If you build on it, a star or a mention is always appreciated.
