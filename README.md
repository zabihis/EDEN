
---

# EDEN: Multiscale Expected Density of Nucleotide Encoding for Enhanced DNA Sequence Classification with Hybrid Deep Learning

## 🧬 Abstract

**Background:** DNA sequences are fundamental carriers of genetic information. Accurate classification is essential for understanding gene regulation and disease mechanisms. Existing encoding methods often struggle to capture both local and long-range dependencies simultaneously.

**Results:** We introduce **EDEN (Expected Density of Nucleotide Encoding)**, a unified multiscale encoding framework based on kernel density estimation. EDEN captures position-specific and context-dependent nucleotide patterns and integrates them into a hybrid deep learning architecture. Across sixteen benchmark datasets, EDEN achieves state-of-the-art performance with significantly fewer parameters than competing models.

**Conclusions:** EDEN provides an efficient, biologically informed representation for genomic sequence classification, demonstrating high practicality for large-scale applications.

---

## 🛠 Project Structure

The repository is modularized for academic reproducibility and clear separation of concerns:

* **`models.py`**: Implementation of the `Hybrid_CNN` architecture (Dual-branch CNN).
* **`utils.py`**: Core engine for **Multiscale EDN Encoding**, data loading, and evaluation metrics.
* **`predict.py`**: Command-line interface (CLI) for performing inference on datasets.
* **`datasets/`**: Genomic benchmark data in CSV format.
* **`models/`**: Pretrained `.pth` weights for various genomic tasks.

---

## 🚀 Installation & Usage

### 1. Requirements

Install the necessary Python packages using pip:

```bash
pip install torch pandas numpy scikit-learn

```

### 2. Running Inference (CLI)

The `predict.py` script allows you to run predictions for specific datasets directly from the terminal.

**Basic Syntax:**

```bash
python predict.py --dataset <dataset_folder_name> --limit <integer_value>

```

**Examples:**

* **Core Promoter Detection (TATA):**
```bash
python predict.py --dataset human_prom_core_tata --limit 70

```


* **Transcription Factor Binding (TF0):**
```bash
python predict.py --dataset human_tf0 --limit 100

```



---

## 📊 CLI Parameters Table

To run the CLI, use the following parameters:

| Dataset Category | `--dataset` (Parameter) | `--limit` (Parameter) |
| --- | --- | --- |
| **Core Promoter** | `human_prom_core_all` | **70** |
| **Core Promoter** | `human_prom_core_notata` | **70** |
| **Core Promoter** | `human_prom_core_tata` | **70** |
| **Promoter (300bp)** | `human_prom_300_all` | **300** |
| **Promoter (300bp)**| `human_prom_300_notata` | **300** |
| **Promoter (300bp)** | `human_prom_300_tata` | **300** |
| **TF Binding (Human)** | `human_tf0` | **100** |
| **TF Binding (Human)** | `human_tf1` | **100** |
| **TF Binding (Human)** | `human_tf2` | **100** |
| **TF Binding (Human)** | `human_tf3` | **100** |
| **TF Binding (Human)** | `human_tf4` | **100** |
| **TF Binding (Mouse)** | `mouse_tf0` | **100** |
| **TF Binding (Mouse)** | `mouse_tf1` | **100** |
| **TF Binding (Mouse)** | `mouse_tf2` | **100** |
| **TF Binding (Mouse)** | `mouse_tf3` | **100** |
| **TF Binding (Mouse)** | `mouse_tf4` | **100** |

---

## 📧 Contact

* **Saman Zabihi**
* **Email:** szabihi@hotmail.com
* **GitHub:** [https://github.com/zabihis/EDEN](https://github.com/zabihis/EDEN)

---

## 📚 Citation

If you use EDEN in your research, please cite our work:

```bibtex
@article{zabihi2026,
  title={EDEN: Multiscale Expected Density of Nucleotide Encoding for Enhanced DNA Sequence Classification with Hybrid Deep Learning},
  author={Zabihi, Saman and Hashemi, Sattar and Mansoori, Eghbal},
  journal={},
  year={2026},
  url={https://github.com/zabihis/EDEN}
}

```

---
