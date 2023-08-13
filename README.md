# COATI: multi-modal contrastive pre-training for representing and traversing chemical space

![](toc_graphic.png)

# Overview

COATI (Contrastive Optimization for Accelerated Therapeutic Inference) is a pre-trained, multi-modal encoder-decoder model of druglike chemical space. COATI is constructed without any human biasing of features, using contrastive learning from text and 3D representations of molecules to allow downstream use with structural models. COATI possesses many of the desired properties of a universal molecular embedding: fixed-dimension, invertibility, autoencoding, accurate regression, and low computation cost. 

COATI can also be used in a metadynamics algorithm for generative optimization for designing molecules that satisfy the multi-parameter optimization task of potency, solubility, and druglikeness.

# Installation

Before installation, we recommend to first setup a Python virtual environment.

```bash
$ git clone https://github.com/terraytherapeutics/COATI.git
$ cd COATI
$ pip install -e .
```

# Examples

- Basic molecule generation: `examples/simple_mol_generation/generation_examples.ipynb`.
- [DUE model](https://arxiv.org/abs/2102.11409) regression: `examples/simple_mol_generation/due_model_regression.ipynb`.
- Molecule generation analysis with Chembl dataset: `examples/simple_mol_generation/chembl_analysis.ipynb`.
- Metadynamics approach to molecular design: `examples/metadynamics/metadynamics.ipynb`.

# Models in the paper

The various models in the paper can be initialied using, e.g.,
```
encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
    freeze=True,
    device=torch.device("cuda:0"),
    # model parameters to load.
    doc_url=<model path>,
)
```
where we provide the following models (`doc_url`) with labels corresponding to their description in the paper:
```
s3://terray-public/models/tall_closed.pkl
s3://terray-public/models/grande_closed.pkl
s3://terray-public/models/grade_closed_fp.pkl
s3://terray-public/models/barlow_closed_fp.pkl
s3://terray-public/models/barlow_closed.pkl
s3://terray-public/models/autoreg_only.pkl
s3://terray-public/models/barlow_venti.pkl
s3://terray-public/models/grande_open.pkl
s3://terray-public/models/selfies_barlow.pkl
```


# Cite
