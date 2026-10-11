# Code for MSeR²

## Abstract

Drug repositioning plays a critical role in accelerating drug discovery by identifying new therapeutic indications for approved drugs. However, existing methods often collapse heterogeneous relations into a unified representation space, which may obscure relation-specific semantics and weaken representation discriminability. Moreover, semantic view dependencies vary across nodes, challenging conventional uniform multi-view fusion strategies. To address these limitations, we propose MSeR², an adaptive multi-view semantic graph learning framework for drug repositioning. MSeR² first initializes entity representations from attribute information and constructs multiple meta-path-driven semantic subgraphs. It learns view-specific representations through neighborhood aggregation and multi-hop propagation, capturing local structural semantics and global pattern dependencies. An adaptive semantic fusion module then integrates complementary representations from different views for drug-disease association (DDA) prediction. Extensive experiments on benchmark datasets demonstrate that MSeR² outperforms existing methods across multiple evaluation metrics. Ablation studies verify the contribution of each component to the proposed framework. A case study further shows that MSeR² can identify meaningful reasoning paths, providing interpretable evidence for drug repositioning.

## Setup Environment

We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).

The code was tested with Python 3.8.20, PyTorch 2.4.1, and CUDA 11.8.
Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete fold-wise encoder and XGBoost experiment using:

```bash
python classifier.py
```

To run a single dataset, for example C-dataset, use:

```bash
python classifier.py --datasets C-dataset
```

