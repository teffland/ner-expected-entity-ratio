# ner-expected-entity-ratio
Implementation and experiments for "Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss" to appear in TACL 2021.

## Installation

Experiments were all run on a p3.2xlarge instance on AWS with the amazon deep learning AMI on ubuntu 18 using python 3.7.
(You'll need at least 16GB of GPU ram to run these experiments as is.)

Install the required environment with: `conda env create -f environment.yml`

Then activate with: `conda activate env`

## Usage

### Getting the data

The CoNLL 2003 data Stephen Mayhew is already in the `data/` dir, but the ontonotes5 data must be created programmatically after you place the corpus
directory into the `data/` dir because of licensing restrictions.

Then in this order run the commands:

Convert the original Ontonotes5 corpus dataset format to ours
```bash
bash scripts/process_ontonotes_data.sh
```

Create the vocab files
```bash
bash scripts/make_vocabs.sh
```

Downsample the datasets
```bash
bash scripts/sample_datasets.sh
```

### Running the experiments

Published experiments can all be run using the bash scripts in `scripts/benchmark_runs`, `scripts/eer_variation_runs`, and `scripts/learning_curve_runs`.

For example:
 
```bash
cd ner-expected-entity-ratio
conda activate env
bash scripts/benchmark_runs/run_o_bias_experiments.sh
```

Many of the experiments were done in parallel using
remote nodes, so either this process needs to be reproduced, or the respective `run_remote_ml_experiment` invocations need to be converted to `run_ml_experiment` commands by removing the remote parameters.

Also, we've left out the codebases for the baselines (Mayhew et al. '19, Li et al. '21) because they aren't our works but if you need these let me know.


## Codebase TOC

- `data`: directory housing data. conll data already provided
- `experiments`: contains base allennlp config files for running experiments and is where runs are logged
- `ml`: source code
- `scripts`: bash scripts for running experiments


## Citing

If you make use of this code or information from the paper please cite:
```
TODO
```