# Rethinking Generalization

This repo replicates experiments performed in Zhang et al.'s ["Understanding Deep Learning Requires Rethinking Generalization"](https://arxiv.org/pdf/1611.03530.pdf) (2017). I only replicate experiments conducted on CIFAR10; some extensions are played with in `extensions.py`.

## Usage

To get setup using the [HMS O2 Cluster](https://harvardmed.atlassian.net/wiki/spaces/O2/overview?homepageId=1586790623):
* ssh in to the server
* request a compute node (e.g., `srun -n 6 --mem 40G --pty -t 10:00:00 -p gpu --gres=gpu:teslaV100:1 bash`)
* *Optional: Launch [tmux](https://github.com/tmux/tmux/wiki)*
* activate relevant conda env (e.g., `source activate sdm_env`)
* run experiments via `python runner.py`. See `python runner.py -h` for params.

## Results