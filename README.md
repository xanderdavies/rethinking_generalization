# Rethinking Generalization

This repo replicates a subset of experiments performed in Zhang et al.'s ["Understanding Deep Learning Requires Rethinking Generalization"](https://arxiv.org/pdf/1611.03530.pdf) (2017). I only replicate experiments conducted on CIFAR10, and use AlexNet and two MLP variants (1x512, and 3x512). As a small extension, I also track dead neuron prevalence in the MLPs. This is mostly an exercise in writing quick and clean PyTorch for me! Run experiments via `python runner.py` (e.g., `runner.py --lr 0.001 --model-name mlp-3x512 --use-adam --random-labels`). See `python runner.py -h` for params.

## Results
Train Accuracy             |  Test Accuracy
:-------------------------:|:-------------------------:
![Train Accuracy](https://user-images.githubusercontent.com/55059966/169595546-c22dbc0a-3297-4dc5-a1c9-70163d72f6f4.png) | ![Test Accuracy](https://user-images.githubusercontent.com/55059966/169595583-5fba6027-8bd9-48e3-a143-79ec98828812.png)
