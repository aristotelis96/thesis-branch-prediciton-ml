# Branch Prediction using artificial neural networks

This repository is for my thesis titled "Branch Prediction using artificial neural networks".
Here you will find my thesis along with a presentation and all the code needed to implement,
train and evaluate models for branch prediction.
The dataset was created using an extended version of the ChampSim Simulator. You can find the 
updated version I used for my thesis [here](https://github.com/aristotelis96/ChampSim).
Also, this repository includes code from the [BranchNet](https://github.com/siavashzk/BranchNet)
repository.

## Abstract
Machine learning and specifically neural networks have made a major
progress in various fields, in recent years. While they made
it possible for computers to be used successfully in numerous fields,
few steps have been made towards using them for improving and  accelerating
computer hardware. 
Therefore, we can examine how neural networks can be used
for enhancing computer architecture and subsequently maximizing processor 
efficiency overall. Designing an advanced computer engineering, is composed of many distinctive parts, each possessing its own complexity and significance. Some of these parts
can be replaced with neural networks, which can either be then implemented in hardware
level, or we can extract information based on their function, in order to create
new and more advanced units.}

This diploma thesis aims to examine existing neural networks, as well as developing new ones,
to be used as branch prediction units. 
Recently Stephen Tarsa and others, 
identified a few branch instructions that systematically produce misspredictions, 
execute multiple times and are independent from program input;
they labeled them as Hard-to-Predict
and showed that predicting them correct
can offer a significant improvement for the processor.
Because these branches have a high execution rate, training and making use of
artificial neural networks can be an option, whilst their independence from program input
allow offline training and usage of pre-trained models for predictions in future executions.
The first artificial neural network for this task
was also proposed by Stephen Tarsa and others;
they used a deep branch history which they encoded and analyzed using convolutional
neural networks, which can identify related branches to the one being predicted.
Siavash Zangeneh and others continued their work to develop BranchNet,
a more complex but also more accurate model. BranchNet uses a deeper (longer)
sequence of history branches to make a prediction by introducing 2 improvements;
one better way of encoding data and some extra neural layers, both of which allow it 
to process deeper branch histories. If we use these models for Hard-to-Predict branches
and pair them
with a state-of-the-art existing branch predictor for the rest of the branches,
we can achieve high performance
at specific tasks.

We studied these artificial neural networks and we propose
a model of our own, using Long Short-Term Memory networks.
Inspired from BranchNet, we encode branch history and process it using convolutional layers
in a similar way. The key difference is that we include an LSTM network as last layer, in which we feed
the processed branch history to make a prediction.
Moreover we embed neural networks in a computer architecture simulator, ChampSim,
so to evaluate them. We see that neural networks can be a powerful approach at predicting
branches; they boost performance of 541.leela more than 10%, regarding
instructions per cycle - IPC. In the future we can either develop more sophisticated models, 
capable of achieving higher accuracy,
or design artificial neural networks on a hardware level.


## ChampSim

Before training models you must first generate the datasets. First you need to identify Hard-to-Predict (H2P) 
branches and collect their branch histories (see more inside 
[ChampSim](https://github.com/aristotelis96/ChampSim) repository, regarding ChampSim traces, H2P tracking
and branch history extraction). 

## Data generation

After that, you need to:
1. specify paths to those files inside <b>`src/branchnet/environment_setup/paths.yaml`</b>
and 
2. run script `src/branchnet/bin/create_ml_datasets.py` in order to generate h5Py datasets.

## Models

You can find code for [Tarsa](https://arxiv.org/abs/1906.08170) predictor inside `src/models.py` as well
as some other branch predicting models, like LSTMs, which were used for training and evaluation.
Also, inside `src/branchnet` you can find code for [BranchNet](https://github.com/siavashzk/BranchNet) predictor which was also used in this project.

## Train models

You can train your models with `src/branchnetTrainer.py`.

* First create `src/output/Benchmark` directory and inside add `H2Ps[benchmark].npy file`, containing 
H2P branches found in that benchmark. <i> You can use `src/branchnet/bin/common.py`
(function `read_hard_brs_from_accuracy_files`) to extract H2P branches for a benchmark then save the 
npy array. </i>

* Train a [MODEL] for each H2P branch, for specified [Benchmark].
```
./branchnetTrainer.py [MODEL] [Benchmark]
```

This will generate `[branch].pt` files inside `outputs/[benchmark]/models` directory. Each `.pt` file has
all the information needed for a H2P neural predictor. 

## Evaluation

<i>This section needs to be improved in the future. `src/BatchDict_testerAll.py` is a complicated script
whith many hard-coded parts that need to be cleaned up. For now feel free to contact me for any help,
but I plan to change this in the future. </i>

I evaluated each model using `src/BatchDict_testerAll.py`. This script reads 
`ChampSim -collect_all_branches` output, which contains a complete branch history of a trace, and evaluates 
models on predicting H2P branches found accross the trace.

In order to speed things up, the script
parses the history and evaluates models when reaching a `batch` threshold (ideally the script
tries to calculate the maximum available batchSize that can fit inside the GPU).

The result is stored in two different folders. 
* `src/outputs/[benchmark]/resultStats/[model]` contains
python dictionaries, for each simpoint of [benchmark]. Each dictionary contains key-value pairs, where
key is the Instruction Pointer of a hard-to-predict branch and the value is how many correct predictions
had the neural network predictor made in that simpoint.

* `src/outputs/[benchmark]/predictionTraces/[model]` contains a list of "[IP_branch] [prediction]"
where [IP_branch] is the instruction pointer of a Hard-to-Predict branch and [prediction] is the predicted
outcome of [model]. This can later be used with [ChampSim](https://github.com/aristotelis96/ChampSim) 
to create simulations where a standart predictor (e.g. Tage) can be used for various branches and
neural networks can be used for Hard-to-Predict branches
