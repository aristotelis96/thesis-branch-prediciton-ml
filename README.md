<p align="center">
  <h1 align="center"> Branch Prediction using artificial neural networks </h1>
  <p> 
  This repository is my thesis titled "Branch Prediction using artificial neural networks".
  Here you will find all the code needed to implement, train and evaluate models for branch prediction.
  The dataset was created using an extended version of <i> ChampSim </i> Simulator. You can find the 
  updated version I used for my thesis <a url='https://github.com/aristotelis96/ChampSim'>here</a>.
  Also this repository includes code from <a url='https://github.com/siavashzk/BranchNet'>BranchNet</a>
  repository.
  <p>
</p>

# ChampSim

Before training models you must first generate the datasets. First you need to identify Hard-to-Predict (H2P) 
branches and collect their branch histories (see more inside 
[ChampSim](https://github.com/aristotelis96/ChampSim) repository, regarding ChampSim traces, H2P tracking
and branch history extraction). 

# Data generation

After that, you need to:
1. specify paths to those files inside <b>`src/branchnet/environment_setup/paths.yaml`</b>
and 
2. run script `src/branchnet/bin/create_ml_datasets.py` in order to generate h5Py datasets.

# Models

You can find code for [Tarsa](https://arxiv.org/abs/1906.08170) predictor inside `src/models.py` as well
as some other branch predicting models, like LSTMs, which were used for training and evaluation.
Also, inside `src/branchnet` you can find code for [BranchNet](https://github.com/siavashzk/BranchNet) predictor which was also used in this project.

# Train models

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

# Evaluation

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
