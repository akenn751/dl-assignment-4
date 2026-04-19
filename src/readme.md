# Readme for train.py

More information about the overall structure of the repo is available in the main readme file in the root directory. This readme is intended to explain how to use the train.py file.

## Core Training Arguments

`--mode {char, word}`

Selects the input representations.

- char - character-level one-hot vectors (256 dim)
- word - word-level one-hot vectors (5000 dim)

`--model {rnn, lstm}`

Chooses the model architecture

- rnn - U/W/V recurrent network
- lstm - LSTM with one-hot inputs

`--hidden <int>`

Hidden layer size (number of units in recurrent layer). Examples 32, 64, 128

`--seq <int>`

Sequence length (number of time steps per training sample) Examples 25, 50

`--batch <int>`

Batch size for training.

`--epochs <int>`

Bumber of epochs (full passes through the dataset)

## Optimization and Device Arguments

`--lr <float>`

Learning rate for the optimizer

`--seed <int>`

Random seed for reproducibility

`--seed-text "<string>"`

Initial text used when generating samples at breakpoints.

`device`

Automatically selected:
- cuda (if available)
- cpu (if cuda is not available)

## Testing Management

`--exp-name <string>`

Name of the test run, used for naming subdirectories inside the save_dir and identifying items in the artifacts folder.

`--save-dir <path>`

Root directory where all artifacts are stored:
- checkpoints
- final model
- samples
- loss logs
- metadata

`--save-checkpoints`

If set, saves intermediate checkpoints during training.

`--save-final-only`

If set, saves only the final model (significantly improves speed and disk space usage over checkpointing)

## Breakpoints & Sampling

`--breakpoints <list>`

Comma-separated list of epochs where text samples should be generated.

E.G. 

--breakpoints 1,2,3,4,5

`--breakpoint-interval <int>`

Alternative to explicit list, generates samples every N epochs

E.G.

--breakpoint-interval 1

`--sample-length <int>`

Number of tokens (chars or words) to generate at each breakpoint

## Training Control

`--max-steps <int>`

Optional method to stop training early. Stops the training after this many steps.

## Example command using all the options

`python -m src.train --mode word --model rnn --hidden 64 --seq 50 --batch 16 --epochs 5 --lr 5e-4 --exp-name example_full_run --save-dir artifacts --save-every 1 --save-checkpoints --breakpoints 1,2,3,4,5 --breakpoint-interval 1 --sample-length 100 --max-steps 10000 --seed 123 --seed-text "The adventure" --save-final-only`
