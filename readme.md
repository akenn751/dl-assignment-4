# Report

A report of findings, testing processes, and answers to the assignment questions can be found in the **written_answers.md** file in the root directory.

# Overview of Repository Structure

## /artifacts

This folder contains outputs model training.

- /checkpoints -- Contains checkpoints stored from longer-running training runs to be used in case of crashes.
- /logs -- Contains logs from training runs, a autograd_sanity.txt log to confirm that the gradient calculations seem reasonable, and epoch_loss.csv to track loss over epochs.
- /metadata -- Contains the parameters used for  training runs in order to track what settings were used (e.g. # of hidden layers, sequence length, mode used)
- /plots -- Contains plotted graphs outputting from training runs.
- /samples -- Contains output text samples from breakpoints for different training runs.
- /(name of run) -- The named folders contain the final model output from a given training run.
- results_summary.txt -- Contains a running log of results from all training runs.

## /data

This folder is sub-divided into two folders:

- /processed - contains a chars.txt, vocab_words.json, and words.txt. These are the outputs of the preprocess.py utility which process the book texts and prepare them for use by the various models.
- /raw - contains the raw book files to be used for model training. Originally contained 8 books, as testing and training went on this was reduced to 3 in order to expedite training runs. 8 books would improve results but was infeasible within the time constraints of the assignment.

## /src

This folder contains the various scripts used in the assignment, organized into different folders.

- /models - this contains the two models used in the train.py script, rnn_lstm.py (LSTM model) and rnn_vanilla.py (Vanilla RNN). Also included is a test_rnn_vanilla.py which was used purely for testing purposes.
- /tests - this folder contains several scripts used for testing during the assignment.
- /utils - this folder contains utility scripts which are used in support of the models and training. Specifically it contains:
    - dataloader.py - Used by models to load data from the data/processed folder, using the appropriate corpus (word or char)
    - preprocess.py - Designed as a one-time use to transform books in data/raw and prepare them for processing by models. Can be re-run to add or remove books from the corpora.
    - test_dataloader.py - Used to test the dataloader
    - test_tokenizer.py - Used to test the tokenizer
    - tokenizer.py - Used to tokenize text for char/word indexing and one-hot-encoding for use in models.
- train.py - This is the primary script used to coordinate the various components and run tests against the different models and modes. This was designed to be able to switch test cases by using different arguments. 

## /root

The root folder contains some powershell scripts which were created to perform batch testing of different approaches and configurations for further investigation. It also contains the written_answers.md.