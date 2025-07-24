# Chopsticks AI

AI agent for the popular hand game, chopsticks. Uses reinforcement learning (q-learning) to gradually learn the best moves.
Inspired by the NIM problem set for CS50 AI.

## Usage

Adjust the number of games played by the AI against itself inside `play.py`
Run `play.py`

## Statistics

`plot.py` is used to perform statistics on the training models. The plots represent the number of number of keys (or rewards) learnt by the model v/s the number of games played against itself.

The results indicate that there's little point in running the model more than 1000 as the keys barely increase. In fact, even 250 training matches do a good job.
