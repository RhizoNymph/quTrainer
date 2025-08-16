# quTrainer

This is an experiment in increasing the query understanding of LLMs for better formed queries in agentic search.

It begins with a reward model, trained on Google's query-wellformedness dataset which consists of annotated data from the Paralex corpus scraped from WikiAnswers.

https://github.com/google-research-datasets/query-wellformedness
https://knowitall.cs.washington.edu/paralex/

This reward model is a fine tune of ModernBERT from Answer.AI.

https://github.com/AnswerDotAI/ModernBERT

It uses pytorch for training and optuna to do a bayesian hyperparameter optimization run.

The goal is to use this reward model in RL based post training and measure performance on an agentic search benchmark before and after. I haven't yet decided what benchmark(s) I will use but I'll update when I do.

Currently the only thing to do here is run:

```
uv run reward_model.py
```

This will start the optuna study and train the reward model, saving an optimization history and the best model. With default settings, this takes a few hours on a 3090.