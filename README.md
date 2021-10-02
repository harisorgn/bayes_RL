# bayes_RL

Fitting reinforcement-learning agents to behavioural data derived from the Affective Bias Task and the Probabilistic Reversal Learning Task. A hierarchical model structure is been used to infer individual- and population-level differences under a range of (pharmacological) manipulations on the task subjects. Turing is the probabilistic programming language of choice for now that runs the sample-based bayesian inference, future experimentation will likely involve Soss as well. 

The agent models include action-value learning with simple δ-rules and different action selection algorithms for comparison : ε-greedy, softmax and ε-softmax (a combination of the first two). Future version will expand on different learning rules as well, likely involving offline learning too. Model comparison with WAIC and LOOIC was not feasible, probably because the likelihood function was highly variable after leaving data from all sessions of one subject out, therefore K-fold cross-validation was implemented to perform the comparison.
