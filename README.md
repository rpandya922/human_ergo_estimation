# human_ergo_estimation
Code for learning a human ergonomic cost function using Baxter and a motion capture system
## Requirements
Python 2.7
all library dependencies listed in requirements.txt, can be installed with pip by running `pip install -r requiremenets.txt`
## Particle Filtering
To see visualization of particle filter, run `python particle_filter.py`. The entropy of the distibution will be printed on each iteration as well (each iteration is reweighting/resampling the particle filter based on a single training point). To change the number of particles, change the `NUM_PARTICLES` constant on line 12 of `particle_filter.py`.
## Probability Estimation
The file `probability_estimation.py` is not meant to be run on its own as it only contains functions for estimation of parameters for particle filtering. To print out the leave one out cross-validation error of an optimized lambda, include the file when in an ipython/python terminal, `ipython -i probability_estimation.py`, and run the function `predictProbs()`. All other functions in the file are helper functions.
## Other Files
`arm_joints_feasible_data.npy` is the training set, and the discretized feasible sets for the corresponding training points is found in `feasible_sets2.npy`.
`distribution.py` contains the class used to create distribution objects during particle filtering.
Files starting with `arm_joints_*` are primarily for data collection and/or early nearest neighbor predictions.