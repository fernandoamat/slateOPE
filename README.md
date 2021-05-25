# Slate Off-Policy Evaluation (OPE)
This repo accompanies and reproduces results from the paper "Control Variates for Slate Off-Policy Evaluation".

## Section 6: Experiments on real data
TO DO


## Section 7: Experiments on synthetic data
In order to reproduce any of the data points in Section 3 Figure 3 of the paper, go to `simulator` folder and run the `main.py` script.

For example, `python main.py -k 3 -d 10 -n 500 -s 1000` runs the simulation for all estimators for a slate of K=3 slots with D=10 actions per slot. Each simulation is run S=1000 times with N=500 samples each.   
This part of the code uses [Joblib package](https://joblib.readthedocs.io/en/latest/) to parallelize simulation and reduce computational time.

