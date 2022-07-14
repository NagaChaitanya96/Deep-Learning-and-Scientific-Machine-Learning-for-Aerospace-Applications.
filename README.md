# Deep-Learning and Scientific Machine Learning for Aerospace Applications.

Deep Learning and Scientific Machine Learning for Aerospace Applications (SCIML). 
Scientific Machine Learning combines the two fields of Scientific Computing and 
Machine Learning. Physics Informed Neural Networks (NeuralPDE) has been used to 
solve many of the ODE’s and PDE’s in CFD & FEM. Whereas, it hasn’t been used for 
solving Optimal Control Problems, in this research project we are going to solve the 
Optimal Control Problems using PINN’s and DNN driven Trajectory generation, we 
analysed the accuracy of PINN based solvers with conventional methods. We 
implemented various approaches of solving the Moon Lander an Optimal control problem, 
such as, modifying the model structurally and through an additional loss function for 
defining our objective function, we also implemented digital twins with some high-fidelity 
data for parameter estimation as well as estimating the neural networks model parameters. 
In addition, we have implemented a Neural Architecture Search to find out the best model 
with hyperparameter tuning to generate a Deep Neural Network which could be used for 
trajectory generation and validated its effectiveness with the conventional and PINN based 
approach. We observed the above approaches give satisfactory results in toy problems. 
Finally, we have solved the actual sized 2D model of Reusable Launch Vehicle which has 
7 states and 2 controls with boundary and path constraints where the aerodynamic analysis 
of model is also involved; data generation includes solving the OCP using PK Adaptive 
algorithm, data generation, data wrangling, Neural Architecture Search, validating the 
model using DNN driven trajectory to utilize it for real time simulations.
