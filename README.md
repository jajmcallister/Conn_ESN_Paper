# Data Files and Code 

This is the code and data files for our paper "Non-random brain connectome wiring enables robust and efficient neural network function under high sparsity" by James McAllister, Conor Houghton, John Wade, and Cian O'Donnell.

## Julia use:

The code for results in the paper was written in Julia (version 1.11). 
The following packages and versions were used in the study:
- LinearAlgebra
- Random
- Plots 
- Distributions v0.25.113
- Statistics v1.11.1
- StatsBase v0.34.3
- OrdinaryDiffEq v6.89.0
- Graphs v1.9.0
- LightGraphs v1.3.5
- GraphRecipes v0.5.13
- HypothesisTests v0.11.3
- FFTW v1.8.0
- CSV v0.10.15
- DataFrames v1.7.0

## Data Files
These are the data files for the networks in this study:

### esn_weight_matrices
This folder (link:    ) contains the weight matrices for the Echo State Networks (ESNs) used in the paper. There are three subfolders: connectome-based, random (Erdos-Renyi), and configuration model (cfg) weight matrices. Within each of these subfolders, there are separate folders for larval and adult-based connectomes.

e.g.

    \connectome

        \larva

            This folder contains the 9 subnetworks derived from the larval Drosophila connectome by hierarchical stochastic block model. These subnetworks differe in size and sparsity. Within each subnetwork folder, there are 30 initialisations of ESN weight matrices.

                \subnetwork1
                - `matrix1.csv` — weight matrix for ESN initialisation 1
                - `matrix2.csv` — weight matrix for ESN initialisation 2
                - ...
                - `matrix30.csv` — weight matrix for ESN initialisation 30
                \subnetwork2
                - `matrix1.csv` — weight matrix for ESN initialisation 1
                - `matrix2.csv` — weight matrix for ESN initialisation 2
                - ...
                - `matrix30.csv` — weight matrix for ESN initialisation 30
                ...
                \subnetwork9
                ...
        \adult
            This folder contains the 28 subnetworks derived from the adult Drosophila hemibrain connectome by hierarchical stochastic block model. These subnetworks differ in size and sparsity. Within each subnetwork folder, there are 30 initialisations of ESN weight matrices.
                \subnetwork1
                - `matrix1.csv` — weight matrix for ESN initialisation 1
                - `matrix2.csv` — weight matrix for ESN initialisation 2
                - ...
                - `matrix30.csv` — weight matrix for ESN initialisation 30
                \subnetwork2
                - `matrix1.csv` — weight matrix for ESN initialisation 1
                - `matrix2.csv` — weight matrix for ESN initialisation 2
                - ...
                - `matrix30.csv` — weight matrix for ESN initialisation 30
                ...
                \subnetwork28
                ...


## Julia Scripts
Julia scripts used:
 - 'src' - source code with necessary .jl files with functions for running ESN initialisation, training, tasks, dynamics, etc.
 - 'run' - specific code for analyses in the study such as pruning, dynamical regimes, parameter sweeps, netwrk feature correlations, weighted task variance, etc.
