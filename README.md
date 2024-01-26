# P2P-MAML and P2P-MAML+

The code implements the experiments performed in the paper. It demonstrates the performance comparison of P2P-MAML and P2P-MAML+ with locally trained MAML at each node. We include the experiments for classification of Omniglot and CIFAR-FS images.

## Dependencies and Setup
All code runs on Python 3.6.7 with torch 1.9.1 and torchmeta 1.8.0 libraries installed.

## Running Experiments
There are two main folders:
1) Omniglot: for distributed MAML on Omniglot dataset.
2) CIFAR_FS: for distributed MAML on CIFAR_FS dataset.

In each folder, there are four different training setups:
1) 8way1shot:   8  classes per task and 1 example per class.
2) 8way5shots:  8  classes per task and 5 example per class.
3) 20way1shot:  20 classes per task and 1 example per class.	(Used for Figures 6 and 8 of the paper)
4) 20way5shots: 20 classes per task and 5 example per class.	(Used for Figures 5 and 7 of the paper)

In each of the above, there are three main files:
1) LocalMAML.py:    Train each node independently with MAML using local dataset
2) P2P_MAML.py:     Train the network using P2P-MAML method
3) P2P_MAMLplus.py: Train the network using P2P-MAML+ method

The above files generate the accuracy of each method for every node and save it in ".txt" file in the "data" folder. Once we get the ".txt" files for each method, the data can be visualized using "data/plots.ipynb".

The rest of the files contains the classes and methods used to implement the above. 

The user can run "LocalMAML.py", "P2P_MAML.py", and "P2P_MAMLplus.py" from any of the sub-folders and then see the accuracy plots by running "data/plots.ipynb".
