#!/bin/bash

# run the program for each city of interest
python3 dRegression.py -c kdfw  # run with the city kdfw
python3 dRegression.py -c kcys  # run with the city kcys
python3 dRegression.py -c kroa  # run with the city kroa

python3 neural_net.py -c kdfw   # run neural network with the city kdfw
python3 neural_net.py -c kcys   # run neural network with the city kcys
python3 neural_net.py -c kroa   # run neural network with the city kroa

python3 create_plot.py          # create the plots for each city