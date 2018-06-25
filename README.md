# scikit-tagger
Interface to scikit-learn classifiers for particle physics analyses


## Prerequisites 
  * matplotlib, numpy
  * pyROOT
  * root_numpy
  * pandas (> v0.18.0)
  * scikit-learn (> v0.17.1)
  
## Input
  * Signal and background input files, either in ROOT format with features as a list of branches, 
    or in csv files with features as columns (assumes header in first row).
    
## Usage
  * All options (classifier and parameters, features to keep or skip, what to plot...) can be found in `settings.py`. More options
    can be added to `training.py`.
  * Plot options can be modified in `plotting.py`.
  
## Output 
  * Classifier accuracy, plots of train/test loss, ROC and SIC curves, feature histograms and relative importance
  * All plots given both as .png and .dat files with raw data
  
