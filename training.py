#!/usr/bin/env python

print "Importing packages..."

import sys, os
import numpy as np
import ROOT
import pandas
pandas.options.mode.chained_assignment = None #switch off that annoying 'chained assignment' warning
import math
import root_numpy
import matplotlib.pyplot as plt

print "Finished importing packages."

print "Loading settings..."

from settings import settings
from plotting import *

print "Finished loading settings."

sigfile = settings["sig_file"]
bkgfile = settings["bkg_file"]
model = settings["model_name"]

## read input into pandas dataframe
sig_ext = os.path.splitext(sigfile)[1]
bkg_ext = os.path.splitext(bkgfile)[1]
if sig_ext == '.csv' and bkg_ext == '.csv':
    ## input events are rows in csv file with header in first line
    df_sig = pandas.read_csv(sigfile,header=0)
    df_bkg = pandas.read_csv(bkgfile,header=0)
elif sig_ext == '.root' and bkg_ext == '.root':
    ## input events are branches in a root file
    df_sig = pandas.DataFrame(root_numpy.root2array(sigfile, treename="tree"))
    df_bkg = pandas.DataFrame(root_numpy.root2array(bkgfile, treename="tree"))
else:
    print "Input file format(s) not understood: make sure they are all root/all csv"
    sys.exit()

## make sure signal and background have same input features/branches    
if list(df_sig) != list(df_bkg):
    print "Error: List of input features not identical for signal and background. Exiting..."
    sys.exit()


print "The following input features are available:"
for feature in list(df_sig):
    print "\t", feature
    
#print list(df_sig)
#sys.exit()

## add truth labels as last column
df_sig["is_signal"] = 1
df_bkg["is_signal"] = 0    
     
## merge signal and background events, and shuffle 
df = pandas.concat([df_sig, df_bkg], ignore_index=True)
df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

## drop some features here
if settings["keep_features"] is not None:
    keep = settings["keep_features"]
if settings["drop_features"] is not None:
    drop = settings["drop_features"]

keep = ['pTD', 'nPF', 'C', 'wPF', 'xmax', 'N95','is_signal']
df = df[keep]
#df = df.drop(columns=drop)
features = list(df)

## split data into train and test samples
train_frac = settings["training_frac"]
df["split"] = np.random.choice([0,1], df.shape[0],p=[train_frac, 1-train_frac])
train = df[ df["split"]==0 ]
test  = df[ df["split"]==1 ]

## remove entries with NaNs
df.dropna(how='any',inplace=True)

## sklearn requires numpy array as input
used_vars = tuple([ col for col in xrange(len(features)-1)])
X_train, y_train =  np.asarray(train)[:,used_vars], np.asarray(train["is_signal"])
X_test, y_test = np.asarray(test)[:,used_vars], np.asarray(test["is_signal"])

## classifier params
classifier = settings["classifier"]
cname = str(classifier)

n_est   = settings["n_estimators"]
lr      = settings["learning_rate"]
maxd    = settings["max_depth"]
verbose = settings["verbose"]

subsamp = settings["subsample"]
minsplit = settings["min_samples_split"]
maxfeat = settings["max_features"]

model = classifier()

#generic params
model.set_params(n_estimators=n_est,
                 learning_rate=lr)

if 'GradientBoostingClassifier' in cname:
    model.set_params(subsample=subsamp,
                     max_depth=maxd,
                     verbose=verbose)
if 'RandomForestClassifier' in cname:
    model.set_params(min_samples_split=minsplit,
                     max_depth=maxd,
                     max_features=maxfeat)
if 'AdaBoostClassifier' in cname:
    model.set_params(base_estimator=DecisionTreeClassifier(max_depth=maxd,
                                              min_samples_split=minsplit,
                                              max_features=maxfeat))
    
model.fit(X_train, y_train)

#get accuracy
print "Accuracy:" , model.score(X_test,y_test)

## predict labels on test data
scores = model.predict_proba(X_test)

if settings["plot_scores"]:
    plot_scores(test, scores)

if settings["plot_loss"]:
    plot_loss(X_train, X_test, y_train, y_test, model)    

if settings["plot_roc"]:    
    plot_roc(scores, y_test)    

if settings["plot_SIC"]:
    plot_sic(scores, y_test)

if settings["plot_importance"]:
    plot_importance(model, features)
