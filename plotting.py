#!/usr/bin/env python

#@TODO: plot 2d features, split .png and .dat plots

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import math

from settings import settings

#plotpath=settings["model_name"]+"_plots"
plotpath=settings["plot_path"]
if not os.path.exists(plotpath):
    os.mkdir(plotpath)
    
## helper functions for plotting

## calculate the decision scores
def plot_scores(test_df, scores):
    class_names = {0: "background",
                   1: "signal"}

    classes = sorted(class_names.keys())
    for cls in classes:
        test_df[class_names[cls]] = scores[:,cls]
    sig = test_df["is_signal"]==1
    bkg = test_df["is_signal"]==0

    ## predicted labels
    probs = test_df["signal"][sig].values
    probb = test_df["signal"][bkg].values

    plt.clf()
    sig, bins, _ = plt.hist(probs, bins=np.linspace(0,1,51), normed=True, alpha=0.4, label="signal")
    bkg, bins, _ = plt.hist(probb, bins=np.linspace(0,1,51), normed=True, alpha=0.4, label="background")
    plt.legend()
    plt.xlabel("Predicted score")
    plt.ylabel("Event fraction")
    plt.savefig(plotpath+"/scores.png")

    xlo, xhi = bins[:-1], bins[1:]
    header="xmin \t xmax \t sig \t bkg"
    fmt=["%.2f", "%.2f", "%.5f", "%.5f"] 
    np.savetxt(plotpath+"/scores.dat",np.c_[xlo, xhi, sig, bkg], delimiter="\t", header=header, fmt=fmt)

##calculate the loss function
def plot_loss(X_train, X_test, y_train, y_test, model):
    n_est = settings["n_estimators"]
    deviance = np.zeros(n_est, dtype=np.float64)
    tt = np.zeros([n_est, 2])
    
    col=0 
    plt.clf()
    for [data, labels, color, label] in [[X_train, y_train, 'orange', 'train loss'],
                                        [X_test, y_test, 'magenta', 'test loss' ]]:

        for i, y_pred in enumerate(model.staged_decision_function(data)):
            deviance[i] = model.loss_(labels, y_pred)
        plt.plot(np.arange(deviance.shape[0])+1, deviance, '-', color=color,label=label)
        tt[:,col] = deviance
        col+=1
    plt.legend()
    plt.xlabel('n_estimators')
    plt.savefig(plotpath+'/loss.png')

    fmt=["%d", "%.7f", "%.7f"]
    header="Iter \t trainloss \t testloss"
    np.savetxt(plotpath+"/loss.dat",np.c_[np.arange(1,deviance.shape[0]+1),tt],delimiter="\t", header=header, fmt=fmt);

## plot the ROC curve    
def plot_roc(scores,y_test):
    X_pred, y_pred = scores[:,0], scores[:,1]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test,y_pred)

    plt.clf()
    plt.plot(fpr,tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(plotpath+"/roc.png")
 
    np.savetxt(plotpath+"/roc.dat",np.c_[fpr, tpr], delimiter="\t", header="FPR \t TPR", fmt="%.5f")

## plot significance improvement curve    
def plot_sic(scores, y_test):
    X_pred, y_pred = scores[:,0], scores[:,1]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test,y_pred)
    
#    sic = tpr/np.sqrt(fpr) if fpr != 0 else 0.
    sic = [x/math.sqrt(y) if y != 0. else 0 for x, y in zip(tpr,fpr) ]
    plt.clf()
    plt.plot(tpr,sic)
    plt.ylabel('False positive rate')
    plt.xlabel('Significance improvement')
    plt.savefig(plotpath+"/sic.png")
 
    np.savetxt(plotpath+"/sic.dat",np.c_[tpr, sic], delimiter="\t", header="TPR \t SIC", fmt="%.5f")
    
## plot histograms of used features
def plot_feature(dataframe, feature):

    is_sig = dataframe["is_signal"]==1
    is_bkg = dataframe["is_signal"]==0

    sig = dataframe[feature][is_sig].values
    bkg = dataframe[feature][is_bkg].values

    plt.clf()
    xmin = min(min(sig), min(bkg))
    xmax = max(max(sig), max(bkg))
    nbins = 50

    sig, bins, _ = plt.hist(sig, bins=np.linspace(xmin,xmax,nbins+1), alpha=0.4, normed=False,label='sig')
    bkg, bins, _ = plt.hist(bkg, bins=np.linspace(xmin,xmax,nbins+1), alpha=0.4, normed=False,label='bkg')
    plt.savefig(plotpath+'/'+feature+'.png')

    xlo, xhi = bins[:-1], bins[1:]
    fmt=["%.2f","%.2f","%.7f", "%.7f"]
    header="xmin \t xmax \t sig \t \bkg"
    np.savetxt(plotpath+'/'+feature+'.dat', np.c_[xlo,xhi,sig,bkg], fmt=fmt, delimiter='\t')
    
def plot_importance(model, features):

    importance = model.feature_importances_
    # normalise to maximum 
    importance = 100.*importance/importance.max()
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.clf()
    plt.barh(pos, importance[sorted_idx], align='center')
    features = [features[ix] for ix in sorted_idx]
    plt.yticks(pos, features )
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.savefig(plotpath+'/importance.png')

    importance = [importance[ix] for ix in sorted_idx]
    fmt=["%.1f", "%.5f"]
    header="y_pos \t importance"
    np.savetxt(plotpath+"/importance.dat",np.c_[pos, importance], delimiter="\t", header=header, fmt=fmt)
    
