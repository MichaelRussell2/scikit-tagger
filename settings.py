from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#@TODO: LogisticRegression, XGBoost

settings = {

    "model_name"        : "quark_vs_gluon_BDT",

    ## Generic classifier parameters ##
    "classifier"        : GradientBoostingClassifier,
    "n_estimators"      : 50,
    "learning_rate"     : 1,

    ## gradient boost specific parameters
    "max_depth"         : 4,
    "subsample"         : 0.9,
    "verbose"           : 1,

    ##random-forest specific parameters
    "max_features"      : "auto",
    "min_samples_split" : 2,

    ##adaBoost specific parameters

    "load_model"        : False, #if True replace w/ filename
    "save_model"        : False, #if True replace w/ filename
    "sig_file"          : "quarks_fs.root",
    "bkg_file"          : "gluons_fs.root",

    "plot_path"         : "./plots",

    "training_frac"     : 0.95,
    "keep_features"     : None, #replace with list of features
    "drop_features"     : None,

    "plot_scores"       : True,
    "plot_loss"         : True,
    "plot_roc"          : True,
    "plot_SIC"          : True,
    "plot_features"     : True,
    "plot_features_2d"  : False,
    "plot_importance"   : True,
    "plot_decisions"    : False}

