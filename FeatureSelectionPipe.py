""" ***************************************************************************
# * File Description:                                                         *
# * Workflow for feature selection.                                           *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2a. Get Madelon like data                                                 *
# * 2b. Define Helper Function to Create Skewed Distributions                 *
# * 2c. Create Critical Features                                              *
# * 2d. Set Critical Feature Thresholds and Create Labels                     *
# * 2e. Visualize Critical Features                                           *
# * 2f. Create Feature Matrix X_all and Target Column y_all                   *
# * 2g. Create train and test set                                             *
# * 3a. Exploratory Data Analysis: Correlation Matrix                         *
# * 3b. Exploratory Data Analysis: Box Plots                                  *
# * 4a. Feature Selection: Removing with a large fraction of constant values  *
# * 4b. Feature Selection: Removing highly correlated features                *
# * 4c. Feature Selection: Selecting relevant features                        *
# * 4d. Feature Selection: Dangers of a model that overfits                   *
# * 4e. Feature Selection: Tuning Estimator                                   *
# * 4f. Feature Selection: Selecting relevant features with tuned model       *
# * 4g. Feature Selection: Model evaluation with selected features            *
# * 5a. Feature Selection Pipeline: Removing Highly Correlated + RFE          *
# * 5b. Feature Selection: Model evaluation with selected features            *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos <frank.ceballos89@gmail.com>                   *
# * --------------------------------------------------------------------------*
# * DATE CREATED: February 08, 2020                                           *
# * --------------------------------------------------------------------------*
# * NOTES: None                                                               *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


###############################################################################
#                        2a. Get Madelon like data                            #
###############################################################################
# Define the number of features for each type
n_features = 100
n_informative = 10
n_redundant = 50
n_repeated = 25
n_useless = 15

# Create Labels
informative_labels = [f'Informative {ii}' for ii in range(1, n_informative + 1)]
redundant_labels = [f'Redundant {ii}' for ii in range(n_informative + 1, n_informative + n_redundant + 1)]
repeated_labels = [f'Repeated {ii}' for ii in range(n_informative + n_redundant+ 1, n_informative + n_redundant + n_repeated + 1)]
useless_labels = [f'Useless {ii}' for ii in range(n_informative + n_redundant + n_repeated + 1, n_features + 1)]
labels = informative_labels + redundant_labels + repeated_labels + useless_labels

# Get data
X_madelon, y_madelon = make_classification(n_samples = 1000, n_features = n_features,
                           n_informative = n_informative,
                           n_redundant = n_redundant , n_repeated = n_repeated,
                           n_clusters_per_class = 2, class_sep = 0.5, flip_y = 0.05,
                           random_state = 42, shuffle = False)


# Numpy array to pandas dataframe and series
X_madelon = pd.DataFrame(X_madelon, columns = labels)


###############################################################################
#          2b. Define Helper Function to Create Skewed Distributions          #
###############################################################################
def randn_skew_fast(N, alpha=0.0, loc=0, scale=1.0):
    """
    Created a skewed distribution by randomly drawing from a skewed probability 
    densiy function.
    
    
    Parameters
    ----------
     N: int
        Number of points
    
    alpha: float
        Value describing the skeweness of the distribution. 
    
    loc: float
        Value describing the mean value of the skewed distribution
    
    scale: float
        Value describing the width of the skewed distribution
    
    
    Author Information
    ------------------
    jamesj629: <https://stackoverflow.com/users/266208/jamesj629>
    
    Source to original function: 
    <https://stackoverflow.com/questions/36200913/generate-n-random-numbers-from-a-skew-normal-distribution-using-numpy>
    """
    sigma = alpha / np.sqrt(1.0 + alpha**2) 
    u0 = np.random.randn(N)
    v = np.random.randn(N)
    u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    return u1


###############################################################################
#                       2c. Create Critical Features                          #
###############################################################################
# Set critical features distribution parameters
No_Critical_Features = 3
NUM_SAMPLES = 1000
SKEW_PARAMS = [0, 0, 0]
AVERAGE = [-10, 0, 10]
SEEDS = [42, 148, 526]
critical_features = []

# Create critical features
for ii in range(No_Critical_Features):
    # Set random seed
    np.random.seed(SEEDS[ii])
    
    # Set skewness and average value
    alpha_skew = SKEW_PARAMS[ii]
    average = AVERAGE[ii]
    
    # Get critical feature
    X_temp = randn_skew_fast(N = NUM_SAMPLES, alpha = alpha_skew, loc = average)
    
    # Append to critical feature list
    critical_features.append(X_temp)


###############################################################################
#            2d. Set Critical Feature Thresholds and Create Labels            #
###############################################################################
# Get critical labels
y_critical_1 = critical_features[0] < -8.3
y_critical_2 = critical_features[1] > -1.6
y_critical_3 = critical_features[2] > 8.3

# Save targets into a list
y_critical_targets = [y_critical_1, y_critical_2, y_critical_3]

# Define y_critical 
y_critical = y_critical_1*y_critical_2*y_critical_3*1


###############################################################################
#                       2e. Visualize Critical Features                       #
###############################################################################
# Set graph style
sns.set(font_scale = 1.5)
sns.set_style({'axes.facecolor': '1.0', 'axes.edgecolor': '0.85', 'grid.color': '0.85',
               'grid.linestyle': '-', 'axes.labelcolor': '0.4', 'xtick.color': '0.4',
               'ytick.color': '0.4', 'axes.grid': False})

# Set figure size
f, ax = plt.subplots(figsize=(12, 4))

for ii in range(No_Critical_Features):
    # Get critical features
    X_temp = critical_features[ii]
    y_temp = y_critical_targets[ii]
    
    # Get indices for true cases
    X_true = [X_temp[jj] for jj in range(len(y_temp)) if y_temp[jj] == 1]
    X_false = [X_temp[jj] for jj in range(len(y_temp)) if y_temp[jj] == 0]
    
    # Plot true values
    sns.distplot(X_true, color = 'g',  kde=False, ax = ax)
    sns.distplot(X_false, color = 'r',  kde=False, ax = ax)

# Generate a bolded horizontal line at y = 0
ax.axhline(y = 0, color = 'black', linewidth = 4, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Set axis labels
plt.xlabel('Feature Value')
plt.ylabel('Counts')

 # Tight layout
plt.tight_layout()

# Save figure
plt.savefig(f'Critical Feature Distribution.png', dpi = 1080)


###############################################################################
#           2f. Create Feature Matrix X_all and Target Column y_all           #
###############################################################################
# Define critical feature labels
critical_feature_labels =  [f'Critical {ii}' for ii in range(1, No_Critical_Features + 1)]

# Convert list to numpy arrays
X_critical = np.asarray(critical_features, dtype=np.float64).T
X_critical = pd.DataFrame(X_critical, columns = critical_feature_labels)

# Create feature matrix
X_all = pd.concat([X_madelon, X_critical], axis = 1)

# Create target
y_all = np.array([y_madelon[ii] * y_critical[ii] for ii in range(len(y_madelon))])


###############################################################################
#                       2g. Create train and test set                         #
###############################################################################
# Split the X_all and y_all
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, 
                                                                    test_size = 0.30,
                                                                    random_state = 42)


###############################################################################
#                3a. Exploratory Data Analysis: Correlation Matrix            #
###############################################################################
# Make correlation matrix
corr_matrix = X_all_train.corr(method = 'spearman').abs()

# Set font scale
sns.set(font_scale = 1)

# Set the figure size
f, ax = plt.subplots(figsize=(12, 12))

# Make heatmap
sns.heatmap(corr_matrix, cmap= 'YlGnBu', square=True, ax = ax)

# Tight layout
f.tight_layout()

# Save figure
f.savefig('correlation_matrix.png', dpi = 1080)


###############################################################################
#                  3b. Exploratory Data Analysis: Box Plots                   #
###############################################################################
# Set graph style
sns.set(font_scale = 0.75)
sns.set_style({'axes.facecolor': '1.0', 'axes.edgecolor': '0.85', 'grid.color': '0.85',
               'grid.linestyle': '-', 'axes.labelcolor': '0.4', 'xtick.color': '0.4',
               'ytick.color': '0.4', 'axes.grid': False})

# Create box plots based on feature type

# Set the figure size
f, ax = plt.subplots(figsize=(9, 12))
sns.boxplot(data=X_all_train, orient="h", palette="Set2")

# Set axis label
plt.xlabel('Feature Value')

# Tight layout
f.tight_layout()

# Save figure
f.savefig(f'Box Plots.png', dpi = 1080)


###############################################################################
#  4a. Feature Selection: Removing with a large fraction of constant values   #
###############################################################################
from tools.data_processing import FeatureSelector

# Define steps
step1 = {'Constant Features': {'frac_constant_values': 0.95}}

# Place steps in a list in the order you want them execute it
steps = [step1]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected = fs.transform(X_all_train)


###############################################################################
#         4b. Feature Selection: Removing highly correlated features          #
###############################################################################
# Define steps
step1 = {'Correlated Features': {'correlation_threshold': 0.95}}

# Place steps in a list in the order you want them execute it
steps = [step1]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected = fs.transform(X_all_train)


###############################################################################
#             4c. Feature Selection: Selecting relevant features              #
###############################################################################
# Define estimator
estimator = RandomForestClassifier(n_estimators = 100, max_depth = 7, 
                                   min_samples_leaf = 2, min_samples_split = 2,
                                   n_jobs = -1)

# Define steps
step1 = {'Relevant Features': {'cv': 5,
                               'estimator': estimator,
                                'n_estimators': 1000,
                                'max_iter': 50,
                                'verbose': 0,
                                'random_state': 42}}

# Place steps in a list in the order you want them execute it
steps = [step1]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected = fs.transform(X_all_train)


###############################################################################
#          4d. Feature Selection: Dangers of a model that overfits            #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit classifier
estimator.fit(X_all_train, y_all_train)

# Make predictions
y_pred_train = estimator.predict(X_all_train)
y_pred_test = estimator.predict(X_all_test)

# Measure performance
accuracy_train = metrics.accuracy_score(y_all_train, y_pred_train)
accuracy_test = metrics.accuracy_score(y_all_test, y_pred_test)

# Message to user
print(f'The accuracy of the classifier on the train set was: {accuracy_train*100}')
print(f'The accuracy of the classifier on the test set was: {accuracy_test*100}')


###############################################################################
#                    4e. Feature Selection: Tuning Estimator                  #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = { 'n_estimators': [200],
                'class_weight': [None, 'balanced'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [3, 4, 5, 6, 7, 8],
                'min_samples_split': [0.005, 0.01, 0.05, 0.10],
                'min_samples_leaf': [0.005, 0.01, 0.05, 0.10],
                'criterion' :['gini', 'entropy']     ,
                'n_jobs': [-1]
                 }

# Initialize GridSearch object
gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'accuracy')

# Fit gscv
gscv.fit(X_all_train, y_all_train)

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_
        
# Update classifier parameters
estimator.set_params(**best_params)

# Fit classifier
estimator.fit(X_all_train, y_all_train)

# Make predictions
y_pred_train = estimator.predict(X_all_train)
y_pred_test = estimator.predict(X_all_test)

# Measure performance
accuracy_train = metrics.accuracy_score(y_all_train, y_pred_train)
accuracy_test = metrics.accuracy_score(y_all_test, y_pred_test)

# Message to user
print(f'The accuracy of the classifier on the train set was: {accuracy_train*100}')
print(f'The accuracy of the classifier on the test set was: {accuracy_test*100}')


###############################################################################
#     4f. Feature Selection: Selecting relevant features with tuned model     #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state=42)

# Update classifier parameters
estimator.set_params(**best_params)

# Define steps
step1 = {'Relevant Features': {'cv': 5,
                               'estimator': estimator,
                                'n_estimators': 1000,
                                'max_iter': 50,
                                'verbose': 0,
                                'random_state': 42}}

# Place steps in a list in the order you want them execute it
steps = [step1]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected = fs.transform(X_all_train)


###############################################################################
#       4g. Feature Selection: Model evaluation with selected features        #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state=42)

# Update classifier parameters
estimator.set_params(**best_params)

# Get selected features
X_selected_train = fs.transform(X_all_train)
X_selected_test = fs.transform(X_all_test)

# Fit classifier
estimator.fit(X_selected_train, y_all_train)

# Make predictions
y_pred_train = estimator.predict(X_selected_train)
y_pred_test = estimator.predict(X_selected_test)

# Measure performance
accuracy_train = metrics.accuracy_score(y_all_train, y_pred_train)
accuracy_test = metrics.accuracy_score(y_all_test, y_pred_test)

# Message to user
print(f'The accuracy of the classifier on the train set was: {accuracy_train*100}')
print(f'The accuracy of the classifier on the test set was: {accuracy_test*100}')


###############################################################################
#    4h. Feature Selection: Features that maximize classifier performance     #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state = 42)

# Update classifier parameters
estimator.set_params(**best_params)

# Define steps
step1 = {'RFECV Features': {'cv': 5,
                            'estimator': estimator,
                            'step': 1,
                            'scoring': 'accuracy',
                            'verbose': 50}}

# Place steps in a list in the order you want them execute it
steps = [step1]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected = fs.transform(X_all_train)


###############################################################################
#                4f. Feature Selection: RFECV performance curve               #
###############################################################################
# Get Performance Data
performance_curve = {'Number of Features': list(range(1, len(fs.rfecv.grid_scores_) + 1)),
                    'Accuracy': fs.rfecv.grid_scores_}
performance_curve = pd.DataFrame(performance_curve)

# Performance vs Number of Features
# Set graph style
sns.set(font_scale = 1.75)
sns.set_style({'axes.facecolor': '1.0', 'axes.edgecolor': '0.85', 'grid.color': '0.85',
               "grid.linestyle": '-', 'axes.labelcolor': '0.4', 'xtick.color': '0.4',
               'ytick.color': '0.4'})
colors = sns.color_palette('RdYlGn', 20)
line_color = colors[3]
marker_colors = colors[-1]

# Plot
f, ax = plt.subplots(figsize=(13, 6.5))
sns.lineplot(x = 'Number of Features', y = 'Accuracy', data = performance_curve,
             color = line_color, lw = 4, ax = ax)
sns.regplot(x = performance_curve['Number of Features'], y = performance_curve['Accuracy'],
            color = marker_colors, fit_reg = False, scatter_kws = {"s": 200}, ax = ax)

# Axes limits
plt.xlim(0, len(fs.rfecv.grid_scores_)/1.5)
plt.ylim(0.54, 0.83)

# Generate a bolded horizontal line at y = 0
ax.axhline(y = 0.55, color = 'black', linewidth = 1.3, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig('Moderately to Highly Correated + RFECV performance_curve.png', dpi = 1080)


###############################################################################
#       5a. Feature Selection Pipeline: Removing Highly Correlated + RFE      #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state = 42)

# Update classifier parameters
estimator.set_params(**best_params)

# Define steps
step1 = {'Correlated Features': {'correlation_threshold': 0.80}}

step2 = {'RFECV Features': {'cv': 5,
                            'estimator': estimator,
                            'step': 1,
                            'scoring': 'accuracy',
                            'verbose': 50}}

# Place steps in a list in the order you want them execute it
steps = [step1, step2]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected = fs.transform(X_all_train)

###############################################################################
#       5b. Feature Selection: Model evaluation with selected features        #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state=42)

# Update classifier parameters
estimator.set_params(**best_params)

# Get selected features
X_selected_train = fs.transform(X_all_train)
X_selected_test = fs.transform(X_all_test)

# Fit classifier
estimator.fit(X_selected_train, y_all_train)

# Make predictions
y_pred_train = estimator.predict(X_selected_train)
y_pred_test = estimator.predict(X_selected_test)

# Measure performance
accuracy_train = metrics.accuracy_score(y_all_train, y_pred_train)
accuracy_test = metrics.accuracy_score(y_all_test, y_pred_test)

# Message to user
print(f'The accuracy of the classifier on the train set was: {accuracy_train*100}')
print(f'The accuracy of the classifier on the test set was: {accuracy_test*100}')


###############################################################################
#    6a. Feature Selection: Tune RF, Select Features, Evaluate Performance    #
###############################################################################
# Initiate classifier instance
estimator = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = { 'n_estimators': [200],
                'class_weight': [None, 'balanced'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [3, 4, 5, 6, 7, 8],
                'min_samples_split': [0.005, 0.01, 0.05, 0.10],
                'min_samples_leaf': [0.005, 0.01, 0.05, 0.10],
                'criterion' :['gini', 'entropy']     ,
                'n_jobs': [-1]
                 }

# Initialize GridSearch object
gscv = GridSearchCV(estimator, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'accuracy')

# Fit gscv
gscv.fit(X_all_train, y_all_train)

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

# Update classifier parameters
estimator.set_params(**best_params)

# Define steps
step1 = {'Constant Features': {'frac_constant_values': 0.95}}

step2 = {'Correlated Features': {'correlation_threshold': 0.80}}

step3 = {'Relevant Features': {'cv': 5,
                               'estimator': estimator,
                                'n_estimators': 1000,
                                'max_iter': 50,
                                'verbose': 0,
                                'random_state': 42}}

step4 = {'RFECV Features': {'cv': 5,
                            'estimator': estimator,
                            'step': 1,
                            'scoring': 'accuracy',
                            'verbose': 50}}

# Place steps in a list in the order you want them execute it
steps = [step1, step2, step3, step4]

# Initialize FeatureSelector()
fs = FeatureSelector()

# Apply feature selection methods in the order they appear in steps
fs.fit(X_all_train, y_all_train, steps)

# Get selected features
X_selected_train = fs.transform(X_all_train)
X_selected_test = fs.transform(X_all_test)

# Fit classifier
estimator.fit(X_selected_train, y_all_train)

# Make predictions
y_pred_train = estimator.predict(X_selected_train)
y_pred_test = estimator.predict(X_selected_test)

# Measure performance
accuracy_train = metrics.accuracy_score(y_all_train, y_pred_train)
accuracy_test = metrics.accuracy_score(y_all_test, y_pred_test)

# Message to user
print(f'The accuracy of the classifier on the train set was: {accuracy_train*100}')
print(f'The accuracy of the classifier on the test set was: {accuracy_test*100}')