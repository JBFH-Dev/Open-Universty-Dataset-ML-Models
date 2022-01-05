#               README
#   - This code requires the use of the following Python packages: 
#       * Pandas
#       * Numpy
#       * Matplotlib
#       * Sklearn
#       * Numexpr
#
#   - These packages must be the versions available as of 28/04/2020 or higher:
#       Pandas - 1.0.3 
#       Numpy - 1.18.3
#       Matplotlib - 3.2.1
#       Sklearn - 0.22.2.post1
#       Numexpr - 2.7.1
#
#   - To install/update these packages type:
#       -$ pip3 install <PACKAGE NAME> -U
#
#   - CSV files are also required which can be downloaded from:
#       - https://analyse.kmi.open.ac.uk/open_dataset
#   
#   RUNNING THE CODE
#   - Ensure that the file structure is as follows:
#      | classifier.py
#        Data/
#          | assessments.csv
#          | courses.csv
#          | studentAssessment.csv
#          | studentInfo.csv
#          | studentRegistration.csv
#          | studentVle.csv
#          | vle.csv
#
#   - Navigate to the parent directory of classifier.py
#
#   - Type the following into the terminal
#       -$ python3 classifier.py
#
#   - You will then be asked whether to run with 2 or 3 classes:
#       - Type '2' to run with a PASS/FAIL classification
#       - Type '3' to run with DISTINCTION/PASS/FAIL classification
#
#   - After selecting, multiple models will be ran and cross-validated
#       - 3-class classification will not produce ROC curves


print("-----RUNNING-------\n")

#IMPORTED MODULES
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
pd.options.mode.chained_assignment = None

print("-- Imported packages\n")

#Load in tables
download_path = os.path.join("Data", "studentAssessment.csv")
stu_ass = pd.read_csv(download_path, low_memory=False)

download_path = os.path.join("Data", "studentVle.csv")
stu_vle = pd.read_csv(download_path, low_memory=False)

download_path = os.path.join("Data", "assessments.csv")
ass = pd.read_csv(download_path, low_memory=False)

download_path = os.path.join("Data", "courses.csv")
cou = pd.read_csv(download_path, low_memory=False)

download_path = os.path.join("Data", "vle.csv")
vle = pd.read_csv(download_path, low_memory=False)

download_path = os.path.join("Data", "studentInfo.csv")
stu_inf = pd.read_csv(download_path, low_memory=False)

download_path = os.path.join("Data", "studentRegistration.csv")
stu_reg = pd.read_csv(download_path, low_memory=False)

print("-- Loaded Tables\n")

#MAKE TABLES TO BE MERGED

#Make table of mean clicks on the VLE per student 
interaction_type_ext = pd.merge(vle, stu_vle, on=['code_module','code_presentation','id_site'])
interaction_type = interaction_type_ext.drop(columns=['week_from','week_to','date','code_module','code_presentation', 'id_site', 'activity_type'])
interaction_type = interaction_type.set_index(['id_student'], drop=True)
interaction_type = interaction_type.groupby(['id_student']).mean()
interaction_type = interaction_type.reset_index()
print("- Created first table")
#Make table of student id and when they registered
eagerness = stu_reg.drop(columns=['date_unregistration'])
eagerness['date_registration'] = eagerness.date_registration * -1
print("- Created second table")
#Make table of average score from assessments so far excluding exams since that felt like cheating
score = stu_ass.drop(columns=['date_submitted','is_banked'])
weight = ass.drop(columns=['code_module','code_presentation','date'])
sco_wei = pd.merge(score, weight, on=['id_assessment'])
sco_wei = sco_wei[sco_wei.assessment_type != 'Exam']
sco_wei = sco_wei.drop(columns=['assessment_type','id_assessment'])
sco_wei['influence'] = sco_wei.weight*sco_wei.score
sco_wei = sco_wei.drop(columns=['weight','score'])
sco_wei = sco_wei.groupby(['id_student']).mean()
sco_wei = sco_wei.reset_index()
print("- Created third table")

#MERGE NEW TABLES INTO STUDENT INFO USING COMMON KEYS

#Merge stu_inf and eagerness
stu_eag = pd.merge(stu_inf, eagerness, on=['code_module','code_presentation','id_student'])

#Merge stu_eag and weight
stu_sco = pd.merge(stu_eag, sco_wei, on=['id_student'])

#Merge stu_sco and interaction type
data = pd.merge(stu_sco, interaction_type, on=['id_student'])

print("-- Merged tables\n")

print("---------CLEANING DATA----------\n")
#BEGIN DATA CLEANING
#Remove any columns that contain over half Null values
half = len(data) / 2
data.dropna(thresh=half, axis=1, inplace = True)

######################COULD FIX WITH AVG##################################
'''
Some samples are missing the imd_band, date_registration
or influence attributes and so I need to decide as to whether
to remove these samples or try to fill them with averages. 
For now I will simply remove them as there are not too many.
'''
data.dropna(inplace=True)
'''
I have thus only removed 3.89% of the samples
'''
###########################################################################

'''
I know that withdrawn does not represent an outcome so I will not
include these in my prediction meanwhile Distinction is
still a pass.
For now I remove the samples that have 'withdrawn' as an outcome
since they hold no relevance to the students final grade.
'''
data = data[data.final_result != 'Withdrawn']

'''
Code can be run on either distinction/pass/fail or just pass/fail
please note that while scores and RMSE still work on three classes,
I do not plot ROC curves for 3-class classification
'''
print("-- How many classes shall I model on?\n--> 3 = DISTINCTION/PASS/FAIL\n--> 2 = PASS/FAIL")
classes = input()

if classes == '3':
    pass_num = 0.5
elif classes == '2':
    pass_num = 1.0
else:
    print("UNRECOGNISED INPUT - USING 2 CLASSES")
    pass_num = 1.0

#Create a dictionary that can be mapped to the data frame to replace data numerically
map_dict = {"final_result":{ "Pass": pass_num, "Fail": 0.0, "Distinction": 1.0}}
#Map dictionary
data = data.replace(map_dict)

#Preparing features for machine learning!!
'''
We need to prepare the data
- Convert categorical to numeric values
- Remove extraneous columns
'''
#Now I will map numerical values to the object datatypes that support numerical order
map_dict = {
    "imd_band": {
        "90-100%": 95,
        "80-90%": 85,
        "70-80%": 75,
        "60-70%": 65,
        "50-60%": 55,
        "40-50%": 45,
        "30-40%": 35,
        "20-30%": 25,
        "10-20": 15,
        "0-10%": 5
        },
    "age_band": {
        "0-35": 18,
        "35-55": 45,
        "55<=": 55
        },
    "highest_education": {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 5
        }
    }
data = data.replace(map_dict)

#Nominal Values
'''
These require encoding into dummy variables (ONEHOT ENCODING)
- Use panda's get_dummies() to make new dataframe
- Use concat() to add dummy columns to original frame
- Drop original columns
'''
cols = ["code_module","code_presentation","gender","region","disability"]
new_cols = pd.get_dummies(data[cols])
data = pd.concat([data, new_cols], axis=1)
data.drop(cols, axis=1, inplace=True)

#STRATIFIED SPLIT INTO TRAIN AND TEST SETS
'''
I now see that influence has a high correlation with final_result
But also has the highest range of any attribute, making it a good
choice of value for stratified sampling.
'''
#Divide into groups to make sampling easier
data["inf_grp"] = np.ceil(data["influence"] / 250) 
#remove final group as there is only one record in there
data["inf_grp"].where(data["inf_grp"] < 11.0, 10.0, inplace=True)
y = data['final_result']
data.drop(columns=['final_result'], inplace=True)
#Create sets
X_Tr, X_Te, Y_Tr, Y_Te = train_test_split(data, y, test_size=0.2, random_state=117, stratify=data['inf_grp'])
X_Tr.drop(columns=['inf_grp'], inplace=True)
X_Te.drop(columns=['inf_grp'], inplace=True)
print("-- Split\n")
'''
MINMAX SCALER
NOW WE HAVE SEPARATED INTO THE TWO SETS, WE WILL:
- FIT SCALER ON TRAIN DATA
- TRANSFORM TRAINING DATA
- TRANSFORM TEST DATA
- PREDICT WITH TRANSFORMED DATA SETS
'''
#Now we will use a min max scaler to reduce the range of the numerical data to 1
scl = MinMaxScaler()
X_Tr = pd.DataFrame(scl.fit_transform(X_Tr), columns = X_Tr.columns)
X_Te = pd.DataFrame(scl.transform(X_Te), columns = X_Te.columns)


print("-- Scaled\n")
print("----------DATA CLEANED--------\n")
if len(Y_Tr.value_counts()) == 3:
    print("--------MODELLING ON DISTINCTION/PASS/FAIL-------\n")
elif len(Y_Tr.value_counts()) == 2:
    print("-------MODELLING ON PASS/FAIL-------\n")

#Dictionary of models to be tested
models = {"Linear Regression":LinearRegression(n_jobs = -1), "SGD Classifier":SGDClassifier(random_state=117, n_jobs=-1), "Random Forest Classifier":RandomForestClassifier(n_jobs=-1, random_state=117), "SVC":svm.SVC(), "LinSVC":LinearSVC(random_state=117), "Logistic Regression":LogisticRegression(n_jobs = -1, random_state=117, max_iter=4000), "Decision Tree Regressor":DecisionTreeRegressor(), "Decision Tree Classifier":DecisionTreeClassifier(), "Ridge Classifier":RidgeClassifierCV()}
#Label encoder to allow for use of data with classifiers
le = preprocessing.LabelEncoder()
le.fit(Y_Tr)


def run_model(name, model):
    '''runs a preliminary test on a model'''
    print("-- ",name)
    start = time.time()
    #fit the model to the training data
    m_fit = model.fit(X_Tr, le.transform(Y_Tr))
    end = time.time()
    elapsed = end-start
    #use the fitted model to predict the test data
    predictions = model.predict(X_Te)
    #calculate RMSE
    mse = mean_squared_error(Y_Te, predictions)
    rmse = np.sqrt(mse)
    print("Score is: ",m_fit.score(X_Te, le.transform(Y_Te)),"\t| Took: ",elapsed," seconds\t| RMSE is: ",rmse)
    print()    

#Tests each model in the dictionary
for m in models:
    run_model(m, models[m])

print("------------COMPLETED PRELIMINARY MODELS---------------\n")

print("----------------CROSS VALIDATION-------------------\n")

########################VALIDATION##################################

def cross_validation(name, model, splits, data):
    '''
    Performed to give indication of overfitting as well as a better insight into the
    quality of each model
    '''
    cvs = cross_val_score(model, data[0], data[1], cv = splits)
    rmse = cross_val_score(model, data[0], data[1], cv=splits, scoring="neg_mean_squared_error", verbose=False)
    #acc = cross_val_score(model, X_Te, le.transform(Y_Te), cv=splits, scoring="accuracy", verbose=False)
    print("-- ",name,"\tScore: ",cvs.mean(),"\tRMSE: ",(rmse.mean()*-1.0))#,"\tAccuracy: ",acc.mean())
    print()
    return cvs.mean()#+acc.mean()

#Finds the best two models based on cross validation testing
top = ("None", -100.0)
second = ("None", -100.0)
for m in models:
    score = cross_validation(m, models[m], 5, (X_Tr, le.transform(Y_Tr)))
    if score > top[1]:
        second = top
        top = (m, score)
    elif score > second[1]:
        second = (m, score)
print("Best two are:\n-- ",top[0],"\n-- ",second[0])

############TUNING THROUGH GRID SEARCH#######################
print("------------CONDUCTING GRID SEARCH---------------\n")
'''
Shall be performing grid search on logistic regression and random forest as they
seem to perform best on the data.
'''
def tune_params(name, model, params):
    '''performs a grid search on the given model with the given parameters'''
    print("-- Tuning ",name)
    start = time.time()
    #Runs grid search on model and refits it with the new parameters
    gsc = GridSearchCV(model, params, n_jobs = -1, cv=5, refit=True, return_train_score=True, verbose=False)
    gsc.fit(X_Tr, le.transform(Y_Tr))
    end = time.time()
    elapsed = end-start
    print("- Best parameters are:\n",gsc.best_params_)
    print()
    tuned_model = gsc.best_estimator_
    cvs = cross_val_score(tuned_model, X_Tr, le.transform(Y_Tr), cv = 5, verbose=False)
    print("- New mean is: ",cvs.mean())
    print("- Took: ",elapsed," seconds")
    print()
    #returns a fitted version of the model with the best found parameters
    return tuned_model

#Tunes each model based on parameters listed in dictionary
parameter_tuning = {"Random Forest Classifier":[{'n_estimators':[50, 100, 200], 'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2', 3, 6, None], 'bootstrap':[False], 'oob_score':[False], 'n_jobs':[-1], 'warm_start':[True, False], 'random_state':[117]}, {'n_estimators':[50, 100, 200], 'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2', 3, 6, None], 'bootstrap':[True], 'oob_score':[True], 'n_jobs':[-1], 'warm_start':[True, False], 'random_state':[117]}, {'n_estimators':[50, 100, 200], 'criterion':['gini','entropy'], 'max_features':['auto','sqrt','log2', 3, 6, None], 'bootstrap':[True], 'oob_score':[False], 'n_jobs':[-1], 'warm_start':[True, False], 'random_state':[117]}], "Logistic Regression":{'solver':['newton-cg','sag','saga','lbfgs'], 'max_iter':[500, 2000, 40000], 'n_jobs':[-1], 'warm_start':[True, False], 'random_state':[117]}}
#216, 24
tuned_models ={}
for m in models:
    if m in parameter_tuning:
        tuned_models[m] = tune_params(m, models[m], parameter_tuning[m])

#tuned_models = {"Random Forest Classifier":RandomForestClassifier(bootstrap=True, criterion= 'gini', max_features=None, n_estimators= 200, n_jobs=-1, oob_score=True, warm_start=True), "Logistic Regression":LogisticRegression(max_iter=500, n_jobs=-1, solver='newton-cg', warm_start= True)}

#{'bootstrap': True, 'criterion': 'gini', 'max_features': None, 'n_estimators': 200, 'n_jobs': -1, 'oob_score': True, 'warm_start': True}
#{'max_iter': 500, 'n_jobs': -1, 'solver': 'newton-cg', 'warm_start': True}

#### COULD TRY ENSEMBLE METHODS HERE

##############EVALUATION OF MODELS#############################

print("------------CONDUCTING EVALUATION---------------\n")



def plot_roc_curve(ytp, ys, name): 
    '''plots the roc curve for a set of pass instances and saves it'''
    fpr, tpr, thresholds = roc_curve(ytp, ys)
    #plots the collection of values given
    plt.plot(fpr, tpr, linewidth=2) 
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])
    #labels each axis
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    #saves plot
    fname = name.replace(' ','_')+'_'+classes+'.png'
    plt.savefig(fname)
    plt.clf()
    print("-- ROC graph saved to: ", fname)
    print()

def final_eval(name, model):
    '''conducts an evaluation on the now-tuned model'''
    print("-- Evaluting ",name)
    #Now creates a set of data of just pass instances of data
    if classes == '3':
       print("-- Further evaluation only available for 2-class model\n")
       return
    Y_Tr_Pass = (Y_Tr == 1.0)
    Y_Te_Pass = (Y_Te == 1.0)
    print()
    #Cross validates trained model
    print("\n---------CROSS VALIDATION--------\n")
    score = cross_validation(m, models[m], 5, (X_Te, Y_Te))
    print("---------TRAINING ON SUCCESSES--------\n")
    print("-- Score when ",name," trained on passes: ",model.fit(X_Tr, Y_Tr_Pass).score(X_Te, Y_Te))
    #generates a confusion matrix to describe the models performance
    print("\n---------CONFUSION MATRIX--------\n")
    cvp = cross_val_predict(model, X_Te, Y_Te_Pass, cv=3, verbose=False)
    print("Confusion for ",name,":\n",confusion_matrix(Y_Te_Pass, cvp))
    print()
    #plots a roc curve for the model
    print("-----------ROC CURVES-----------\n")
    ys = cross_val_predict(model, X_Te, Y_Te_Pass, cv=3, verbose=False, method='predict_proba')
    ys = 1 - ys
    plot_roc_curve(Y_Te_Pass, ys[:,0], name)

for m in tuned_models:
    #tuned_models[m].fit(X_Tr, le.transform(Y_Tr))
    final_eval(m, tuned_models[m])
