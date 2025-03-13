# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.neighbors import KNeighborsClassifier
# from scipy.stats import uniform,randint

# data=pd.read_csv(r"C:\C_py\Python\Placement\data\placementdata.csv")
# data.drop(columns='StudentID',inplace=True)

# values=['ExtracurricularActivities',
#        'PlacementTraining','PlacementStatus']

# # label=LabelEncoder()
# # for i in values:
# #     data[i]=label.fit_transform(data[i])

# # x=data.drop(columns='PlacementStatus')
# # y=data['PlacementStatus']

# # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# # scaler=StandardScaler()

# # x_train_sca=scaler.fit_transform(x_train)
# # x_test_sca=scaler.transform(x_test)

# # smote=SMOTE()
# # x_train_smo,y_train_smo=smote.fit_resample(x_train_sca,y_train)
# # param={
#     ##For Decision Tree
#     # 'criterion': ['gini', 'entropy'],
#     # 'splitter': ['best', 'random'],
#     # 'max_depth': [None,5,10,15],
#     # 'min_samples_split': [2, 5, 10],
#     # 'min_samples_leaf': [1, 5, 10],
#     # 'max_features': ['sqrt', 'log2', None],
#     # 'ccp_alpha': [0.0,0.01,0.1]

#     # 'n_estimators': [100, 200],
#     # 'learning_rate': [0.01, 0.1, 1],
#     # 'max_depth': [1,3,5],
#     # 'subsample': [0.8, 1.0],
#     # 'colsample_bytree': [0.6,0.8, 1.0],

#     # """Logistic Regression"""
#     'penalty': ['l1', 'l2', 'elasticnet'],
#     'C': [0.001, 0.01, 0.1, 1, 10],
#     'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
#     'max_iter': [100, 500, 1000, 5000],
#     'class_weight': ['balanced'],
#     'tol': [1e-4, 1e-3, 1e-2],
#     'fit_intercept': [True, False],
#     'dual': [False],
#     'l1_ratio': [0.1, 0.5, 0.9]


#     ##for randomforestclassifier
#     # 'n_estimators':[50,100,200],
#     # 'criterion': ['gini', 'entropy', 'log_loss'],
#     # 'max_depth': [1,10, 20, 50],
#     # 'min_samples_split': [2, 5, 10],
#     # 'min_samples_leaf': [1, 2, 4],
#     # 'max_features': ['sqrt', 'log2'],
#     # 'bootstrap': [True, False],
#     # 'class_weight': ['balanced', 'balanced_subsample'],
#     # 'oob_score': [True, False],
#     # 'ccp_alpha': [0.0, 0.01, 0.1]

#     ##For KNN
#     # 'n_neighbors': [3,5,7,9,11],
#     # 'weights':['uniform','distance'],
#     # 'metric':['minkowski','chebyshev'],
#     # 'p':[1,2],
#     # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     # 'leaf_size': [10, 20, 30, 40, 50]

#     ##GradientBoosting Classifier
#     # 'n_estimators': [50, 100, 200, 500],  
#     # 'learning_rate': [0.01, 0.05, 0.1, 0.2],  
#     # 'max_depth': [3, 5, 10, 20],  
#     # 'min_samples_split': [2, 5, 10],  
#     # 'min_samples_leaf': [1, 5, 10],  
#     # 'subsample': [0.7, 0.8, 0.9, 1.0],  
#     # 'max_features': ['sqrt', 'log2', None]
# } 

# # model=GridSearchCV(estimator=LogisticRegression(),param_grid=param,n_jobs=-1,cv=5,scoring='accuracy',verbose=1)
# # model.fit(x_train_smo,y_train_smo)
# # y_pred=model.predict(x_test_sca)
# # print(pd.DataFrame({"Actual":y_test,"Predicted":y_pred}))
# # print(f"Accuracy Score: {accuracy_score(y_test,y_pred)*100:.2f}%")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r"C:\C_py\Python\Placement\data\placementdata.csv")
data.drop(columns='StudentID', inplace=True)

values = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
label_encoders = {}
for i in values:
    label_encoders[i] = LabelEncoder()
    data[i] = label_encoders[i].fit_transform(data[i])

X = data.drop(columns='PlacementStatus')
y = data['PlacementStatus']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

model = LogisticRegression()
model.fit(X_resampled, y_resampled)

joblib.dump(model, 'C:\C_py\Python\Placement\models/placement_model.pkl')
joblib.dump(scaler, 'C:\C_py\Python\Placement\models/scaler.pkl')
joblib.dump(label_encoders, 'C:\C_py\Python\Placement\models/label_encoders.pkl')

st.title("Placement Probability Predictor")

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
internships = st.number_input("Internships", min_value=0, max_value=10, step=1)
projects = st.number_input("Projects", min_value=0, max_value=10, step=1)
workshops = st.number_input("Workshops/Certifications", min_value=0, max_value=10, step=1)
apti_score = st.number_input("Aptitude Test Score", min_value=0, max_value=100, step=1)
soft_skills = st.number_input("Soft Skills Rating", min_value=0.0, max_value=10.0, step=0.1)
extra_activities = st.radio("Extracurricular Activities", ["No", "Yes"])
training = st.radio("Placement Training", ["No", "Yes"])
ssc_marks = st.number_input("SSC Marks", min_value=0, max_value=100, step=1)
hsc_marks = st.number_input("HSC Marks", min_value=0, max_value=100, step=1)

if st.button("Predict Placement Probability"):
    model = joblib.load('C:\C_py\Python\Placement\models/placement_model.pkl')
    scaler = joblib.load('C:\C_py\Python\Placement\models/scaler.pkl')
    label_encoders = joblib.load('C:\C_py\Python\Placement\models/label_encoders.pkl')
    
    extra_activities_enc = 1 if extra_activities == "Yes" else 0
    training_enc = 1 if training == "Yes" else 0
    
    input_data = np.array([[cgpa, internships, projects, workshops, apti_score, soft_skills, extra_activities_enc, training_enc, ssc_marks, hsc_marks]])
    input_data_scaled = scaler.transform(input_data)
    
    prob = model.predict_proba(input_data_scaled)[:, 1][0] * 100
    
    st.success(f"Predicted Probability of Getting Placed: {prob:.2f}%")
