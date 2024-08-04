import pandas as pd
import numpy as np
# laoding the csv file to the pandas dataframe
data = pd.read_csv('heart.csv')

# Splitting the feature and target variable
X = data.drop(columns = ['target'], axis = 1)
Y = data['target']

# splitting the data into training and test dateset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# Accuracy Score of training model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on Training data :: ", training_data_accuracy)

# Accuracy Score of test model
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on Test data :: ", test_data_accuracy)

input_data = []
try:
    age = int(input("Enter the age(in years) --> "))
    if age<1:
        raise ValueError("Not a vaild age")
    else:
        input_data.append(age)
        
    sex = int(input("Enter the sex(1 = male and 0 = female) --> "))
    if 0<=sex<2:
        input_data.append(age)
    else:
        raise ValueError("Not a vaild sex(1=male and 0=female)")
        
    cp = int(input("Enter the chest pain type (4 values) --> "))
    if 0<=cp<4:
        input_data.append(cp)
    else:
        raise ValueError("Not a vaild chest pain type(4 values)")
        
    trestbps = int(input("Enter the resting blood pressure(in mm Hg on admission to the hospital) --> "))
    input_data.append(trestbps)
    
    chol = int(input("Enter the serum cholestoral in mg/dl --> "))
    input_data.append(chol)
    
    fbs = int(input("Enter the fasting blood sugar > 120 mg/dl(1 = true; 0 = false) --> "))
    if 0<=fbs<2:
        input_data.append(fbs)
    else:
        raise ValueError("Not a vaild fasting blood sugar > 120 mg/dl(1 = true; 0 = false)")
        
    
    restecg = int(input("Enter the resting electrocardiographic results (values 0,1,2) --> "))
    if 0<=restecg<3:
        input_data.append(restecg)
    else:
        raise ValueError("Not a vaild resting electrocardiographic results (values 0,1,2)")
    
    thalach = int(input("Enter the maximum heart rate achieved --> "))
    input_data.append(thalach)
    
    exang = int(input("Enter the exercise induced angina (1 = yes; 0 = no) --> "))
    if 0<=exang<2:
        input_data.append(exang)
    else:
        raise ValueError("Not a vaild exercise induced angina (1 = yes; 0 = no)")
        
    oldpeak = float(input("Enter the ST depression induced by exercise relative to rest --> "))
    input_data.append(oldpeak)
    
    slope = int(input("Enter the slope of the peak exercise ST segment --> "))
    input_data.append(slope)
    
    ca = int(input("Enter the number of major vessels (0-3) colored by flourosopy --> "))
    if 0<=ca<4:
        input_data.append(ca)
    else:
        raise ValueError("Not a vaild number of major vessels (0-3) colored by flourosopy")
        
    thal = int(input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect --> "))
    input_data.append(thal)
except Exception as e:
    print(e)

# change the input data to a numpy array
input_data_as_nparray = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_nparray.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print("The person does not have a Heart Disease")
else:
    print("The person has Heart Disease")

# 55,1,0,160,289,0,0,145,1,0.8,1,1,3,0