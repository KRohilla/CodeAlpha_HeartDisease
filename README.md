# CodeAlpha_HeartDisease
Heart Disease Prediction ML Model and Deploy using Streamlit
<br>
Heart disease prediction using machine learning and data analytics to assess an individual's risk of developing heart-related conditions. The process typically involves the following steps:
<br>
<strong>Data Collection:</strong>Collecting a comprehensive dataset containing patient information.
The dataset is downloaded from the kaggle(https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and also the dataset is given in git also.
This data can include:<br>
age, sex, chest pain type (4 values), resting blood pressure, serum cholestoral in mg/dl, fasting blood sugar > 120 mg/dl
, resting electrocardiographic results (values 0,1,2)
, maximum heart rate achieved
, exercise induced angina
, oldpeak = ST depression induced by exercise relative to rest
, the slope of the peak exercise ST segment
, number of major vessels (0-3) colored by flourosopy
, thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

<br><br>
<strong>Data Preprocessing:</strong> Preparing the data for analysis by cleaning, normalizing, and encoding categorical variables. This step also involves handling missing values and ensuring that the data is in a suitable format for model training.
<br>The available dataset from the kaggle website is already a processed dataset with zero missing values and null values.

<br><br>
<strong>Exploratory Data Analysis (EDA):</strong> It is a crucial step in the machine learning pipeline, particularly for heart disease prediction. EDA involves a thorough examination of the dataset to understand its structure, detect anomalies, identify patterns, and uncover relationships between variables. This process helps in making informed decisions about data preprocessing, feature selection, and model building. 
<br>For this we have used the pandas_profiling library and make the output file for all the univariate and multivariate data analysis.

<br><br>
<strong>Feature Selection:</strong> Identifying the most relevant features that contribute to heart disease prediction. This step may involve statistical analysis, correlation studies, and domain expertise to select the most informative features.
<br>
As, of now we select all the feature columns from the available data without using the feature selection.

<br><br>
<strong>Model Selection and Training:</strong> Choosing appropriate machine learning models to predict heart disease. Commonly used algorithms include:<br>
Logistic Regression
Decision Trees
Random Forest
Support Vector Machines (SVM)
Neural Networks
Gradient Boosting Machines (e.g., XGBoost)
<br>
We have selected the Loistics Regression Algorithm to train our model for prediction of the heart disease.<br>
The selected models are trained on the prepared dataset, learning patterns and relationships within the data to make predictions.

<br><br>
<strong>Model Evaluation:</strong> Assessing the model's performance using various metrics, such as accuracy, precision, recall, F1-score, and the area under the receiver operating characteristic (ROC) curve (AUC-ROC) but we have only show the accuracy_score of the training and test dataset using the sklearn_metrics library of scikit-learn package. Cross-validation techniques can also be used to ensure the model's generalizability.

<br><br>
<strong>Prediction and Interpretation:</strong> Using the trained model to predict the likelihood of heart disease in new patients. Interpretation techniques, such as feature importance analysis and SHAP values, can help understand the contribution of different features to the model's predictions.

<br><br>
The goal of using machine learning for heart disease prediction is to provide healthcare professionals with a tool to identify high-risk individuals early, enabling timely interventions and personalized treatment plans.
