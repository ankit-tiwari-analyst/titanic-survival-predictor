🚢 Titanic Survival Predictor – Random Forest Machine Learning


📌 Project Overview

This project builds a Titanic Survival Prediction Model using Random Forest Classifier and demonstrates a complete Machine Learning workflow:

Data Loading
Data Cleaning
Feature Engineering
Model Training
Cross Validation
Model Evaluation
Prediction on New Passengers

The project shows how Machine Learning can discover patterns from historical Titanic data and predict whether a completely new passenger is likely to survive.

📂 Project Structure
├── Week_5_Lesson_4_ML.py
├── titanic.csv
├── README.md
└── requirements.txt (optional)
🔧 Technologies Used
Python
Pandas
Scikit-Learn
Random Forest Classifier
📊 Data Preprocessing
Missing Values
Age → Filled using Median value
Encoding
Male   → 0
Female → 1
⚙️ Feature Engineering

Three new features were created to improve model performance.

Feature	Description
FamilySize	SibSp + Parch + 1
IsAlone	1 if travelling alone else 0
AgeGroup	Categorized age into groups

Age Groups

Age	Label
0–12	0
13–18	1
19–35	2
36–60	3
60+	4
🤖 Model
Algorithm:
RandomForestClassifier

Parameters:
n_estimators = 100
random_state = 42
📈 Model Performance
Cross Validation Accuracy
CV Accuracy = 81.3%
Improvement over Lesson 3 = +1.6%
Previous Best = 78.7%

This improvement shows that feature engineering successfully increased the model's ability to generalize to unseen data.

Model Evaluation

The project also prints:

Test Accuracy
Confusion Matrix
Classification Report

to evaluate overall classification performance.

🚢 Passenger Predictions
Passenger 1
Feature	Value
Pclass	1
Sex	Female
Age	22
Fare	26.5
FamilySize	3
IsAlone	1
AgeGroup	Young Adult
Expected

✅ Survive

Model Prediction

✅ Survived

Reason

Female passenger
First Class
Higher survival probability according to historical patterns
Passenger 2
Feature	Value
Pclass	3
Sex	Male
Age	35
Fare	7.25
FamilySize	1
IsAlone	1
AgeGroup	Young Adult
Expected

❌ Die

Model Prediction

❌ Died

Reason

Third Class
Male
Travelling alone
Low fare
Passenger 3
Feature	Value
Pclass	2
Sex	Male
Age	9
Fare	24.5
FamilySize	4
IsAlone	0
AgeGroup	Child
Expected

✅ Survive

Model Prediction

✅ Survived

Reason

Child passenger
Family travelling together
Historical survival rates favor children
🚀 How to Use
1. Clone Repository
https://github.com/ankit-tiwari-analyst/titanic-survival-predictor

cd titanic-survival-predictor
2. Install Dependencies
pip install pandas scikit-learn
3. Place Dataset

Ensure the following file exists in the project folder:

titanic.csv
4. Run the Project
python Week_5_Lesson_4_ML.py
📝 Predict for Your Own Passenger

Create a new passenger using the same feature format:

new_passenger = pd.DataFrame({
    "Pclass": [1],
    "Sex_encoded": [1],   # Female = 1, Male = 0
    "Age": [30],
    "Fare": [80],
    "FamilySize": [2],
    "IsAlone": [0],
    "AgeGroup": [2]
})

prediction = rf.predict(new_passenger)[0]

print("Survived" if prediction == 1 else "Died")

You can modify:

Pclass
Sex_encoded
Age
Fare
FamilySize
IsAlone
AgeGroup

to predict survival for any custom passenger.

💡 Key Learning

This project demonstrates the difference between Exploratory Data Analysis (EDA) and Machine Learning.

EDA
Describes historical data
Answers predefined questions using aggregation and visualization
Machine Learning
Learns patterns automatically
Uses multiple features simultaneously
Predicts outcomes for passengers the model has never seen before
🎯 Final Result

✅ Complete Machine Learning Pipeline

✅ Feature Engineering

✅ Cross Validation Accuracy: 81.3%

✅ Random Forest Classification

✅ Prediction for Custom Passengers

✅ End-to-End Titanic Survival Prediction Project

👨‍💻 Author

Ankit Tiwari

Machine Learning & Data Analytics Learner | Python | Pandas | Scikit-Learn | SQL | Data Visualization
