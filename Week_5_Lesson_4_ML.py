
# Lesson 4 — The final piece. Everything from 5 weeks in one file. 🏆
 
# Part 1 — Feature engineering: creating better inputs
# Turn raw columns into more informative features

'''import pandas as pd

df = pd.read_csv('titanic.csv')

# Clean (same as always)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# Feature engineering — 3 new derived features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[0, 1, 2, 3, 4]   # child, teen, young adult, adult, elderly
).astype(int)

# All features including new ones
features = df[['Pclass', 'Sex_encoded', 'Age', 'Fare',
               'FamilySize', 'IsAlone', 'AgeGroup']]
target = df['Survived']'''


# Part 2 — Train the final model with cross-validation
# Use cross-validation from the start, not as an afterthought

'''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# Cross-validate first to get reliable performance estimate
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, features, target, cv=5)
print(f"CV Average: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

# Then train a final model on ALL training data for predictions
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)'''


# Part 3 — Full evaluation block
# Everything from Lesson 3, in one clean block

'''from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)

print("="*45)
print("  TITANIC SURVIVAL PREDICTOR — RESULTS")
print("="*45)
print(f"CV Accuracy:   {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
print(f"Test Accuracy: {accuracy_score(y_test, predictions):.1%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))'''


# Part 4 — Predict on a brand new passenger
# The real test: predict someone the model has never seen

# Create a new passenger — you can change these values
'''new_passenger = pd.DataFrame({
    'Pclass':      [3],
    'Sex_encoded': [0],    # 0 = male
    'Age':         [22],
    'Fare':        [7.25],
    'FamilySize':  [1],    # travelling alone
    'IsAlone':     [1],
    'AgeGroup':    [2]     # young adult 18-35
})

prediction = rf.predict(new_passenger)[0]
probability = rf.predict_proba(new_passenger)[0]

print(f"Prediction: {'Survived' if prediction == 1 else 'Died'}")
print(f"Probability of survival: {probability[1]:.1%}")
print(f"Probability of death:    {probability[0]:.1%}")'''


# Lesson 4: Assignment 

# Build the complete pipeline: load, clean, feature engineer (FamilySize, IsAlone, AgeGroup), train Random Forest

import pandas as pd 
df = pd.read_csv('titanic.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex_encoded'] = df['Sex'].map({'male':0, 'female':1})

# Feature engineering — 3 new derived features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['AgeGroup'] = pd.cut(df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# All features including new ones
features = df[['Pclass', 'Sex_encoded', 'Age', 'Fare',
                'FamilySize', 'IsAlone', 'AgeGroup']]

target = df['Survived']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, features, target, cv=5)
print(f"CV Average: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

# Then train a final model on ALL training data for predictions
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# Print the full evaluation block — CV accuracy, test accuracy, confusion matrix, classification report

from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report)

print("="*45)
print("  TITANIC SURVIVAL PREDICTOR — RESULTS")
print("="*45)
print(f"CV Accuracy:   {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
print(f"Test Accuracy: {accuracy_score(y_test, predictions):.1%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Compare: does the new feature-engineered model beat your best previous result (78.7% CV from Lesson 3)?

'''the new feature-engineered model beat the previous best result 78.7% with the margin of 1.6 ,
score 81.3% which show the increase in accuracy mean better result.'''


# Predict survival for 3 different passengers you invent yourself — one who you expect to survive (e.g. 1st class female), one who you expect to die (e.g. 3rd class male alone), and one who is genuinely uncertain (e.g. 2nd class male, child)

New_Passenger1 = pd.DataFrame({
    'Pclass': [1],
    'Sex_encoded': [1],
    'Age':         [22],
    'Fare':        [26.5],
    'FamilySize':  [3],    
    'IsAlone':     [1],
    'AgeGroup':    [2] 
})

prediction = rf.predict(New_Passenger1)[0]
print(f"Prediction: {'Survived' if prediction == 1 else 'Died'}")

New_Passenger2 = pd.DataFrame({
    'Pclass': [3],
    'Sex_encoded': [0],
    'Age':         [35],
    'Fare':        [7.25],
    'FamilySize':  [1],    
    'IsAlone':     [1],
    'AgeGroup':    [2] 
})

prediction1 = rf.predict(New_Passenger2)[0]
print(f'prediction: {'Survived' if prediction1 == 1 else 'Died'}')

New_Passenger3 = pd.DataFrame({
    'Pclass': [2],
    'Sex_encoded': [0],
    'Age':         [9],
    'Fare':        [24.5],
    'FamilySize':  [4],    
    'IsAlone':     [0],
    'AgeGroup':    [0] 
})

prediction2 = rf.predict(New_Passenger3)[0]
print(f'prediction: {'Survived' if prediction2 == 1 else 'Died'}')


# For each prediction, state what you EXPECTED before running it and whether the model agreed with your expectation

'''as per my analysis for New_Passenger1 Detail show that high expected to 'Survived' because multiple aspect are in favour of passenger,
most important facters are Gender, Pclass.'''

'''as per my analysis for New_Passenger2 Detail show that high expected to 'Died' because multiple aspect are not in favour of passenger,
most important facters are Gender, Pclass, FamilySize.'''

'''as per my analysis for New_Passenger3 Detail show that high expected to 'Survived' because multiple aspect are in favour of passenger,
most important facters are Age, AgeGroup, FamilySize.'''


# Write a final summary: what did your ML model add beyond your manual EDA from Week 2? What can the model do that groupby cannot?

# Summary:

'''if you ask me what ML model add beyond in my manul EDA, this make analysis more efficient and organised way,
and also much analysis in some few code.'''

'''"Manual EDA with groupby answers questions I already thought to ask — survival by Sex, survival by Pclass. 
he ML model finds patterns across ALL 7 features simultaneously and weights them automatically (Sex=58% importance, Pclass=21%). More importantly, the model can predict survival for a completely NEW passenger 
it has never seen — groupby can only describe what already happened. That's the core difference: EDA describes the past, ML predicts the future."'''



