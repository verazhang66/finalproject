import pandas as pd
csv_file_path = "social_media_usage.csv"
s = pd.read_csv(csv_file_path)
dimensions = s.shape
print("Number of rows:", dimensions[0])
print("Number of columns:", dimensions[1])
import numpy as np
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# toy dataframe with three rows and two columns
toy_df = pd.DataFrame({
    'Column1': [1, 2, 0],  
    'Column2': [0, 1, 3]  
})

cleaned_df = toy_df.applymap(clean_sm)

print(cleaned_df)
ss = pd.DataFrame()
def clean_fem(x):
    return np.where(x == 2, 1, 0)
ss['sm_li'] = s['web1h'].apply(clean_sm)
ss['income'] = s['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)
ss['education'] = s['educ2'].apply(lambda x: x if 1 <= x <= 8 else np.nan)
ss['parent'] = s['par'].apply(clean_sm)
ss['married'] = s['marital'].apply(clean_sm)
ss['female'] = s['gender'].apply(clean_fem)
ss['age'] = s['age'].apply(lambda x: x if x <= 98 else np.nan)

# Drop missing values
ss.dropna(inplace=True)

# Exploratory analysis
descriptive_stats = ss.describe()
correlation_matrix = ss.corr()

print(descriptive_stats)
print(correlation_matrix)
target_column = 'sm_li'

# Create the target vector y
y = ss[target_column]

# Create the feature set X by dropping the target column from the DataFrame
X = ss.drop(target_column, axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=42)  # set for reproducibility
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def clean_sm(x):
    return np.where(x == 1, 1, 0)

def clean_fem(x):
    return np.where(x == 2, 1, 0)

st.header("LinkedIn Usage Prediction App")

age = st.slider("Age", min_value=0, max_value=97, value=30)
education_options = [
    "Less than high school",
    "High school incomplete",
    "High school graduate",
    "Some Collge, no degree",
    "Two-year associate degree",
    "Four-year college or university degree",
    "Some postgraduate or professional schooling",
    "Postgraduate or professional degree",
]
education = st.selectbox("Education", education_options)

income_options = [
    "Less than $10,000",
    "$10,000 to under $20,000",
    "$20,000 to under $30,000",
    "$30,000 to under $40,000",
    "$40,000 to under $50,000",
    "$50,000 to under $75,000",
    "$75,000 to under $100,000",
    "$100,000 to under $150,000",
    "$150,000 or more",
    "Don't know",
    "Refused",
]
income = st.selectbox("Income", income_options)

parent_options = [
    "Yes",
    "No",
    "Don't know",
    "Refused",
]
parent = st.selectbox("Are you a parent of a child under 18 living in your home?", parent_options)

marital_options = [
    "Married",
    "Living with a partner",
    "Divorced",
    "Separated",
    "Widowed",
    "Never been married",
    "Don't know",
    "Refused",
]
married = st.selectbox("Marital",marital_options)

gender_options = [
    "Male",
    "Female",
    "Other",
    "Don't know",
    "Refused",
]
female = st.selectbox("Gender",gender_options)


def process_inputs(age, education, income, parent, married, gender):

    education_mapping = {edu: idx for idx, edu in enumerate(education_options)}
    education_num = education_mapping[education]

    income_mapping = {inc: idx for idx, inc in enumerate(income_options)}
    income_num = income_mapping[income]
    if income in ["Don't know", "Refused"]:
        income_num = np.nan

    parent_num = 1 if parent == "Yes" else 0

    married_num = 1 if married == "Married" else 0

    female_num = 1 if gender == "Female" else 0

    return [age, education_num, income_num, parent_num, married_num, female_num]

if st.button('Predict LinkedIn Usage'):
    processed_inputs = process_inputs(age, education, income, parent, married, female)

    input_df = pd.DataFrame([processed_inputs], columns=['age', 'education', 'income', 'parent', 'married', 'female'])

    prediction = logreg.predict(input_df)
    probability = logreg.predict_proba(input_df)[:, 1]

    st.subheader('Prediction')
    st.write('LinkedIn User' if prediction[0] else 'Not a LinkedIn User')
    st.subheader('Prediction Probability')
    st.write(f"The probability of the person using LinkedIn is: {probability[0]:.2f}")
