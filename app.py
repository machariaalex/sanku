import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming you have the balanced_train and test_data available
# If not, make sure to load your datasets appropriately
balanced_train = pd.read_csv('balan.csv')

test_data = pd.read_csv('test.csv')

st.image('sanku_logo.png', width=200)

# Add description
st.markdown("""
            
# Dosifier Prototype Model (V1)

This is the first version (V1) of the prototype model, intended for full deployment into production. The model classifies Dosifier Offline Technical and non-technical issues based on the input features as displayed below.

**Note:** The test and train data span from January 2022 to January 2024.
""")

# Create a list of models to fit
models = [KNeighborsClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]

# Fit each model to the transformed dataset
for model in models:
    model.fit(balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'])

# Streamlit App
st.title("Model Deployment with Streamlit")

# User input for features
st.sidebar.header('Input Features')
user_input = {}
for feature in balanced_train.drop(['CATEGORY'], axis=1).columns:
    user_input[feature] = st.sidebar.slider(f'Select {feature}', float(balanced_train[feature].min()), float(balanced_train[feature].max()))

# Create a dataframe with user input
user_input_df = pd.DataFrame([user_input])

# Display user input with two decimal places
st.write("User Input:")
st.write(round(user_input_df, 2))

# Dropdown to select the model
st.sidebar.header('Select Model')
selected_model_name = st.sidebar.selectbox("Select a model", [model.__class__.__name__ for model in models])
selected_model = next((model for model in models if model.__class__.__name__ == selected_model_name), None)

if selected_model is None:
    st.warning("Invalid model selected.")
    st.stop()

# Predict the category using the selected model and general features
prediction_general = selected_model.predict(user_input_df)
st.write(f"Predicted Category For General Features: {prediction_general[0]}")
    
# Display confusion matrix
if st.checkbox("Show Confusion Matrix"):
    confusion_mat = confusion_matrix(balanced_train['CATEGORY'], selected_model.predict(balanced_train.drop(['CATEGORY'], axis=1)))

    # Display confusion matrix as heatmap
    st.write("Confusion Matrix Heatmap:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Oranges", xticklabels=selected_model.classes_, yticklabels=selected_model.classes_)
    st.pyplot(fig)

    # Display classification report
    st.header("Classification Report")
    classification_rep = classification_report(balanced_train['CATEGORY'], selected_model.predict(balanced_train.drop(['CATEGORY'], axis=1)))
    st.text_area("Classification Report", classification_rep, height=200)

# Display model evaluation results with two decimal places
st.header("Model Evaluation Results")
for model in models:
    accuracy = cross_val_score(model, balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'], cv=5)
    st.write(f"Accuracy of {model.__class__.__name__}: {round(accuracy.mean(), 2)}")

# Display predictions and probabilities with two decimal places
predictions = selected_model.predict(test_data.drop(['CATEGORY'], axis=1))
probabilities = selected_model.predict_proba(test_data.drop(['CATEGORY'], axis=1))
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(round(dosifier_predictions_final, 2))
