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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have the balanced_train and test_data available
# If not, make sure to load your datasets appropriately
balanced_train = pd.read_csv('balanced_train.csv', index_col='SN')
test_data = pd.read_csv('test_data.csv', index_col='SN')
query_columns = pd.read_csv('buttons.csv')

st.image('sanku_logo.png', width=200)

# Add description
st.markdown("""
# Dosifier Prototype Model (V1)

This is the first version (V1) of the prototype model, intended for full deployment into production. The model classifies Dosifier Offline Technical and non-technical issues based on the input features as displayed below.

**Note:** The test and train data span from January 2022 to January 2024.
""")

# Create a list of models to fit
models = [
    KNeighborsClassifier(),
    AdaBoostClassifier(algorithm='SAMME'),  # Explicitly set algorithm to "SAMME"
    GradientBoostingClassifier(),
    RandomForestClassifier()
]

# Fit each model to the transformed dataset
for model in models:
    model.fit(balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'])

# Streamlit App
st.title("Model Deployment with Streamlit")

# User input for features
st.sidebar.header('Query Columns')
# Create dropdowns for selecting columns
selected_date_added = st.sidebar.selectbox("Select Date Added Column", query_columns['DATE ADDED'].unique())
selected_sn = st.sidebar.selectbox("Select Serial Number Column", query_columns['SN'].unique())

# Display user input
st.write("User Input:")
st.write(f"Selected Date Added: {selected_date_added}")
st.write(f"Selected Serial Number: {selected_sn}")

# Determine CATEGORY based on selected columns
selected_row = query_columns[
    (query_columns['DATE ADDED'] == selected_date_added) &
    (query_columns['SN'] == selected_sn)
]

if not selected_row.empty:
    predicted_category = selected_row['CATEGORY'].values[0]
    st.write(f"Predicted Category: {predicted_category}")
else:
    st.warning("No matching row found for the selected columns.")


st.sidebar.header('Input Features')
user_input = {}
for feature in balanced_train.drop(['CATEGORY'], axis=1).columns:
    user_input[feature] = st.sidebar.slider(f'Select {feature}', float(balanced_train[feature].min()), float(balanced_train[feature].max()))

# Create a dataframe with user input
user_input_df = pd.DataFrame([user_input])

# Display user input
st.write("User Input:")
st.write(user_input_df)

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
    
# Checkbox to control result visibility
show_confusion_matrix = st.checkbox("Show Confusion Matrix")
show_results = st.checkbox("Show Results", value=False)

# Display confusion matrix
if show_confusion_matrix:
    confusion_mat = confusion_matrix(balanced_train['CATEGORY'], selected_model.predict(balanced_train.drop(['CATEGORY'], axis=1)))

    # Display confusion matrix as heatmap
    st.write("Confusion Matrix Heatmap:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=selected_model.classes_, yticklabels=selected_model.classes_)
    st.pyplot(fig)

    # Display classification report
    st.header("Classification Report")
    classification_rep = classification_report(balanced_train['CATEGORY'], selected_model.predict(balanced_train.drop(['CATEGORY'], axis=1)))
    st.text_area("Classification Report", classification_rep, height=200)

# Display model evaluation results
st.header("Model Evaluation Results")
for model in models:
    accuracy = cross_val_score(model, balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'], cv=5)
    st.write(f"Accuracy of {model.__class__.__name__}: {accuracy.mean()}")

# Predictions and probabilities
# Display predictions and probabilities with two decimal places
predictions = selected_model.predict(test_data.drop(['CATEGORY'], axis=1))
probabilities = selected_model.predict_proba(test_data.drop(['CATEGORY'], axis=1))
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(round(dosifier_predictions_final, 2))

# Handle commas in the index if it's a string and is convertible to an integer
dosifier_predictions_final.index = dosifier_predictions_final.index.map(lambda x: int(str(x).replace(',', '')) if str(x).replace(',', '').isdigit() else x)

# Add input box and button for searching SN
st.header('Search by SN')
search_sn = st.text_input("Enter SN:", "")
search_button = st.button("Search")

# Check if the Search button is clicked
if search_button:
    search_sn_cleaned = search_sn.replace(',', '')
    if 'dosifier_predictions_final' in locals() and search_sn_cleaned in dosifier_predictions_final.index:
        st.header(f"Predictions and Probabilities for SN {search_sn_cleaned}")

        # Obtain predicted category from predictions_df
        predicted_category = dosifier_predictions_final.idxmax(axis=1).loc[search_sn_cleaned]
        
        # Obtain probabilities from dosifier_predictions_final
        probabilities_df = dosifier_predictions_final.loc[[search_sn_cleaned]].transpose().reset_index()
        probabilities_df.columns = ['Category', 'Probability']

        # Display predicted category and probabilities as a DataFrame
        st.write("Predicted Category:")
        st.write(pd.DataFrame({'Category': [predicted_category]}))
        st.write("Probabilities:")
        st.dataframe(probabilities_df)

    elif show_results:
        st.warning(f"No data found for SN {search_sn_cleaned}")
