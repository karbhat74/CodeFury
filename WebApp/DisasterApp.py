import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('disaster_data.csv', encoding='ISO-8859-1')

# Define features and target variable based on the dataset
features = ['Magnitude', 'Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless',
            'Disaster Group', 'Disaster Subgroup', 'Disaster Type']
target_variable = 'Disaster Subtype'  # Replace with the actual target variable name

# Adjust features list based on the existing columns
existing_features = [feature for feature in features if feature in data.columns]

# Use the adjusted features list to create X and y
X = data[existing_features]
y = data[target_variable]  # Replace with the correct target variable name

# Preprocessing: Convert categorical variables to numerical
X = pd.get_dummies(X)

# Handle missing values (e.g., fill with mean or median)
X.fillna(X.mean(), inplace=True)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (e.g., RandomForest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=0)

st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_rep)

# Define a function to preprocess user input
def preprocess_user_input(user_input):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    
    # Align user input with training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    return input_df

# Streamlit user input
st.header("Disaster Prediction")

magnitude = st.number_input("Magnitude", min_value=0.0, value=7.2)
total_deaths = st.number_input("Total Deaths", min_value=0, value=1500)
no_injured = st.number_input("No. Injured", min_value=0, value=3000)
no_affected = st.number_input("No. Affected", min_value=0, value=50000)
no_homeless = st.number_input("No. Homeless", min_value=0, value=10000)

disaster_group = st.selectbox("Disaster Group", options=['Natural', 'Technological'])
disaster_subgroup = st.selectbox("Disaster Subgroup", options=['Geophysical', 'Hydrological', 'Meteorological', 'Climatological', 'Biological', 'Extra-terrestrial'])
disaster_type = st.selectbox("Disaster Type", options=['Earthquake', 'Volcanic activity', 'Mass movement (dry)', 'Storm', 'Flood', 'Mass movement (wet)', 'Wildfire', 'Extreme temperature', 'Drought', 'Glacial lake outburst', 'Insect infestation', 'Epidemic', 'Animal accident', 'Impact', 'Space weather'])

# Preprocess user input
user_input = {
    'Magnitude': magnitude,
    'Total Deaths': total_deaths,
    'No. Injured': no_injured,
    'No. Affected': no_affected,
    'No. Homeless': no_homeless,
    'Disaster Group': disaster_group,
    'Disaster Subgroup': disaster_subgroup,
    'Disaster Type': disaster_type,
}

preprocessed_input = preprocess_user_input(user_input)

# Make prediction
if st.button("Predict"):
    user_prediction = model.predict(preprocessed_input)
    st.write("Prediction for the user input:", user_prediction[0])

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.sidebar.title("Disaster Prediction")
    st.sidebar.info("Enter the parameters for disaster prediction on the left.")
