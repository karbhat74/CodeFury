import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Function to preprocess the data
def preprocess_data(data, target_variable, features):
    # Adjust features list based on the existing columns
    existing_features = [feature for feature in features if feature in data.columns]
    
    # Use the adjusted features list to create X and y
    X = data[existing_features]
    y = data[target_variable]
    
    # Convert categorical variables to numerical
    X = pd.get_dummies(X)
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    return X, y

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        results[name] = {"accuracy": accuracy, "report": report, "model": model}
    
    return results

# Streamlit app
st.title("Disaster Prediction Model Comparison")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.write("Data Preview:")
    st.write(data.head())
    
    features = st.multiselect("Select Features", options=data.columns.tolist())
    target_variable = st.selectbox("Select Target Variable", options=data.columns.tolist())
    
    if st.button("Train and Evaluate Models"):
        if features and target_variable:
            X, y = preprocess_data(data, target_variable, features)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            st.write("### Model Evaluation Results")
            for name, result in results.items():
                st.write(f"#### {name}")
                st.write(f"Accuracy: {result['accuracy']}")
                st.write(pd.DataFrame(result['report']).transpose())
                
            # Store the trained models
            st.session_state['trained_models'] = {name: result['model'] for name, result in results.items()}
            st.session_state['features'] = features
            st.session_state['X_columns'] = X.columns

# Preprocess user input
def preprocess_user_input(user_input, X_columns):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_columns, fill_value=0)
    return input_df

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

if 'trained_models' in st.session_state:
    preprocessed_input = preprocess_user_input(user_input, st.session_state['X_columns'])
    
    # Model selection for prediction
    model_name = st.selectbox("Select Model for Prediction", options=list(st.session_state['trained_models'].keys()))
    selected_model = st.session_state['trained_models'][model_name]
    
    if st.button("Predict"):
        user_prediction = selected_model.predict(preprocessed_input)
        st.write(f"Prediction for the user input: {user_prediction[0]}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.sidebar.title("Disaster Prediction")
    st.sidebar.info("Upload your dataset, select features, and evaluate different models for disaster prediction.")
