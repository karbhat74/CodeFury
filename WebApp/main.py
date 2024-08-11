from multiprocessing.connection import Client
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
    
        
        # Disaster Management Response
        st.write("## Disaster Management Response")
        st.write(f"### Predicted Disaster Subtype: {user_prediction[0]}")
        
        # Suggested actions and management strategies
        response_dict = {
            "Earthquake": "1. Evacuate to an open space.\n2. Drop, cover, and hold on during shaking.\n3. Check for injuries and provide first aid.\n4. Be prepared for aftershocks.\n5. Follow local authorities' instructions.",
            "Volcanic activity": "1. Evacuate from the affected area.\n2. Avoid low-lying areas to escape lava flows.\n3. Wear masks to avoid inhaling ash.\n4. Protect property from ash fall.\n5. Follow evacuation orders from local authorities.",
            "Storm": "1. Secure your home and outdoor items.\n2. Evacuate if instructed by authorities.\n3. Stock up on emergency supplies.\n4. Stay indoors and away from windows.\n5. Monitor weather updates and follow local advisories.",
            "Flood": "1. Move to higher ground.\n2. Avoid walking or driving through floodwaters.\n3. Turn off utilities and move valuable items to higher levels.\n4. Follow evacuation orders.\n5. Stay informed through local news and weather reports.",
            "Wildfire": "1. Evacuate if advised by authorities.\n2. Remove flammable materials around your home.\n3. Close all windows and doors to prevent embers from entering.\n4. Wear protective clothing and masks.\n5. Monitor local news for updates and instructions.",
            "Drought": "1. Conserve water and use it efficiently.\n2. Implement drought-resistant landscaping.\n3. Support community water conservation efforts.\n4. Follow water use restrictions.\n5. Stay informed about drought conditions and forecasts."
            # Add more disaster management responses as needed
        }
        
        st.write(f"### Suggested Actions and Management Strategies:\n{response_dict.get(user_prediction[0], 'Follow local authorities’ instructions and stay safe.')}")
        
        notification_message = f"Disaster Prediction: {user_prediction[0]}\nSuggested Actions: {response_dict.get(user_prediction[0], 'Follow local authorities’ instructions and stay safe.')}" # type: ignore
        for i in to_phone_numbers: # type: ignore
            send_notification(notification_message,i) # type: ignore
        
        # Resources and contacts
        st.write("### Important Resources and Emergency Contacts")
        st.write("1. National Disaster Management Authority\n2. Local Emergency Services\n3. Red Cross\n4. FEMA (Federal Emergency Management Agency)\n5. Local Government Websites")


# Twilio credentials
account_sid = 'SK4683f7b5122953216b23ad0911a7e5ac'
auth_token = 'ekIIOEV4hpzlR6Ccx0RF1ONrqe8kFYzf'
from_phone_number = '+919980419338'
to_phone_numbers = ['+919108942601','+917411105495','+917892012242']


client = Client(account_sid, auth_token)

def send_notification(message_body,to_phone_number):
    try:
        message = client.messages.create(
            body=message_body,
            from_=from_phone_number,
            to=to_phone_number
        )
        st.success("Notification sent successfully.")
    except Exception as e:
        st.error(f"Failed to send notification: {e}")
        notification_message = f"Disaster Prediction: {user_prediction[0]}\nSuggested Actions: {response_dict.get(user_prediction[0], 'Follow local authorities’ instructions and stay safe.')}"
        for i in to_phone_numbers:
            send_notification(notification_message,i)
