import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from twilio.rest import Client
import folium
from streamlit_folium import st_folium

# Twilio credentials
account_sid = 'AC84782d595383c0e22583352bf451a54c'
auth_token = '60f6060dedcf1a6387fd64feeab88611'
from_phone_number = '+14025881918'
to_phone_numbers = ['+919696781896', '+916393868175', '+919794401041']

client = Client(account_sid, auth_token)

def send_notification(message_body, to_phone_number):
    try:
        message = client.messages.create(
            body=message_body,
            from_=from_phone_number,
            to=to_phone_number
        )
        st.success("Notification sent successfully.")
    except Exception as e:
        st.error(f"Failed to send notification: {e}")

# Set Streamlit page configuration
st.set_page_config(page_title="Disaster Prediction App", layout="wide")

# Sidebar content
st.sidebar.title("Navigation")
if 'section' not in st.session_state:
    st.session_state.section = "Disaster Prediction"
section = st.sidebar.radio("Go to", ["Disaster Prediction", "Model Results"], index=0 if st.session_state.section == "Disaster Prediction" else 1)

# Load dataset from file (replace 'disaster_data.csv' with your file path)
dataset_path = 'disaster_data.csv'
data = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Function to preprocess the data
def preprocess_data(data, target_variable, features):
    existing_features = [feature for feature in features if feature in data.columns]
    X = data[existing_features]
    y = data[target_variable]
    X = pd.get_dummies(X)
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

# Preprocess user input
def preprocess_user_input(user_input, X_columns):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_columns, fill_value=0)
    return input_df

# Train models on inbuilt data
features = ['Magnitude', 'Total Deaths', 'No. Injured', 'No. Affected', 'No. Homeless', 'Disaster Group', 'Disaster Subgroup', 'Disaster Type']
target_variable = 'Disaster Type'
X, y = preprocess_data(data, target_variable, features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
st.session_state['trained_models'] = {name: result['model'] for name, result in results.items()}
st.session_state['features'] = features
st.session_state['X_columns'] = X.columns

# Section 1: Disaster Prediction
if section == "Disaster Prediction":
    st.title("🌀 Disaster Prediction 🌀")

    if 'trained_models' in st.session_state:
        with st.container():
            st.markdown("## Enter Disaster Details")
            
            with st.form(key='disaster_prediction_form'):
                col1, col2 = st.columns(2)

                with col1:
                    magnitude = st.slider("Magnitude", min_value=0.0, max_value=10.0, value=7.2, step=0.1)
                    total_deaths = st.slider("Total Deaths", min_value=0, max_value=10000, value=1500)
                    no_injured = st.slider("No. Injured", min_value=0, max_value=10000, value=3000)
                    no_affected = st.slider("No. Affected", min_value=0, max_value=100000, value=50000)
                    no_homeless = st.slider("No. Homeless", min_value=0, max_value=10000, value=10000)

                with col2:
                    disaster_group = st.selectbox("Disaster Group", options=['Natural', 'Technological'])
                    disaster_subgroup = st.selectbox("Disaster Subgroup", options=['Geophysical', 'Hydrological', 'Meteorological', 'Climatological', 'Biological', 'Extra-terrestrial'])
                    disaster_type = st.selectbox("Disaster Type", options=['Earthquake', 'Volcanic activity', 'Mass movement (dry)', 'Storm', 'Flood', 'Mass movement (wet)', 'Wildfire', 'Extreme temperature', 'Drought', 'Glacial lake outburst', 'Insect infestation', 'Epidemic', 'Animal accident', 'Impact', 'Space weather'])

                submit_button = st.form_submit_button(label='Predict')

                if submit_button:
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

                    preprocessed_input = preprocess_user_input(user_input, st.session_state['X_columns'])
                    model_name = st.selectbox("Select Model for Prediction", options=list(st.session_state['trained_models'].keys()))
                    selected_model = st.session_state['trained_models'][model_name]

                    user_prediction = selected_model.predict(preprocessed_input)
                    st.write(f"**Predicted Disaster Type: {user_prediction[0]}**")
                    st.session_state['user_prediction'] = user_prediction[0]
                    
                    for to_phone_number in to_phone_numbers:
                        send_notification(f"Predicted Disaster: {user_prediction[0]}, Magnitude: {magnitude}, Total Deaths: {total_deaths}, No. Injured: {no_injured}, No. Affected: {no_affected}, No. Homeless: {no_homeless}, Disaster Group: {disaster_group}, Disaster Subgroup: {disaster_subgroup}", to_phone_number)

# Section 2: Model Results
elif section == "Model Results":
    st.title("📊 Model Results 📊")

    st.write("### Data Preview")
    st.write(data.head())

    st.write("### Model Evaluation Results")
    comparison_df = pd.DataFrame({
        "Model": [name for name in results.keys()],
        "Accuracy": [result['accuracy'] for result in results.values()],
        "Precision (Macro)": [result['report']['macro avg']['precision'] for result in results.values()],
        "Recall (Macro)": [result['report']['macro avg']['recall'] for result in results.values()],
        "F1 Score (Macro)": [result['report']['macro avg']['f1-score'] for result in results.values()]
    })
    st.write(comparison_df)

    if 'user_prediction' in st.session_state:
        st.write("### Disaster Management Response")
        st.write(f"**Predicted Disaster Type:** {st.session_state['user_prediction']}")
        
        response_dict = st.session_state.get('response_dict', {})
        st.write(f"**Suggested Actions and Management Strategies:**\n{response_dict.get(st.session_state['user_prediction'], 'Follow local authorities’ instructions and stay safe.')}")
        
        st.write("### Important Resources and Emergency Contacts")
        st.write("1. National Disaster Management Authority\n2. Local Emergency Services\n3. Red Cross\n4. FEMA (Federal Emergency Management Agency)\n5. Local Government Websites")
        
        notification_message = f"Disaster Prediction: {st.session_state['user_prediction']}\nSuggested Actions: {response_dict.get(st.session_state['user_prediction'], 'Follow local authorities’ instructions and stay safe.')}"
        for i in to_phone_numbers:
            send_notification(notification_message, i)

        # Adding the live cloud map
        st.write("### Live Cloud Map")
        map_center = [26.4499, 80.3319]  # Coordinates of Kanpur, Uttar Pradesh, India
        disaster_map = folium.Map(location=map_center, zoom_start=6)

        # Add some sample markers
        folium.Marker(
            location=[26.4499, 80.3319],
            popup="Kanpur, Uttar Pradesh",
            icon=folium.Icon(color='blue')
        ).add_to(disaster_map)

        # Display the map
        st_folium(disaster_map, width=700, height=500)

# Footer
st.markdown("---")
st.write("Developed by Perfect cube")
