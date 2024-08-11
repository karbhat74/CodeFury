import streamlit as st
import pandas as pd
import requests
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from twilio.rest import Client

# Twilio credentials
account_sid = 'SK4683f7b5122953216b23ad0911a7e5ac'
auth_token = 'ekIIOEV4hpzlR6Ccx0RF1ONrqe8kFYzf'
from_phone_number = '+919980419338 '
to_phone_numbers = ['+919108942601', '+917411105495', '+917892012242']

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

# Replace with your actual API key
API_KEY = '898faf64800f441290495339242007'

# Function to get weather data
def get_weather_data(location):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}"
    response = requests.get(url)
    return response.json()

# Function to get latitude and longitude of a location
def get_lat_lon(location):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(location)
    return location.latitude, location.longitude

# Function to create a map with weather data
def create_weather_map(location, weather_data):
    lat, lon = get_lat_lon(location)
    map_ = folium.Map(location=[lat, lon], zoom_start=10,
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Map data © <a href="https://www.esri.com/en-us/home">Esri</a> contributors')
    
    weather_info = f"""
    <b>Location:</b> {weather_data['location']['name']}, {weather_data['location']['country']}<br>
    <b>Temperature:</b> {weather_data['current']['temp_c']}°C ({weather_data['current']['temp_f']}°F)<br>
    <b>Condition:</b> {weather_data['current']['condition']['text']}<br>
    <b>Wind:</b> {weather_data['current']['wind_kph']} kph ({weather_data['current']['wind_mph']} mph)<br>
    <b>Humidity:</b> {weather_data['current']['humidity']}%<br>
    <b>Pressure:</b> {weather_data['current']['pressure_mb']} mb ({weather_data['current']['pressure_in']} in)<br>
    <b>Visibility:</b> {weather_data['current']['vis_km']} km ({weather_data['current']['vis_miles']} miles)<br>
    <b>UV Index:</b> {weather_data['current']['uv']}<br>
    <b>Feels Like:</b> {weather_data['current']['feelslike_c']}°C ({weather_data['current']['feelslike_f']}°F)
    """
    
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(weather_info, max_width=300),
        tooltip=weather_data['location']['name']
    ).add_to(map_)
    
    return map_

# Set Streamlit page configuration
st.set_page_config(page_title="Disaster Prediction and Weather App", layout="wide")

# Sidebar content
st.sidebar.title("Navigation")
if 'section' not in st.session_state:
    st.session_state.section = "Disaster Prediction"
section = st.sidebar.radio("Go to", ["Disaster Prediction", "Model Results", "Weather Map"], index=0 if st.session_state.section == "Disaster Prediction" else 1)

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
target_variable = 'Disaster Subtype'
X, y = preprocess_data(data, target_variable, features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
st.session_state['trained_models'] = {name: result['model'] for name, result in results.items()}
st.session_state['features'] = features
st.session_state['X_columns'] = X.columns

# Section 1: Disaster Prediction
if section == "Disaster Prediction":
    st.title("🌀 ThreatMatrix 🌀")

    if 'trained_models' in st.session_state:
        with st.container():
            st.markdown("## Enter Disaster Details")
            
            with st.form(key='disaster_prediction_form'):
                col1, col2 = st.columns(2)

                with col1:
                    location = st.text_input("Location", "Mysuru")
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
                        'Location': location,
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
                    st.session_state['user_prediction'] = user_prediction[0]
                    st.session_state['response_dict'] = {
            "Earthquake": "1. Evacuate to an open space.\n2. Drop, cover, and hold on during shaking.\n3. Check for injuries and provide first aid.\n4. Be prepared for aftershocks.\n5. Follow local authorities' instructions.",
            "Volcanic activity": "1. Evacuate from the affected area.\n2. Avoid low-lying areas to escape lava flows.\n3. Wear masks to avoid inhaling ash.\n4. Protect property from ash fall.\n5. Follow evacuation orders from local authorities.",
            "Mass movement (dry)": "1. Evacuate the area immediately.\n2. Avoid steep slopes and landslide-prone areas.\n3. Stay informed about local warnings.\n4. Keep emergency supplies ready.\n5. Follow instructions from local authorities.",
            "Storm": "1. Secure your home and outdoor items.\n2. Evacuate if instructed by authorities.\n3. Stock up on emergency supplies.\n4. Stay indoors and away from windows.\n5. Monitor weather updates and follow local advisories.",
            "Flood": "1. Move to higher ground.\n2. Avoid walking or driving through floodwaters.\n3. Turn off utilities and move valuable items to higher levels.\n4. Follow evacuation orders.\n5. Stay informed through local news and weather reports.",
            "Mass movement (wet)": "1. Evacuate the area immediately.\n2. Avoid steep slopes and landslide-prone areas.\n3. Stay informed about local warnings.\n4. Keep emergency supplies ready.\n5. Follow instructions from local authorities.",
            "Wildfire": "1. Evacuate if advised by authorities.\n2. Remove flammable materials around your home.\n3. Close all windows and doors to prevent embers from entering.\n4. Wear protective clothing and masks.\n5. Monitor local news for updates and instructions.",
            "Extreme temperature": "1. Stay indoors during extreme temperatures.\n2. Drink plenty of water and avoid strenuous activities.\n3. Check on vulnerable individuals (elderly, children).\n4. Use air conditioning or visit cooling centers.\n5. Follow health advisories and stay informed.",
            "Drought": "1. Conserve water and use it efficiently.\n2. Implement drought-resistant landscaping.\n3. Support community water conservation efforts.\n4. Follow water use restrictions.\n5. Stay informed about drought conditions and forecasts.",
            "Glacial lake outburst": "1. Evacuate downstream areas.\n2. Monitor early warning systems.\n3. Avoid riverbanks and low-lying areas.\n4. Follow local authorities' evacuation orders.\n5. Stay informed about potential risks and updates.",
            "Insect infestation": "1. Use appropriate insect control measures.\n2. Protect crops and livestock.\n3. Follow agricultural advisories.\n4. Implement sanitation and hygiene practices.\n5. Stay informed about infestation levels and control methods.",
            "Epidemic": "1. Follow public health guidelines.\n2. Practice good hygiene (handwashing, masks).\n3. Get vaccinated if available.\n4. Avoid crowded places and maintain social distancing.\n5. Stay informed about health advisories and updates.",
            "Animal accident": "1. Stay calm and avoid provoking animals.\n2. Seek immediate medical attention if injured.\n3. Report incidents to local authorities.\n4. Follow wildlife safety guidelines.\n5. Stay informed about potential risks in the area.",
            "Impact": "1. Evacuate the area if necessary.\n2. Stay indoors and away from windows.\n3. Follow local authorities' instructions.\n4. Keep emergency supplies ready.\n5. Stay informed about impact risks and updates.",
            "Space weather": "1. Protect electronic devices from electromagnetic pulses.\n2. Follow space weather forecasts.\n3. Be prepared for communication disruptions.\n4. Follow guidelines from space weather agencies.\n5. Stay informed about potential impacts and updates."
        }
                    
                    for to_phone_number in to_phone_numbers:
                        send_notification(f"Predicted Disaster: {user_prediction[0]}, Magnitude: {magnitude}, Total Deaths: {total_deaths}, No. Injured: {no_injured}, No. Affected: {no_affected}, No. Homeless: {no_homeless}, Disaster Group: {disaster_group}, Disaster Subgroup: {disaster_subgroup}", to_phone_number)

                    st.session_state['prediction_location'] = location  # Save the location for Weather Map section
                    st.session_state.section = "Model Results"
                    st.experimental_rerun()

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
        for to_phone_number in to_phone_numbers:
            send_notification(notification_message, to_phone_number)

# Section 3: Weather Map
elif section == "Weather Map":
    st.title('🌦️ Weather Map 🌦️')

    location = st.session_state.get('prediction_location', 'Mysuru')  # Default to Mysuru if no location available

    if location:
        weather_data = get_weather_data(location)
        if 'error' not in weather_data:
            st.write(f"### Weather data for {location}:")
            
            # Extract and display detailed weather information
            location_name = weather_data['location']['name']
            country = weather_data['location']['country']
            temp_c = weather_data['current']['temp_c']
            temp_f = weather_data['current']['temp_f']
            condition_text = weather_data['current']['condition']['text']
            wind_kph = weather_data['current']['wind_kph']
            wind_mph = weather_data['current']['wind_mph']
            humidity = weather_data['current']['humidity']
            pressure_mb = weather_data['current']['pressure_mb']
            pressure_in = weather_data['current']['pressure_in']
            vis_km = weather_data['current']['vis_km']
            vis_miles = weather_data['current']['vis_miles']
            uv = weather_data['current']['uv']
            feelslike_c = weather_data['current']['feelslike_c']
            feelslike_f = weather_data['current']['feelslike_f']
            
            st.write(f"**Location:** {location_name}, {country}")
            st.write(f"**Temperature:** {temp_c}°C ({temp_f}°F)")
            st.write(f"**Condition:** {condition_text}")
            st.write(f"**Wind:** {wind_kph} kph ({wind_mph} mph)")
            st.write(f"**Humidity:** {humidity}%")
            st.write(f"**Pressure:** {pressure_mb} mb ({pressure_in} in)")
            st.write(f"**Visibility:** {vis_km} km ({vis_miles} miles)")
            st.write(f"**UV Index:** {uv}")
            st.write(f"**Feels Like:** {feelslike_c}°C ({feelslike_f}°F)")

            weather_map = create_weather_map(location, weather_data)
            folium_static(weather_map, width=800, height=600)
        else:
            st.write("Weather data could not be retrieved. Please try again later.")

# Footer
st.markdown("---")
st.write("Developed by karbhat")
