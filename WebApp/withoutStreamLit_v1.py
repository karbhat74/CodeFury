import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from twilio.rest import Client

# Twilio credentials (should be handled securely in production)
account_sid = 'AC84782d595383c0e22583352bf451a54c'
auth_token = 'c721e801e44adfa2cc3913ae9c632441'
from_phone_number = '+14025881918'
to_phone_number = '+916393868175'

client = Client(account_sid, auth_token)

def send_notification(message_body):
    try:
        message = client.messages.create(
            to=to_phone_number, 
            from_=from_phone_number,
            body=message_body
        )
        return message.sid
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return None

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

# Load and preprocess data
data = pd.read_csv('disaster_data.csv', encoding='ISO-8859-1')
features = ['Disaster Group', 'Disaster Subgroup', 'Total Deaths','No. Injured']  # Replace with actual feature names
target_variable = 'Disaster Type'  # Replace with actual target variable name
X, y = preprocess_data(data, target_variable, features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
trained_models = {name: result['model'] for name, result in results.items()}
X_columns = X.columns

# Define user input (example values)
user_input = {
    'Magnitude': 7.2,
    'Total Deaths': 1500,
    'No. Injured': 3000,
    'No. Affected': 50000,
    'No. Homeless': 10000,
    'Disaster Group': 'Natural',
    'Disaster Subgroup': 'Geophysical',
    'Disaster Type': 'Earthquake',
}

def preprocess_user_input(user_input, X_columns):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_columns, fill_value=0)
    return input_df

# Predict disaster type
preprocessed_input = preprocess_user_input(user_input, X_columns)
selected_model = trained_models['Random Forest']  # Change model as needed
user_prediction = selected_model.predict(preprocessed_input)
predicted_disaster = user_prediction[0]

# Disaster response dictionary
response_dict = {
    "Earthquake": "1. Evacuate to an open space.\n2. Drop, cover, and hold on during shaking.\n3. Check for injuries and provide first aid.\n4. Be prepared for aftershocks.\n5. Follow local authorities' instructions.",
    # Add other disaster responses as necessary
}

# Print and send the prediction and response
print(f"Prediction for the user input: {predicted_disaster}")
print(f"Suggested Actions and Management Strategies:\n{response_dict.get(predicted_disaster, 'Follow local authorities’ instructions and stay safe.')}")

notification_message = f"Disaster Prediction: {predicted_disaster}\nSuggested Actions: {response_dict.get(predicted_disaster, 'Follow local authorities’ instructions and stay safe.')}"
message_sid = send_notification(notification_message)
if message_sid:
    print(f"Notification sent successfully. Message SID: {message_sid}")