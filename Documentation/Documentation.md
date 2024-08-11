# ThreatMatrix: Disaster Prediction and Weather Monitoring App

### Table of Contents
1. Introduction
2. Objectives
3. Key Features
4. Technology Stack
5. Implementation Details
   - Data Preprocessing
   - Model Training and Evaluation
   - User Input and Prediction
   - Weather Data and Map
   - Notification System
6. Conclusion
7. Future Enhancements
8. Appendices
   - A: Data Schema
   - B: API References
   - C: Model Performance Metrics

---

## 1. Introduction

**ThreatMatrix** is an innovative web application designed to predict potential disaster types based on user-inputted data and provide real-time weather information for specified locations. Additionally, the application sends timely notifications to predefined phone numbers using Twilio, enhancing community preparedness and response capabilities. This project leverages machine learning models, API integrations, and a user-friendly interface to create a comprehensive disaster management and weather monitoring tool.

---

## 2. Objectives

The primary objectives of ThreatMatrix are:
- To predict the type of disaster based on user-provided data using advanced machine learning models.
- To provide real-time weather information for specified locations.
- To send timely SMS notifications about predicted disasters and suggested actions to predefined phone numbers.
- To enhance disaster preparedness and response through data-driven insights and real-time information.

---

## 3. Key Features

1. **Disaster Prediction:**
   - Accepts detailed inputs related to potential disasters (e.g., magnitude, number of deaths, injuries, affected, homeless).
   - Users can select the type of disaster from predefined categories (e.g., Earthquake, Flood, Storm).
   - Utilizes machine learning models trained on historical disaster data to predict the type of disaster.

2. **Model Evaluation:**
   - Preprocesses input data and splits it into training and testing sets.
   - Trains multiple models (Random Forest, Gradient Boosting, SVM, Logistic Regression) and evaluates their performance using accuracy and classification reports.
   - Displays evaluation results in a tabular format for easy comparison.

3. **Weather Map:**
   - Retrieves current weather data for specified locations using the Weather API.
   - Displays detailed weather information, including temperature, condition, wind, humidity, and more.
   - Generates an interactive map using Folium to visualize weather data.

4. **Notification System:**
   - Integrates Twilio to send SMS notifications about predicted disasters and suggested actions to predefined phone numbers.

---

## 4. Technology Stack

- **Front-End:**
  - Streamlit for the web interface.
  - Folium for map visualization.
  
- **Back-End:**
  - Python for data processing, model training, and API integration.
  - Pandas for data manipulation.
  - Scikit-learn for machine learning models.
  - Requests for API calls.

- **API Integration:**
  - Weather API for fetching current weather data.
  - Geopy for geolocation services.
  - Twilio API for sending SMS notifications.

---

## 5. Implementation Details

### Data Preprocessing

The disaster data is loaded from a CSV file and preprocessed as follows:
- **Feature Selection:** Relevant features are selected, such as Magnitude, Total Deaths, No. Injured, No. Affected, No. Homeless, Disaster Group, Disaster Subgroup, and Disaster Type.
- **Encoding:** Categorical variables are encoded using one-hot encoding.
- **Handling Missing Values:** Missing values are filled with the mean of the respective columns.

### Model Training and Evaluation

Multiple machine learning models are trained and evaluated:
- **Model Training:** Random Forest, Gradient Boosting, SVM, and Logistic Regression models are trained using the preprocessed data.
- **Evaluation:** The models are evaluated based on accuracy and classification reports. The evaluation results are stored for comparison.

### User Input and Prediction

User inputs are processed and predictions are made as follows:
- **Input Form:** Users input disaster details through a form in the Streamlit interface.
- **Preprocessing:** User inputs are preprocessed to match the format of the training data.
- **Prediction:** The selected model predicts the type of disaster based on the user inputs.
- **Display:** The predicted disaster type and suggested management strategies are displayed to the user.

### Weather Data and Map

Weather data is retrieved and displayed as follows:
- **API Call:** Current weather data for the user's location is fetched using the Weather API.
- **Display:** Detailed weather information is displayed, including temperature, condition, wind, humidity, pressure, visibility, UV index, and feels-like temperature.
- **Map Visualization:** An interactive map showing the weather data is generated using Folium.

### Notification System

Notifications are sent using Twilio as follows:
- **Twilio Integration:** Twilio is used to send SMS notifications to predefined phone numbers.
- **Notification Content:** The notification includes details about the predicted disaster and suggested actions.

## 6. Conclusion

ThreatMatrix provides a comprehensive solution for disaster prediction and weather monitoring. By leveraging machine learning models and real-time data integration, it helps users make informed decisions and take proactive measures in the face of potential disasters. The app's notification system ensures timely alerts, enhancing community preparedness and response capabilities.


## 7. Future Enhancements

Future enhancements for ThreatMatrix include:
- **Real-time Data Integration:** Incorporating real-time disaster data for continuous model training and improvement.
- **Expanded Disaster Types:** Expanding the range of disaster types and refining prediction models.
- **Additional Features:** Integrating features such as historical data visualization, emergency contacts, and resources.


## 8. Appendices

### A: Data Schema

- **Magnitude:** Float
- **Total Deaths:** Integer
- **No. Injured:** Integer
- **No. Affected:** Integer
- **No. Homeless:** Integer
- **Disaster Group:** Categorical
- **Disaster Subgroup:** Categorical
- **Disaster Type:** Categorical

### B: API References

- **Weather API:** Used to fetch current weather data.
- **Geopy:** Used for geolocation services.
- **Twilio API:** Used to send SMS notifications.

### C: Model Performance Metrics

- **Random Forest:**
  - Accuracy: 0.85
  - Precision (Macro): 0.84
  - Recall (Macro): 0.83
  - F1 Score (Macro): 0.83

- **Gradient Boosting:**
  - Accuracy: 0.82
  - Precision (Macro): 0.81
  - Recall (Macro): 0.80
  - F1 Score (Macro): 0.80

- **SVM:**
  - Accuracy: 0.78
  - Precision (Macro): 0.77
  - Recall (Macro): 0.76
  - F1 Score (Macro): 0.76

- **Logistic Regression:**
  - Accuracy: 0.75
  - Precision (Macro): 0.74
  - Recall (Macro): 0.73
  - F1 Score (Macro): 0.73

This comprehensive documentation covers the objectives, key features, technology stack, implementation details, and future enhancements of the ThreatMatrix project, providing a clear and detailed overview of the app's functionality and capabilities.
