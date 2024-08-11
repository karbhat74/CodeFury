import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

# Set the app title and icon
st.set_page_config(page_title="Beautiful Streamlit App", page_icon=":sparkles:")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content {
        background-color: #fff;
    }
    .css-1d391kg {
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        color: #fff;
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Section 1", "Section 2"])

# Main content
st.title("Welcome to My Beautiful Streamlit App")

st.markdown(
    """
    This is a sample Streamlit app showcasing various components and styling techniques to make the interface more beautiful and user-friendly.
    """
)

if section == "Section 1":
    # Section 1 content
    st.header("Section 1: Image and Form")
    
    # Display an image
    image = Image.open("streamlit.png")
    st.image(image, caption="Beautiful Image", use_column_width=True)

    # Create a form
    with st.form("my_form"):
        st.write("Inside the form")
        slider_val = st.slider("Form slider")
        checkbox_val = st.checkbox("Form checkbox")

        # Every form must have a submit button
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Slider value:", slider_val, "Checkbox value:", checkbox_val)

elif section == "Section 2":
    # Section 2 content
    st.header("Section 2: Metrics and DataFrame")
    
    # Create columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", "70 °F", "1.2 °F")
    col2.metric("Wind", "9 mph", "-8%")
    col3.metric("Humidity", "86%", "4%")

    # Progress bar
    st.progress(70)

    # Expander
    with st.expander("See explanation"):
        st.write("Here you can add more details or explanation.")

    # Display a data frame
    df = pd.DataFrame(
        np.random.randn(50, 20),
        columns=('col %d' % i for i in range(20))
    )
    st.dataframe(df)

# Add a button
if st.button("Click me"):
    st.write("Button clicked!")
