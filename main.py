import streamlit as st
import requests
from PIL import Image

image2=Image.open('assets/logo.jpg')
st.set_page_config(
    page_title="LoanWise",
    page_icon=image2,
)

image = Image.open('assets/poster.png')
st.image(image, caption='LOANWISE')
st.write("# Welcome to LOANWISE 👋")


with st.sidebar:
    st.write(f'''
                        <a target="blank" href="https://loanwise-eda.vercel.app/">
                            <button style=" border-color: orange; padding:10px 20px;   background-color: #fa6400f0; color:white;  border: none;  border-radius: .25rem;" >
                                View EDA report
                            </button>
                        </a>
                        ''',
                        unsafe_allow_html=True
                    ) 




st.markdown(
    """

    ## A step-by-step guide 

    The process is quite straightforward. BOB offers loans to eligible applicants with strong financial profiles. 
    Individuals need to provide their basic personal, employment, income and property details to know if you are fit to apply for a loan.


    ### Loan page
    
    - Offer all relevant details such as loan amount, loan history, income, etc.
    - Click on the submit option once you have filled in all the details.
    - Our algorithm will assess your eligibility based on the details provided by you and you will be awarded with a `yes` or a `no`.
"""
)

# Function to send data to the website API and get the prediction
def get_prediction(data):
    url = "http://localhost:8501/predict"  # Replace with the actual website URL
    response = requests.post(url, json=data)
    prediction = response.json()["prediction"]
    return prediction

# Assuming you have some user input data, replace this with your actual user input
user_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
prediction = get_prediction(user_input)

# Display the prediction in Streamlit
st.write("Prediction:", prediction)