import streamlit as st
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

