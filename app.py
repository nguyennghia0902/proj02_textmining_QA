import streamlit as st
from st_pages import Page, show_pages

st.set_page_config(page_title="Question Answering", page_icon="🏠")

show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page(
            "SampleQA.py", "Sample in Dataset", "📝"
        ),
        Page(
            "QuestionAnswering.py", "Question Answering", "📝"
        ),
    ]
)

st.title("Project in Text Mining and Application")
st.header("Question Answering use a pre-trained model - ELECTRA")
st.markdown(
    """
    **Team members:**
    | Student ID | Full Name                | Email                          |
    | ---------- | ------------------------ | ------------------------------ |
    | 1712603    | Lê Quang Nam             | 1712603@student.hcmus.edu.vn   |
    | 19120582   | Lê Nhựt Minh             | 19120582@student.hcmus.edu.vn  |
    | 19120600   | Bùi Nguyên Nghĩa         | 19120600@student.hcmus.edu.vn  |
    | 21120198   | Nguyễn Thị Lan Anh       | 21120198@student.hcmus.edu.vn  |
    """
)

st.header("The Need for Question Answering")
st.markdown(
    """
    In the rapidly advancing field of Natural Language Processing (NLP), the Question Answering (QA) 
    task has become increasingly essential. QA systems are pivotal for efficient information retrieval, 
    enabling users to obtain precise answers to their queries quickly. This is particularly valuable in 
    domains such as customer service, education, and healthcare, where timely and accurate information 
    is crucial.
    """
)

st.header("Technology used")
st.markdown(
    """
    The ELECTRA model, specifically the "google/electra-small-discriminator" used here, 
    is a deep learning model in the field of natural language processing (NLP) developed 
    by Google. This model is an intelligent variation of the supervised learning model 
    based on the Transformer architecture, designed to understand and process natural language efficiently.
    For this Question Answering task, we choose two related classes: ElectraTokenizerFast and 
    TFElectraForQuestionAnswering to implement.
    """
)