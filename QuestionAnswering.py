from os import path
import streamlit as st
import tensorflow as tf
from transformers import ElectraTokenizerFast, TFElectraForQuestionAnswering

model_hf = "nguyennghia0902/electra-small-discriminator_0.0001_16_15e"
tokenizer = ElectraTokenizerFast.from_pretrained(model_hf)
reload_model = TFElectraForQuestionAnswering.from_pretrained(model_hf)

@st.cache_resource
def predict(question, context):
    inputs = tokenizer(question, context, return_offsets_mapping=True,return_tensors="tf",max_length=512, truncation=True)
    offset_mapping = inputs.pop("offset_mapping")
    outputs = reload_model(**inputs)
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    start_char = offset_mapping[0][answer_start_index][0]
    end_char = offset_mapping[0][answer_end_index][1]
    predicted_answer_text = context[start_char:end_char]
    
    return predicted_answer_text

def main():
    st.set_page_config(page_title="Question Answering", page_icon="📝")

    # giving a title to our page
    col1, col2 = st.columns([2, 1])
    col1.title("Question Answering")

    col2.link_button("Explore my model", "https://huggingface.co/"+model_hf)

    
    text = st.text_area(
        "CONTEXT: Please enter a context:",
        placeholder="Enter your context here",
        height=200,
    )
    question = st.text_area(
        "QUESTION: Please enter a question:",
        placeholder="Enter your question here",
        height=5,
    )
    prediction = ""

    upload_file = st.file_uploader("QUESTION: Or upload a file with some questions", type=["txt"])
    if upload_file is not None:
        question = upload_file.read().decode("utf-8")

        for line in question.splitlines():
            line = line.strip()
            if not line:
                continue

            prediction = predict(line, text)

            st.success(line + "\n\nAnswer: " + prediction)

    
    # Create a prediction button
    elif st.button("Predict"):
        prediction = ""
        stripped_text = text.strip()
        if not stripped_text:
            st.error("Please enter a context.")
            return
        stripped_question = question.strip()
        if not stripped_question:
            st.error("Please enter a question.")
            return

        prediction = predict(stripped_question, stripped_text)
        if prediction == "":
            st.error(prediction)
        else:
            st.success(prediction)

if __name__ == "__main__":
    main()