from os import path
import streamlit as st
import tensorflow as tf
import random
from transformers import ElectraTokenizerFast, TFElectraForQuestionAnswering
from datasets import Dataset, DatasetDict, load_dataset

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
    st.set_page_config(page_title="Sample in Dataset", page_icon="📝")

    # giving a title to our page
    col1, col2 = st.columns([2, 1])
    col1.title("Sample in Dataset")

    new_data = load_dataset("nguyennghia0902/project02_textming_dataset", data_files={'train': 'raw_newformat_data/traindata-00000-of-00001.arrow', 'test': 'raw_newformat_data/testdata-00000-of-00001.arrow'})
    
    sample = random.choice(new_data['test'])
    sampleQ = sample['question']
    sampleC = sample['context']
    sampleA = sample['answers']["text"][0]

    text = st.text_area(
        "Sample CONTEXT:",
        sampleC,
        height=200,
    )
    question = st.text_area(
        "Sample QUESTION: ",
        sampleQ,
        height=5,
    )
    
    answer = st.text_area(
        "True ANSWER:",
        sampleA,
        height=5,
    )

    # Create a prediction button
    if st.button("Sample & Predict"):
        prediction = ""
        prediction = predict(question, text)
        if prediction == "":
            st.error(prediction)
        else:
            st.success(prediction)

if __name__ == "__main__":
    main()