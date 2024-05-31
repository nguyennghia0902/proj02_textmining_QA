# Question Answering - Streamlit Demo
Project 02 - Text Mining and Application - FIT@HCMUS - 2024

## Description
A simple QA app demo using ELECTRA and Streamlit in Vietnamese:
- Dataset: [Kaggle-CSC15105](https://www.kaggle.com/datasets/duyminhnguyentran/csc15105), [HuggingFace-project02_textming_dataset](https://huggingface.co/datasets/nguyennghia0902/project02_textming_dataset)
- Model: [nguyennghia0902/electra-small-discriminator_0.0001_16_15e](https://huggingface.co/nguyennghia0902/electra-small-discriminator_0.0001_16_15e)
- Streamlit app: [App](https://proj02textminingapp-nghiaitus.streamlit.app/)

## Team members:
| ID | Fullname | Email |
|---|---|---|
| 1712603| Lê Quang Nam | 1712603@student.hcmus.edu.vn|
| 19120582| Lê Nhựt Minh | 19120582@student.hcmus.edu.vn|
| 19120600| Bùi Nguyên Nghĩa | 19120600@student.hcmus.edu.vn |
| 21120198 | Nguyễn Thị Lan Anh  | 21120198@student.hcmus.edu.vn |
    

## How to use
### Using conda environment
1. Create new conda environment and install required dependencies:
```
$ conda create -n <env_name> -y python=3.11
$ conda activate <env_name>
$ pip3 install -r requirements.txt
```
2. Host streamlit app
```
$ streamlit run app.py
```
