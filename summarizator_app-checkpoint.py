import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import streamlit as st
from summarizer import Summarizer

available_models = {
    "IlyaGusev/rugpt3medium_sum_gazeta": "Russian Summarization (IlyaGusev/rugpt3medium_sum_gazeta)",
    "Shahm/t5-small-german": "German Summarization (Shahm/t5-small-german)",
    "Falconsai/medical_summarization": "English Summarization (Falconsai/medical_summarization)",
    "sacreemure/med_t5_summ_ru":"Russian Medical Texts Summarization (sacreemure/med_t5_summ_ru)"
}


def hugging_face_summarize(article, model_name):
    if "rugpt3medium" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        input_ids = tokenizer(article, return_tensors='pt', max_length=400, truncation=True, padding=True)["input_ids"]
        output_ids = model.generate(input_ids, max_new_tokens=300, repetition_penalty = 7.0, num_return_sequences=5, temperature = 0.7, top_k=50, early_stopping=True)[0]
        summary = tokenizer.decode(output_ids, skip_special_tokens=True)
        
    elif "medical" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        input_ids= tokenizer(article, return_tensors='pt', max_length=504, truncation=True, padding=True)["input_ids"]
        output_ids = model.generate(input_ids, max_new_tokens=500)
        summary = tokenizer.decode(output_ids, skip_special_tokens=True)

    elif "med_t5" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        input_ids = tokenizer(article, return_tensors='pt', max_length=2048, truncation=True)["input_ids"]
        output_ids = model.generate(input_ids, min_length=800, max_length=1000, repetition_penalty = 2.0, num_return_sequences=1, temperature = 0.7, top_k=50, early_stopping=True)[0]
        summary = tokenizer.decode(output_ids, skip_special_tokens=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_fast=False)
        inputs = tokenizer(article, return_tensors="pt", max_length=800, truncation=True, padding=True)
        output_ids = model.generate(inputs.input_ids, max_new_tokens=100, num_return_sequences=1)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        

    return summary

def main():
            
    st.title("Суммаризиризатор медицинских текстов")
    st.write("Вы можете выбрать модель суммаризации для русского, английского или немецкого")

    selected_model = st.selectbox("Выберете модель:", list(available_models.values()))

    article_text = st.text_area("Введите текст:")

    if st.button("Суммаризировать"):
        if article_text:
            model_name = [name for name, model in available_models.items() if model == selected_model][0]
            summary = hugging_face_summarize(article_text, model_name)
                
            st.subheader("Сокращенный текст:")
            st.write(summary)
        else:
            st.warning("Пожалуйста, введите текст.")

if __name__ == "__main__":
    main()
