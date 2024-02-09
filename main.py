import torch
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import spacy
import pandas as pd
import base64
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

# Streamlit app for Text Summarization and NER
def text_summarization_ner_app():
    st.title('Text Summarization and Named Entity Recognition Demonstration')
    st.markdown('Using BART and T5 transformer models for summarization, and spaCy for Named Entity Recognition')

    model = st.selectbox('Select the model for summarization', ('BART', 'T5'))

    if model == 'BART':
        _num_beams, _no_repeat_ngram_size, _length_penalty = 4, 3, 1
        _min_length, _max_length, _early_stopping = 12, 128, True
    else:
        _num_beams, _no_repeat_ngram_size, _length_penalty = 4, 3, 2
        _min_length, _max_length, _early_stopping = 30, 200, True

    col1, col2, col3 = st.columns(3)
    _num_beams = col1.number_input("num_beams", value=_num_beams)
    _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
    _length_penalty = col3.number_input("length_penalty", value=_length_penalty)

    col1, col2, col3 = st.columns(3)
    _min_length = col1.number_input("min_length", value=_min_length)
    _max_length = col2.number_input("max_length", value=_max_length)
    _early_stopping = col3.number_input("early_stopping", value=_early_stopping)

    text = st.text_area('Text Input')

    def run_summarization_model(input_text):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model == "BART":
            bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            input_text = str(input_text).replace('\n', '')
            input_text = ' '.join(input_text.split())
            input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
            summary_ids = bart_model.generate(input_tokenized,
                                            num_beams=_num_beams,
                                            no_repeat_ngram_size=_no_repeat_ngram_size,
                                            length_penalty=_length_penalty,
                                            min_length=_min_length,
                                            max_length=_max_length,
                                            early_stopping=_early_stopping)

            output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                    summary_ids]
            st.subheader('Summary:')
            st.success(output[0])

        else:
            t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
            t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
            input_text = str(input_text).replace('\n', '')
            input_text = ' '.join(input_text.split())
            input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
            summary_task = torch.tensor([[21603, 10]]).to(device)
            input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
            summary_ids = t5_model.generate(input_tokenized,
                                            num_beams=_num_beams,
                                            no_repeat_ngram_size=_no_repeat_ngram_size,
                                            length_penalty=_length_penalty,
                                            min_length=_min_length,
                                            max_length=_max_length,
                                            early_stopping=_early_stopping)
            output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]
            st.subheader('Summary:')
            st.success(output[0])

    def run_ner(input_text):
        doc = nlp(input_text)

        entities = [(ent.text, ent.label_) for ent in doc.ents]

        st.subheader("Named Entities:")
        st.table(pd.DataFrame(entities, columns=["Entity", "Entity Type"]))

        if entities:
            csv_data = pd.DataFrame(entities, columns=["Entity", "Entity Type"]).to_csv(index=False)
            download_button = st.button("Download Entities CSV", key="download_button", help="Click to download")
            if download_button:
                download_link = get_csv_download_link(csv_data)
                st.markdown(download_link, unsafe_allow_html=True)

    def get_csv_download_link(csv_data):
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<div style="display: flex; justify-content: center;"><a href="data:file/csv;base64,{b64}" download="named_entities.csv"><button style="padding: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">Download Entities CSV</button></a></div>'
        return href

    if st.button('Submit'):
        run_summarization_model(text)
        run_ner(text)

# Streamlit app for Sentiment Analysis
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'
def sentiment_analysis_app():
    st.header('Sentiment Analysis')

    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')
        if text:
            label = analyze_sentiment(text)
            st.write('Sentiment: ', label)
            st.write('Score: ', round(TextBlob(text).sentiment.polarity, 2))

    with st.expander('Analyze CSV'):
        uploaded_file = st.file_uploader('Upload file')
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            del df['Liked']
            df['score'] = df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
            df['analysis'] = df['score'].apply(lambda x: 'Positive' if x >= 0.5 else 'Negative' if x <= -0.5 else 'Neutral')
            st.write(df.head(10))

            @st.cache
            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')

            csv_data = convert_df_to_csv(df)

            st.download_button(
                label="Download data as CSV",
                data=csv_data,
                file_name='sentiment.csv',
                mime='text/csv',
            )

# Main App
def main():
    st.sidebar.title("Navigation")
    app_selection = st.sidebar.selectbox("Select App", ["Text Summarization and NER", "Sentiment Analysis"])

    if app_selection == "Text Summarization and NER":
        text_summarization_ner_app()
    elif app_selection == "Sentiment Analysis":
        sentiment_analysis_app()

if __name__ == '__main__':
    main()

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)