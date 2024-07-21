import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import plotly.graph_objs as go
import os
import re
import string
import calendar
from wordcloud import WordCloud
from collections import Counter
from streamlit_option_menu import option_menu
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

#NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Settings
st.set_page_config(page_title="Analisis Sentimen")

st.markdown("""
    <style>
        div.block-container {padding-top:2rem;}
        div.block-container {padding-bottom:2rem;}
    </style>
""", unsafe_allow_html=True)

# MongoDB Connection
load_dotenv(".env")
client = MongoClient("mongodb+srv://yukipm2ypm:WFTaqxEJvZXmdfIP@clusterhonda.hjc2t9k.mongodb.net/?retryWrites=true&w=majority&appName=Clusterhonda")
db = client["AnalisisSentimenDB"]
collection = db["User"]


## Load Data NEW
def load_saved_model():
    with open('model/label_lexicon/model.pkl', 'rb') as model_file:
        load_model = pickle.load(model_file)
    
    with open('model/label_lexicon/label_encoder.pkl', 'rb') as le_file:
        load_le = pickle.load(le_file)
    
    with open('model/label_lexicon/tfidf_vectorizer.pkl', 'rb') as tfidf_file:
        load_tfidf = pickle.load(tfidf_file)
    
    return load_model, load_le,load_tfidf


# Add Data
def add_data(username, access_control, name, password):
    # Check if the username already exists
    if collection.find_one({"username": username}):
        return {"status": "fail", "message": "Username already exists"}

    # If username doesn't exist, proceed to insert the new data
    data = {
        "username": username,
        "access_control": access_control,
        "name": name,
        "password": password
    }

    result = collection.insert_one(data)
    return {"status": "success", "inserted_id": str(result.inserted_id)}

# Delete Data
def delete_data(username):
    return collection.delete_one({"username": username})

# Update User
def update_user(username, access_control, name, password):
    collection.update_one(
        {"username": username},
        {"$set": {
            "access_control": access_control,
            "name": name,
            "password": password
        }}
    )

# Login
def validate_login(username, password):
    user = collection.find_one({"username": username, "password": password})
    return user

# Cache Data
@st.cache_data
def load_users_data():
    users_data = list(collection.find({}, {"_id": 0, "username": 1, "access_control": 1, "name": 1}))
    df_users = pd.DataFrame(users_data).astype(str)
    return df_users

# Clear Cache
def clear_cache():
    load_users_data.clear()

# Halaman Login
def login_page():
    st.markdown("<h1 style='text-align: center;'>Selamat Datang!</h1>", unsafe_allow_html=True)
    st.text("")

    col1, col2, col3 = st.columns([0.5, 1.5, 0.5])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username Anda")
            password = st.text_input("Password", placeholder="Masukkan password Anda", type="password")
            login_button = st.form_submit_button("Login", type="primary")

        if login_button:
            user = validate_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Username atau password salah. Silakan coba lagi.")

# Halaman About
def about_page():
    st.title('Halaman Perbandingan Analisis Sentimen Pada Komentar Channel Youtube WeloveHondaIndonesia Pada Kasus Rangka Esaf')
    # Membaca file Excel
    data = pd.read_excel('hasil_labeling_lexicon_manual.xlsx')

    st.write("**Data Labeling**:")
    st.dataframe(data, use_container_width=True)

    # Mendapatkan jumlah Label Positif, Netral, dan Negatif
    label_counts_lexicon = data['sentiment_label_lexicon'].value_counts()
    label_counts_manual = data['sentiment_label_manual'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        # Visualisasi Grafik Pie untuk Lexicon Sentiment
        st.subheader('Grafik Pie untuk Sentimen Label (Lexicon)')
        fig, ax = plt.subplots(figsize = (6, 6))
        sizes = [count for count in data['sentiment_label_lexicon'].value_counts()]
        labels = list(data['sentiment_label_lexicon'].value_counts().index)
        explode = (0.1 , 0)
        #jika pakek netral
        # explode = (0.1, 0, 0) 
        colors = ['#66b3ff', '#ffcc99', '#ff9999']
        ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
        ax.set_title(f"Sentiment Lexicon on Review Apps Data \n (total = {str(len(data['text']))}  review)", fontsize = 16, pad = 20)
        st.pyplot(fig, use_container_width=True)
        
    with col2:
        # Visualisasi Grafik Pie untuk Manual Sentiment
        st.subheader('Grafik Pie untuk Sentimen Label (Manual)')
        fig, ax = plt.subplots(figsize = (6, 6))
        sizes = [count for count in data['sentiment_label_manual'].value_counts()]
        labels = list(data['sentiment_label_manual'].value_counts().index)
        explode = (0.1 , 0)
        #jika pakek netral
        # explode = (0.1, 0, 0) 
        colors = ['#66b3ff', '#ffcc99', '#ff9999']
        ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
        ax.set_title(f"Sentiment Manual on Review Apps Data \n (total = {str(len(data['text']))}  review)", fontsize = 16, pad = 20)
        st.pyplot(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Visualisasi Grafik Batang untuk Lexicon Sentiment
        st.subheader('Grafik Batang untuk Sentimen Label (Lexicon)')
        fig_bar_lexicon = go.Figure(data=[go.Bar(x=label_counts_lexicon.index, y=label_counts_lexicon.values)])
        fig_bar_lexicon.update_xaxes(title='Sentiment Label (Lexicon)')
        fig_bar_lexicon.update_yaxes(title='Count')
        st.plotly_chart(fig_bar_lexicon, use_container_width=True)
    with col2:
        # Visualisasi Grafik Batang untuk Manual Sentiment
        st.subheader('Grafik Batang untuk Sentimen Label (Manual)')
        fig_bar_manual = go.Figure(data=[go.Bar(x=label_counts_manual.index, y=label_counts_manual.values)])
        fig_bar_manual.update_xaxes(title='Sentiment Label (Manual)')
        fig_bar_manual.update_yaxes(title='Count')
        st.plotly_chart(fig_bar_manual, use_container_width=True)

    # Filter teks berdasarkan sentimen
    # Step 1: Convert the 'text' column to strings
    data['text'] = data['text'].astype(str)
    # st.write("**Create Word Cloud**:")
    # st.dataframe(data, use_container_width=True)
    # Adding a selectbox for choosing the sentiment label type
    label = st.selectbox('Pilih jenis label sentimen:', ['sentiment_label_lexicon', 'sentiment_label_manual'])
    # Assuming you want to use the lexicon sentiment labels for the word clouds
    positive_text = ' '.join(data[data[label] == 'positive']['text'])
    negative_text = ' '.join(data[data[label] == 'negative']['text'])
    all_text = ' '.join(data['text'])

    # Visualisasi Wordcloud untuk sentimen positif
    st.subheader('Wordcloud untuk Sentimen Positif')
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    st.image(wordcloud_positive.to_array(), caption='Wordcloud Sentimen Positif', use_column_width=True)

    # Visualisasi Wordcloud untuk sentimen negatif
    st.subheader('Wordcloud untuk Sentimen Negatif')
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    st.image(wordcloud_negative.to_array(), caption='Wordcloud Sentimen Negatif', use_column_width=True)

    # Visualisasi Wordcloud untuk semua kata
    st.subheader('Wordcloud untuk Semua Kata')
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), caption='Wordcloud Semua Kata', use_column_width=True)

    # Function to create bar charts for word frequencies
    def create_word_freq_bar_chart(text, title, color):
        words = text.split()
        word_freq = Counter(words)
        top_words = word_freq.most_common(10)
        words, freq = zip(*top_words)
        fig = go.Figure([go.Bar(x=words, y=freq, marker=dict(color=color))])
        fig.update_layout(title=title, xaxis_title='Kata', yaxis_title='Frekuensi')
        return fig

    # Sentimen positif
    fig_positive = create_word_freq_bar_chart(positive_text, '10 Kata yang Paling Sering Muncul (Sentimen Positif)',
                                              'green')
    # Sentimen negatif
    fig_negative = create_word_freq_bar_chart(negative_text, '10 Kata yang Paling Sering Muncul (Sentimen Negatif)',
                                              'red')
    # Semua kata
    fig_all = create_word_freq_bar_chart(all_text, '10 Kata yang Paling Sering Muncul (Semua)', 'blue')

    # Menampilkan grafik menggunakan Plotly
    st.plotly_chart(fig_positive, use_container_width=True)
    st.plotly_chart(fig_negative, use_container_width=True)
    st.plotly_chart(fig_all, use_container_width=True)

    # Menambahkan kalimat tambahan
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            Berikut adalah hasil evaluasi label Lexicon:
    
            - **Akurasi Naive Bayes :** 80.60%
            - **Presisi             :** 90.24%
            - **Recall              :** 51.53%
            - **F1-score            :** 47.58%
        """)
    with col2:
        st.markdown("""
            Berikut adalah hasil evaluasi label manual:
    
            - **Akurasi Naive Bayes :** 72.50%
            - **Presisi             :** 68.18%
            - **Recall              :** 51.13%
            - **F1-score            :** 46.95%
        """)



#CLEANSING
def cleaning(Text, replacements=None):
    if pd.isnull(Text):  # Check if Text is NaN
        return ""
    # Mengganti huruf yang berulang-ulang ('oooooo' menjadi '00')
    Text = re.sub(r'(.)\1+', r'\1\1', Text)
    # Mengganti 2 atau lebih titik dengan spasi
    Text = re.sub(r'\.{2,}', ' ', Text)
    # Menghapus @username
    Text = re.sub('@[^\s]+','', Text)
    # Menghapus angka
    Text = re.sub('[0-9]+', '', Text)
    # Menghapus URL
    Text = re.sub(r"http\S+", "", Text)
    # Menghapus hashtag
    Text = re.sub(r'#', '', Text)
    # Menghapus spasi, tanda kutip ganda ("), dan tanda kutip tunggal (') dari teks
    Text = Text.strip(' "\'')
    # Mengganti beberapa spasi dengan satu spasi
    Text = re.sub(r'\s+', ' ', Text)
    # Menghapus tanda baca
    Text = Text.translate(str.maketrans("", "", string.punctuation))
    # Menghapus karakter tidak diinginkan menggunakan kamus pengganti khusus jika disediakan
    if replacements:
        for old, new in replacements.items():
            Text = Text.replace(old, new)
    # Mengembalikan teks yang telah dibersihkan
    return Text

#CASE FOLDING DAN CLEAN EMOJI
def casefolding(text):
  # Mengubah teks ke huruf kecil (lowercase)
  text = text.lower()
  # Menghapus emoticon
  # Pola regex untuk mendeteksi berbagai karakter emoticon dan simbol
  emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
  # Mengganti semua emoticon dan simbol yang terdeteksi dengan string kosong
  text = emoji_pattern.sub(r'', text)
  # Menghapus karakter non-ASCII
  encoded_string = text.encode("ascii", "ignore")
  # Mengubah kembali byte string menjadi string normal
  text = encoded_string.decode()
  # Mengembalikan teks yang sudah dibersihkan
  return text


#STEMMING
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#SLANGWORDS
kbba_dictionary = pd.read_csv('kbba.txt', delimiter='\t', names=['slang', 'formal'], header=None, encoding='utf-8')
slang_dict = dict(zip(kbba_dictionary['slang'], kbba_dictionary['formal']))
# kbba_dictionary
def convert_slangword(text):
    words = text.split()
    normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]
    normalized_text = ' '.join(normalized_words)
    return normalized_text

#STOPWORD
from nlp_id.stopword import StopWord
stopword = StopWord()
stopwords = stopword.get_stopword()
# Hapus kata "tidak" dari daftar stopwords
if "tidak" in stopwords:
    stopwords.remove("tidak")
# Fungsi untuk menghapus stopwords dengan pengecualian kata "tidak"
def remove_stopwords_with_exception(text):
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]
    return " ".join(cleaned_words).strip()

#UNWANTED WORD REMOVAL
unwanted_words = ['jan','feb','mar','apr','mei','jun','jul','aug','sep','oct','nov','dec','uaddown','weareuad','Iam','https','igshid']
import nltk
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt')
def RemoveUnwantedwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [word for word in word_tokens if not word in unwanted_words]
    return ' '.join(filtered_sentence)

#MENGHAPUS KATA DIBAWAH 3 HURUF


#TOKENIZING
def tokenize(teks):
    list_teks = []
    words = teks.split(" ")
    i = 0
    while i < len(words):
        if words[i] == "tidak" and i + 1 < len(words) and words[i + 1] == "percaya":
            list_teks.append("tidak percaya")
            i += 2
        else:
            list_teks.append(words[i])
            i += 1
    return list_teks

# Halaman Predict Text

def predict_text_page():
    st.subheader("Predict Sentiment from Text")
    tweet = st.text_input('Enter your tweet')
    submit = st.button('Predict')

    if submit:
        start = time.time()
        model, le, tfidf = load_saved_model()
        
        # Pembersihan teks menggunakan fungsi-fungsi di atas secara berurutan
        tweet_cleaning = cleaning(tweet)
        tweet_casefolding = casefolding(tweet_cleaning)
        tweet_stemming = stemmer.stem(tweet_casefolding)
        tweet_slangword = convert_slangword(tweet_stemming)
        tweet_stopwords = remove_stopwords_with_exception(tweet_slangword)
        tweet_unwanted = RemoveUnwantedwords(tweet_stopwords)
        tweet_final = ' '.join(re.findall(r'\w{3,}', tweet_unwanted))
        tweet_tokenize = tokenize(tweet_final)
        
        # Menampilkan hasil preprocessing di setiap langkah
        st.subheader("Cleansing")
        st.write("Menghapus tanda baca,url, spasi dll.")
        st.write(tweet_cleaning)
        
        st.subheader("Case Folding")
        st.write("Merubah huruf kapital menjadi huruf kecil dan membersihkan emoticon")
        st.write(tweet_casefolding)

        st.subheader("Stemming")
        st.write("kata yang memiliki imbuhan menjadi kata dasar")
        st.write(tweet_stemming)
        
        st.subheader("SlangWords")
        st.write("mengindentifikasi kata-kata slang (kata gaul) lalu akan diganti ke kata yang lebih baku atau umum")
        st.write(tweet_slangword)
        
        st.subheader("Stopwords")
        st.write("menghapus seluruh kata yang dianggap tidak penting")
        st.write(tweet_stopwords)
        
        st.subheader("Unwanted Words")
        st.write("menghapus Kata-kata yang tidak diinginkan yang akan dihapus dari teks")
        st.write(tweet_unwanted)

        st.subheader("Menghapus kata di bawah 3 huruf")
        # st.write("Removing unwanted words")
        st.write(tweet_final)
        
        st.subheader("Tokenizing")
        st.write("proses pemisahan kata, pemisahan kata pada setiap kalimat dilakukan berdasarkan delimeter yaitu adanya spasi pada setiap kata")
        st.write(tweet_tokenize)
        
        text_tfidf = tfidf.transform([tweet_unwanted])
        pred = model.predict(text_tfidf)
        sentiment = le.inverse_transform(pred)
        
        end = time.time()
        
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
        st.write('Predicted Sentiment:', sentiment[0])

# Halaman Predict DataFrame
def create_template():
    # Create a template DataFrame with example data
    data = {
        "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
        "other_column": ["Example 1", "Example 2", "Example 3"],
        "new_column": ["Value 1", "Value 2", "Value 3"]  # Add new column here
    }
    data = {
        "coba": ["Sample text 1", "Sample text 2", "Sample text 3"],
        "other_column": ["Example 1", "Example 2", "Example 3"],
        "new_column": ["Value 1", "Value 2", "Value 3"]  # Add new column here
    }

    template_df = pd.DataFrame(data)
    return template_df


def predict_dataframe_page():
    st.subheader("Predict Sentiment from DataFrame")
        
    # Create a download link for the template
    template_df = create_template()
    st.markdown("### Download Template")
    csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name='template.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Upload your DataFrame", type=['xlsx', 'csv'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file) if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' else pd.read_csv(uploaded_file)
        st.write('Uploaded DataFrame:')
        st.dataframe(df, use_container_width=True)
        # Create a list of column headers from the DataFrame
        column_df = df.columns.tolist()
        # Add a select box for choosing the 'access_control' column
        selectbox = st.selectbox("Select Features", column_df, placeholder="Pilih kolom yang akan di prediksi", index=None)
        submit_df = st.button(f'Predict Column {selectbox} Sentiments from DataFrame')









        if submit_df:
            
            start = time.time()
            
            #cleansing
            df["textClean_cleaning"] = df[selectbox].apply(cleaning)
            df.drop_duplicates(subset=["textClean_cleaning"], keep="first", inplace=True)
            st.subheader("Cleansing Text")
            st.write("removing hastag, reply and username")
            st.dataframe(df["textClean_cleaning"], use_container_width=True)

            #casefolding
            st.subheader("Case Folding Text")
            st.write("change text to lower case")
            df['textClean_casefolding'] = df['textClean_cleaning'].apply(casefolding)
            st.dataframe(df["textClean_casefolding"], use_container_width=True)

            #stemming
            st.subheader("Stemming Text")
            st.write("stemming the text, to become the base form")
            df['textClean_stemmer'] = df['textClean_casefolding'].apply(stemmer.stem)
            st.dataframe(df["textClean_stemmer"], use_container_width=True)

            #slangwords
            st.subheader("slang Text")
            st.write("removing slang")
            df['textClean_slang'] = df['textClean_stemmer'].apply(convert_slangword)
            st.dataframe(df["textClean_slang"], use_container_width=True)

            #stopword
            st.subheader("stopword Text")
            st.write("stopword removal")
            df["Text_Clean"] = df["textClean_slang"].apply(remove_stopwords_with_exception)
            df = df[df['Text_Clean'].str.strip().astype(bool)]
            df.dropna(axis=1, how='all', inplace=True)
            st.dataframe(df["Text_Clean"], use_container_width=True)

            #unwanted word removal
            st.subheader("unwanted word removal")
            st.write("unwanted word removal")
            df["Text_Clean_lagi"] = df["Text_Clean"].apply(RemoveUnwantedwords)
            st.dataframe(df["Text_Clean_lagi"], use_container_width=True)
            
            #menghapus kata dibawah 3 huruf
            st.subheader("Removing Words with Less Than 3 Characters")
            df["Text_Clean_lagi"] = df["Text_Clean_lagi"].str.findall(r'\b\w{3,}\b').str.join(' ')
            st.dataframe(df["Text_Clean_lagi"], use_container_width=True)

            # tokenize
            st.subheader("Tokenize Text")
            st.write("tokenize the text to split each word")
            df["Text_Clean_split"] = df["Text_Clean_lagi"].apply(tokenize)
            st.dataframe(df["Text_Clean_split"], use_container_width=True)

            selected_column = df["Text_Clean_lagi"]  # Remove NaN values
            # Menerapkan proses transformasi teks baru ke dalam vektor yang dapat dipahami oleh model
            model, le, tfidf = load_saved_model()
            
            
            column_vectorized = tfidf.transform(selected_column)
            predictions = model.predict(column_vectorized)
            sentiment = le.inverse_transform(predictions)
            
            end = time.time()
            st.write('Prediction time taken: ', round(end-start, 2), 'seconds')

            st.write('Predicted Sentiments:')

            predictions_df = pd.DataFrame(sentiment, index=selected_column.index, columns=['sentiment'])
            # Concatenate the original column with the predictions
            result_df = pd.concat([selected_column, predictions_df], axis=1)
            st.dataframe(result_df, use_container_width=True)
            
            # Replace sentiment values with their Indonesian equivalents
            result_df["sentiment"] = result_df["sentiment"].replace({"positive": "positif", "negative": "negatif"})
            
            # Daftar Kata-kata positif bahasa indonesia
            df_positive = pd.read_csv('https://raw.githubusercontent.com/YUKIPM24/kamus_lexicon/main/positive.txt', sep='\t',names=['positive'])
            list_positive = list(df_positive.iloc[::,0])

            # Daftar Kata-Kata negatif bahasa indonesia
            df_negative = pd.read_csv('https://raw.githubusercontent.com/YUKIPM24/kamus_lexicon/main/negative.txt', sep='\t',names=['negative'])
            list_negative = list(df_negative.iloc[::,0])


            # menghitung kata-kata positif/negatif pada dictionary lalu menentukan sentimennya :
            def sentiment_kamus_lexicon_id(text):
                score = 0
                positive_words = []
                negative_words = []
                neutral_words = []
                
                for word in text:
                    if word in list_positive:
                        score += 1
                        positive_words.append(word)
                    elif word in list_negative:
                        score -= 1
                        negative_words.append(word)
                    else:
                        neutral_words.append(word)
                
                polarity = ''
                if score > 0:
                    polarity = 'positive'
                elif score < 0:
                    polarity = 'negative'
                else:
                    polarity = 'neutral'
                
                result = {'positif': positive_words, 'negatif': negative_words, 'neutral': neutral_words}
                return score, polarity, result, positive_words, negative_words, neutral_words


            hasil = df['Text_Clean_split'].apply(sentiment_kamus_lexicon_id)
            hasil = list(zip(*hasil))
            df['polarity_score'] = hasil[0]
            df['polarity'] = hasil[1]
            hasil_kata_positive = hasil[3]
            hasil_kata_negative = hasil[4] 

            df = df[df.polarity != 'neutral']
            
            
            
            # LABELING LEXICON
            st.subheader("Labeling Lexicon Based")
            
            st.write("Proses pemberian label")
            st.write("Hasil:")
            labeling_col1, labeling_col2 = st.columns(2)
            
            with labeling_col1:
                st.info("Jumlah Data")
                st.markdown(f"`{len(result_df)}`")
                
            with labeling_col2:
                st.info("Detail")
                # Count the sentiment values
                sentiment_counts = df['polarity'].value_counts().rename_axis('polarity').reset_index(name='counts')

                # Set 'sentiment' sebagai index
                sentiment_counts.set_index('polarity', inplace=True)
                # Rename 'counts' kolom menjadi 'sentiment'
                sentiment_counts.columns = ['polarity']

                # Formating angka sentiment dengan koma
                sentiment_counts['polarity'] = sentiment_counts['polarity'].apply(lambda x: f"{x:,}")
                
                # display tabe
                st.table(sentiment_counts)
                
            # Menambahkan grafik pie
            sentiment_counts = pd.Series(df["polarity"]).value_counts() #################
            st.subheader('Sentiment Distribution')

            # Membuat data untuk grafik pie
            labels = sentiment_counts.index
            values = sentiment_counts.values

            # Menentukan palet warna
            palette_colors = ['#be185d', '#500724']

            # fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values,  marker=dict(colors=palette_colors))])
            # st.plotly_chart(fig_pie, use_container_width=True)
            
            #######
            fig, ax = plt.subplots(figsize = (6, 6))
            sizes = [count for count in df["polarity"].value_counts()]
            labels = list(df["polarity"].value_counts().index)
            explode = (0.1 , 0)
            #jika pakek netral
            # explode = (0.1, 0, 0) 
            colors = ['#66b3ff', '#ffcc99', '#ff9999']
            ax.pie(x = sizes, labels = labels, colors=colors, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
            ax.set_title(f"Sentiment on Review Apps Data \n (total = {str(len(result_df))}  review)", fontsize = 16, pad = 20)
            st.pyplot(fig, use_container_width=True)
            #######

            # Menambahkan grafik bar
            st.subheader('Sentiment Counts')

            fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=palette_colors)])
            fig_bar.update_layout(xaxis_title='Sentiment', yaxis_title='Count')
            st.plotly_chart(fig_bar, use_container_width=True)
                
                
            # TOP POSITIF DAN NEGATIF
            st.subheader("Top 10 kata Positif dan Negatif")
            st.write("Proses menampilkan Top 10 kata Positif dan Negatif")
            st.write("Hasil:")
            pos_col1, neg_col2 = st.columns(2)
            
            def top_words(hasil_kata_positive, hasil_kata_negative):
                # Menggabungkan semua kata positif dan negatif
                all_positive_words = [word for sublist in hasil_kata_positive for word in sublist]
                all_negative_words = [word for sublist in hasil_kata_negative for word in sublist]
                
                # Menghitung frekuensi kata positif dan negatif
                positive_freq = pd.Series(all_positive_words).value_counts().reset_index().rename(columns={'index': 'Positive Word', 0: 'Frequency'})
                negative_freq = pd.Series(all_negative_words).value_counts().reset_index().rename(columns={'index': 'Negative Word', 0: 'Frequency'})
                
                # Mengambil 11 kata teratas
                top_20_positive = positive_freq.head(11)
                top_20_negative = negative_freq.head(11)
                
                return top_20_positive, top_20_negative
            
            top_kata_positive, top_kata_negative = top_words(hasil_kata_positive, hasil_kata_negative)

            # Mengubah hasil menjadi DataFrame
            result3 = pd.DataFrame(top_kata_positive)
            result4 = pd.DataFrame(top_kata_negative)
            
            # Top positive and negative sentences
            with pos_col1:
                st.subheader("Top 10 Postive Word")
                st.dataframe(result3)
                
            with neg_col2:
                st.subheader("Top 10 Negative Word")
                st.dataframe(result4)
                
            # WORDCLOUD LEXICON
            positive_words_str = ' '.join(result3['Positive Word'])
            negative_words_str = ' '.join(result4['Negative Word'])
                
            # Visualisasi Wordcloud untuk sentimen positif
            st.subheader('Wordcloud untuk Lexicon Positif')
            wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_words_str)
            st.image(wordcloud_positive.to_array(), caption='Wordcloud Lexicon Positif', use_column_width=True)

            # Visualisasi Wordcloud untuk sentimen negatif
            st.subheader('Wordcloud untuk Lexicon Negatif')
            wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_words_str)
            st.image(wordcloud_negative.to_array(), caption='Wordcloud Lexicon Negatif', use_column_width=True)
                
            
            # Analisis TFIDF
            st.subheader("Analisis TF-IDF")
            st.write("Proses pembobotan kata")
            st.write("Hasil:")

            analisis_tfidf = TfidfVectorizer()
            result_tfidf = analisis_tfidf.fit_transform(df["Text_Clean_lagi"])
            # Mengakses nilai-nilai non-zero dari sparse matrix
            coo_matrix = result_tfidf.tocoo()

            # Menampilkan hanya 5 hasil pertama dalam format yang diinginkan
            max_results = 5
            results_displayed = 0

            for row, col, data in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                if results_displayed < max_results:
                    st.text(f"({row}, {col})\t{data}")
                    results_displayed += 1
                else:
                    break
                
                
            # Data Train and Data Test
            st.subheader("Data Train and Data Test")
            st.write("Proses pemisahan antara data train dan test")
            st.write("Hasil:")
            
            X_train, X_test, y_train, y_test = train_test_split(result_df["Text_Clean_lagi"], result_df["sentiment"], test_size=0.1, random_state=0, stratify=result_df["sentiment"])
            data_split = {
                "Nama Data": ["Total Data", "Total Data Latih", "Total Data Test"],
                "Hasil Data": [len(predictions_df), len(X_train), len(X_test)]
            }
            st.dataframe(data_split)
            
            
            # Menghitung Akurasi Naive Bayes
            st.subheader("Menghitung Akurasi Naive Bayes")
            st.write("Proses menghitung menggunakan metode Naive Bayes")
            st.write("Hasil:")
            
            from sklearn import naive_bayes
            
            df['sentiment'] = df['polarity'].astype(str)

            le = LabelEncoder()
            df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
            
            df["Text_Clean_lagi"] = df['Text_Clean_lagi'].astype(str)
            tfidf = TfidfVectorizer()
            X = tfidf.fit_transform(df['Text_Clean_lagi'].values.tolist())
            y = df['sentiment_encoded']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
            
            model_train = naive_bayes.MultinomialNB()
            model_train.fit(X_train, y_train)

            # test predict
            y_pred = model_train.predict(X_test)

            # check accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            st.progress(int(accuracy*100), text=f"{accuracy * 100:.2f}%")
            
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            st.write("Proses Menghitung akurasi,precison,recall dan f score dengan confusion matrix yang didapat menggunakan metode naive bayes clasifier")
            st.write("Hasil:")
            
            conf_matrix_nb = confusion_matrix(y_test, y_pred)
            
            fig = plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix Multinomial Naive Bayes')
            st.pyplot(fig)
            
            data_confusion = {
                "Tipe Data": ["True Negative", "True Positive", "False Negative", "False Positive"],
                "Hasil": [conf_matrix_nb[0,0], conf_matrix_nb[1,1], conf_matrix_nb[1,0], conf_matrix_nb[0,1]]
            }
            
            st.dataframe(data_confusion, use_container_width=True)       
            # Calculate accuracy
            tn, fp, fn, tp = conf_matrix_nb.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            with st.expander("Rumus"):
                st.markdown(r'''
                    $$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$
                ''')
                st.markdown(rf'''
                    $$\text{{Accuracy}} = \frac{{{tp}+{tn}}}{{{tp}+{tn}+{fp}+{fn}}} = {accuracy:.2%}$$
                ''')

                
            # Assuming 'Positif' is the positive class label
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label='Positif', average='macro')
            recall = recall_score(y_test, y_pred, pos_label='Positif', average='macro')
            f1 = f1_score(y_test, y_pred, pos_label='Positif', average='macro')
            
            # Calculate accuracy, precision, recall, and f1-score
            accuracy_percent = accuracy * 100
            precision_percent = precision * 100
            recall_percent = recall * 100
            f1_percent = f1 * 100
            
            data_akurasi = {
                "Nama": ["Akurasi", "Precision", "Recall", "F1-Score"],
                "Hasil": [f"{accuracy_percent:.2f}%", f"{precision_percent:.2f}%", f"{recall_percent:.2f}%", f"{f1_percent:.2f}%"]
            }
            
            st.dataframe(data_akurasi, use_container_width=True)
            st.download_button(label='Download CSV',data =df.to_csv(), file_name="model_df.csv" ,mime='text/csv')
            
            
            
                        
# Halaman Access Management Admin
def access_management_page_admin():
    with st.sidebar:
        access_management_page_option = st.selectbox("Menu Access Management", ["Add User", "Delete User", "Edit User"])

    users_data = load_users_data()

    with st.container(border=True):
        st.subheader("Users Data")
        users_df = st.dataframe(users_data, use_container_width=True)

    st.divider()

    if access_management_page_option == "Add User":
        with st.form("add_user_form", clear_on_submit=True):
            st.subheader("Form Add User")

            add_username = st.text_input("Username", placeholder="Masukkan username")
            add_access_control = st.selectbox("Access Control", ["Admin", "User"], placeholder="Pilih access control", index=None)
            add_name = st.text_input("Nama", placeholder="Masukkan nama")
            add_password = st.text_input("Password", placeholder="Masukkan password", type="password")
            add_user_button = st.form_submit_button("Add User", type="primary")

            if add_user_button:
                if add_username and add_access_control and add_name and add_password:
                    result = add_data(add_username, add_access_control, add_name, add_password)
                    if result['status'] == 'success':
                        message_success = st.success(f"User {add_name} telah berhasil ditambahkan")
                        time.sleep(3)
                        message_success.empty()

                        clear_cache()
                        users_data = load_users_data()
                        users_df.data = users_data
                        st.experimental_rerun()
                    else:
                        message_error = st.error(result['message'])
                        time.sleep(3)
                        message_error.empty()
                else:
                    message_error = st.error("Harap isi semua field")
                    time.sleep(3)
                    message_error.empty()

    elif access_management_page_option == "Delete User":
        with st.form("delete_user_form", clear_on_submit=True):
            st.subheader("Form Delete User")

            delete_username = st.selectbox("Username", np.sort(pd.DataFrame(users_data)["username"].unique()), placeholder="Pilih username", index=None)
            delete_user_button = st.form_submit_button("Delete User", type="primary")

            if delete_user_button:
                if delete_username:
                    delete_data(delete_username)
                    message_success = st.success(f"User {delete_username} telah berhasil dihapus")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap pilih username")
                    time.sleep(3)
                    message_error.empty()

    else:
        with st.form("edit_user_form", clear_on_submit=True):
            st.subheader("Form Edit User")

            edit_username = st.selectbox("Username", np.sort(pd.DataFrame(users_data)["username"].unique()), placeholder="Pilih username", index=None)
            edit_access_control = st.selectbox("Access Control", ["Admin", "User"], placeholder="Pilih access control", index=None)
            edit_name = st.text_input("Nama", placeholder="Masukkan nama")
            edit_password = st.text_input("Password", placeholder="Masukkan password", type="password")
            edit_user_button = st.form_submit_button("Edit User", type="primary")

            if edit_user_button:
                if edit_username and edit_access_control and edit_name and edit_password:
                    update_user(edit_username, edit_access_control, edit_name, edit_password)
                    message_success = st.success(f"User {edit_username} telah berhasil diperbarui")
                    time.sleep(3)
                    message_success.empty()

                    clear_cache()
                    users_data = load_users_data()
                    users_df.data = users_data
                    st.rerun()
                else:
                    message_error = st.error("Harap isi semua field")
                    time.sleep(3)
                    message_error.empty()

# Halaman Access Management User
def access_management_page_user():
    current_user = st.session_state.get("user", {})
    edit_username = current_user.get("username", "")
    edit_access_control = current_user.get("access_control", "User")

    with st.form("edit_user_form_user", clear_on_submit=True):
        st.subheader("Form Edit User")

        edit_name = st.text_input("Username", placeholder="Username", value=edit_username)
        edit_password = st.text_input("Password", placeholder="Masukkan password", type="password")
        edit_user_button = st.form_submit_button("Edit User", type="primary")

        if edit_user_button:
            if edit_name and edit_password:
                update_user(edit_username, edit_access_control, edit_name, edit_password)
                message_success = st.success(f"User {edit_username} telah berhasil diperbarui")
                time.sleep(3)
                message_success.empty()

                clear_cache()
                st.session_state.user["name"] = edit_name
                st.rerun()
            else:
                message_error = st.error("Harap isi semua field")
                time.sleep(3)
                message_error.empty()

# Halaman Report
def report_page():
    st.subheader("Report Dataset Komentar Youtube Honda")

    report_data = pd.read_excel("Klarifikasi_Kemunculan_Warna_Kuning_Pada_Rangka_Honda.xlsx")
    report_data["pubdate"] = pd.to_datetime(report_data["pubdate"])

    col1, col2 = st.columns(2)
    with col1:
        year_filter = st.selectbox(
            "Tahun",
            np.sort(report_data["pubdate"].dt.year.unique()),
            placeholder="Pilih Tahun",
            index=None
        )

    with col2:
        if year_filter:
            year_filtered_report_data = report_data[
                report_data["pubdate"].dt.year == year_filter
            ].reset_index(drop=True)

            unique_months = np.sort(year_filtered_report_data["pubdate"].dt.month.unique())
            month_names = [calendar.month_name[month] for month in unique_months]

            month_filter = st.selectbox(
                "Bulan",
                month_names,
                placeholder="Pilih Bulan",
                index=None
            )

            if month_filter:
                month_number = {name: num for num, name in enumerate(calendar.month_name) if num in unique_months}[month_filter]

                year_month_filtered_report_data = year_filtered_report_data[
                    year_filtered_report_data["pubdate"].dt.month == month_number
                ].reset_index(drop=True)
        else:
            month_filter = st.selectbox(
                "Bulan",
                ["Pilih tahun terlebih dahulu"],
                placeholder="Pilih Bulan",
                index=None
            )

    if year_filter and month_filter:
        st.dataframe(year_month_filtered_report_data, use_container_width=True)
    elif year_filter:
        st.dataframe(year_filtered_report_data, use_container_width=True)
    else:
        st.dataframe(report_data, use_container_width=True)

# Session State untuk Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Tampilkan Halaman Login Jika Belum Login
if not st.session_state.logged_in:
    login_page()
else:
    # Tampilkan Halaman Isi
    with st.sidebar:
        user = st.session_state.get("user", {})
        user_name = user.get("name", "User")
        user_access = user.get("access_control", "user")
        st.markdown(f"Selamat datang, {user_access} {user_name}")

        menu_title = "Menu Admin" if user_access.lower() == "admin" else "Menu User"
        menu_options = ["Predict Text", "Predict DataFrame", "Access Management"] ####################################

        if user_access.lower() == "admin":
            menu_options.insert(0, "About")
            menu_options.append("Report")

        option = option_menu(
            menu_title,
            menu_options,
            menu_icon="cast",
            default_index=0
        )

        logout_button = st.button("Logout", type="primary")

    # Logout
    if logout_button:
        st.session_state.logged_in = False
        st.rerun()

    # Menentukan Halaman yang Sesuai
    if option == "About":
        about_page()
    elif option == "Predict Text":
        predict_text_page()
    elif option == "Predict DataFrame":
        predict_dataframe_page()
    elif option == "Access Management":
        if user_access.lower() == "admin":
            access_management_page_admin()
        else:
            access_management_page_user()
    else:
        report_page()
