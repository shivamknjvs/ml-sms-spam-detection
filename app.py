import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pandas import pd

# ps = PorterStemmer()

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
def app():
    st.title("SMS Spam Detector")
    st.write("Upload a .csv file with SMS messages to classify them as spam or not spam")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a file")

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Load the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Preprocess the SMS data
        # Remove stop words, stem, and convert to lowercase

        # Extract features from the SMS data
        X = tk.transform(df['text'])

        # Use the trained model to make predictions
        y_pred = model.predict(X)

        # Add the predicted labels to the DataFrame
        df['label'] = y_pred

        # Download the labeled CSV file
        st.download_button(label="Download labeled CSV", data=df.to_csv(), file_name="labeled_sms.csv", mime="text/csv")

app()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)


# tk = pickle.load(open("vectorizer.pkl", 'rb'))
# model = pickle.load(open("model.pkl", 'rb'))

# st.title("SMS Spam Detection Model")
# #
    

# # input_sms = st.text_input("Enter the SMS Here")
# file = st.file_uploader("Upload file")
# show_file = st.empty()
# if not file:
#     show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
    
# content = file.getvalue()
# data = pd.read_csv(file,encoding='ISO-8859-1')
# st.dataframe(data.head())  
# message = data['v2'] 
# preds = []
# for i in message:
#            i = [i]
#            i = tk.transform(i)
#            ans  = model.predict(i)
#            preds.append(ans)

# st.dataframe(preds)

# data['ans'] = preds
# csv = data.to_csv(index=False).encode('utf-8')
# st.download_button(
#           "Press to Download",
#           csv,
#           "file.csv",
#           "text/csv",
#           key='download-csv'
#         )
        
# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tk.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
