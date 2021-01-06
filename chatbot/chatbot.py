import nltk
import json
import random
import train_bot
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')

intents = json.loads(open('response_data.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
stop_words = set(stopwords.words('english'))


class ChatBot:

    def __init__(self, text):
        self.user_text = text

    def get_response(self):
        ints = self.predict_class(self.user_text)
        res = self.get_processed_output(ints)
        return res

    def predict_class(self, sentence):
        # filter out predictions below a threshold
        p = self.bow(sentence)
        res = model.predict(np.array([p]))[0]
        error_threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_processed_output(self, ints):
        tag = ints[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])

    def bow(self, sentence):
        # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
        return np.array(bag)

    def clean_up_sentence(self,sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [
            lemmatizer.lemmatize(word.lower()) for word in sentence_words
        ]
        return sentence_words


if __name__ == '__main__':
    flag = True
    while flag:
        user_text = input('you: ').lower()
        if 'thanks' in user_text or 'thank you' in user_text or 'bye' in user_text:
            print(f'Chatbot: {ChatBot(user_text).get_response()}')
            flag = False
        else:
            print(f'Chatbot: {ChatBot(user_text).get_response()}')












