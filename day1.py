import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid= SentimentIntensityAnalyzer()
text="I am so happy today! the weather is beautiful,and everything is going well.i fell very positive and motivated"
def detect_emotion(text):
    scores=sid.polarity_scores(text)
    print("Sentiment Scores:",scores)

    if scores["compound"]>=0.5:
        emotion="Joy"
    elif scores["compound"]<=-0.5:
        emotion="Sadness"
    elif scores["neg"]>0.5:
        emotion="Anger"
    elif scores['neu']>0.7:
        emotion="Neutral"
    else:
        emotion="mixed emotions"
    return emotion

emotion=detect_emotion(text)









print("detected emotion:",emotion)