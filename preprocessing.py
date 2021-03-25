from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import regex as re

def read_stop_words():
    stop_words = []
    with open("EnglishStopWords.txt", 'r',  encoding='utf-8', errors='ignore') as f:
        read_file = f.readlines()
        for word in read_file:
            stop_words.append(word.strip())
    
    return {key: None for key in stop_words}

#initialise some variables
stop_words = read_stop_words()
stemmer = EnglishStemmer()

def preprocess_string(text, stopping = True, stemming = True, lowercasing = True):
    global stop_words
    global stemmer
    if not text:
        text = ""
    # print(f"Input to preprocesser: {text}")
    text = text.encode("utf-8",errors="ingore").decode("utf-8", errors="ingore")
    # text = unicode(text, errors='ignore')
    html_tag_regex = re.compile('<.*?>')
    non_word_regex = re.compile('[^\w\']')
    non_alpha_numeric_chars = re.compile('[^a-z-A-Z-0-9\' ]')

    # HTML stripped
    html_stripped_string = re.sub(html_tag_regex, ' ', text)
    # Tokenize and remove odd/nonalphanumeric characters. (except for unknown characters by the regex)
    newline_removed_string = " ".join(re.split(non_word_regex,html_stripped_string))
    #clean text from leftover unkown characters
    stripped_string = re.sub(non_alpha_numeric_chars, "", newline_removed_string)
    temp_string = stripped_string

    if stopping:
        # stopword removal
        stripped_no_stopword_list = [word for word in stripped_string.split() if not word in stop_words]
        #remove extra empty strings left after last removal
        stripped_string = " ".join(stripped_no_stopword_list)
        temp_string = stripped_string

    if lowercasing:
        # Lowercasing.
        temp_string = temp_string.lower()

    if stemming:
        # Stemming
        temp_string = " ".join([stemmer.stem(word) for word in temp_string.split()])

    return temp_string

def preprocess_QA_text(text):
    if not text:
        text = ""
    text = text.encode("utf-8",errors="ingore").decode("utf-8", errors="ingore")
    # html removal
    clean_text = re.sub('<.*?>', ' ', text)
    # remove other special characters except thos helping the meaning of a sentence
    clean_text = re.sub('[^a-zA-Z0-9,\'.?!:\-()\[\] ]', '', clean_text)
    return clean_text

