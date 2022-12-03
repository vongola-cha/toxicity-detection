import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm

# #################### parameters ####################
TRAIN_FILE = "./data/train.csv"
USE_SYMBOLS = False  # if you want to remove all symbols and retain only alphabets, set it False
SHOW_CLEAN_DATA = True  # Show training data after cleaning
REM_STOP=True  # remove stopwords, set it True
NEW_FIlE="./data/new_train.csv"
BALANCED_FIlE="./data/balanced_train.csv"

# #################### clean data ####################
def cleantxt(txt):
    """
    Cleans the string passed. Cleaning Includes-
    1. convert text to lower-case
    2. keep or remove symbols   default: USE_SYMBOLS=True
    3. deal with repeating characters
    4. remove stop-words
    """

    # collecting english stop words from nltk-library
    stpw = stopwords.words('english')

    # Adding custom stop-words
    stpw.extend(['www', 'http', 'utc'])
    stpw = set(stpw)

    # -----using regex to clean the text------
    # 1. convert text to lower-case
    txt = txt.lower()

    # 2. deal with symbols
    if USE_SYMBOLS:
        txt = re.sub(r"[\“\’]", "", txt)
        # Use Symbols
        txt = re.sub(r"\n", " \n ", txt)
        # Replace repeating characters more than 3 times to length of 3
        txt = re.sub(r'([*!?\'])\1\1{2,}', r'\1\1\1', txt)  # symbols
        # Add space around repeating characters
        txt = re.sub(r'([*!?\']+)', r' \1 ', txt)
    else:
        # Not Use Symbols
        # remove special characters/punctuations
        txt = re.sub(r"\n", " ", txt)
        txt = re.sub("[\<\[].*?[\>\]]", "", txt)
        # retain only alphabets
        txt = re.sub(r"[^a-z ]", " ", txt)  # characters not in a-z

    # 3. patterns with consecutive repeating characters
    txt = re.sub(r'([a-zA-Z])\1{2,}\b', r'\1\1', txt)
    txt = re.sub(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1', txt)
    txt = re.sub(r'[ ]{2,}', ' ', txt).strip()

    # 4. remove stop-words
    # ATTENTION: remove stopwords may cause nan!  e.g. "Me too!"
    if REM_STOP:
        txt = " ".join([x for x in txt.split() if x not in stpw])

    return txt

# # ################## Test clean function ########################
# # a dataframe example to display function cleantxt(txt)
# test_clean_df = pd.DataFrame({"text":
#                                   ["The cat",
#                                     "Me too!",
#                                    "heyy\n\nkkdsfj",
#                                    "hi   how/are/you ???",
#                                    "hey?????",
#                                    "noooo!!!!!!!!!   comeone !! ",
#                                    "cooooooooool     brooooooooooo  coool brooo",
#                                    "naaaahhhhhhh"]})
# test_clean_df['text'] = test_clean_df.text.apply(lambda x: cleantxt(x))
# print(test_clean_df)
# test_clean_df=test_clean_df.replace("",np.nan).dropna(subset=['text'])
# print(test_clean_df)
# exit()


    # ################## load data ####################
def do_clean():
    """
    Cleans data and saves
    """
    # Load the train dataset
    df = pd.read_csv(TRAIN_FILE)

    # Clean the text
    print("start data cleaning...")
    tqdm.pandas(desc='pandas bar')
    # clean
    df['comment_text'] = df.comment_text.progress_apply(lambda x: cleantxt(x))
    # remove space and nan
    df = df.replace("", np.nan).dropna(subset=['comment_text'])

    # get text and target--> save
    X = df.iloc[:, 2]  # text
    y = df.iloc[:, 1]  # scores
    newdata=pd.concat([X,y],axis=1)
    # save
    newdata.to_csv(NEW_FIlE, index=False)
    # data balance
    toxic=df.loc[df['target']>0]
    nottoxic = df.loc[df['target'] == 0]
    subset = nottoxic.sample(n=toxic.shape[0])
    newdata=pd.concat([toxic,subset],sort=False)
    newdata.to_csv(BALANCED_FIlE, index=False)
    return


def get_clean_data():
    """
    Gets data after cleaning and returns train, val, and test splits
    """
    # get newdata
    df = pd.read_csv(BALANCED_FIlE)

    X = df['comment_text']  # text
    y = df['target']  # scores
    # split for cross-validation (train-60%, validation 20% and test 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    return X_train, X_val, X_test, y_train, y_val, y_test

def analyze_data():
    """
    Display Data Distribution
    """
    # get newdata
    df = pd.read_csv(BALANCED_FIlE)
    print("\n")
    print("------- Display Data Distribution --------")
    # df = df.dropna(subset=['comment_text'])
    X = df['comment_text']  # text
    y = df['target']  # scores
    length=y.shape[0]
    zero_cnt=0
    not_zero_cnt=0
    # Data Distribution
    for score in y:
        if score==0:
            zero_cnt+=1
        else:
            not_zero_cnt+=1
    print("Not Toxic Comments:",zero_cnt, "  %.2f%%" %(zero_cnt/length*100))
    print("Toxic Comments:", not_zero_cnt, "  %.2f%%" %(not_zero_cnt/length*100))

    # Data Length
    len_sum=0
    for comment in X:
        len_sum+=int(len(comment))
    print("Average length of comments: %.2f" %(len_sum/length))
    return




print("use symbols:", USE_SYMBOLS,"   remove stopwords:", REM_STOP, "   Show training data example:", SHOW_CLEAN_DATA,"\n")

# Clean the text
# do_clean()

# Data Distribution
analyze_data()

# How to get data after cleaning
X_train, X_val, X_test, y_train, y_val, y_test = get_clean_data()
# Display
print("\n")
print("------- Display Data Splitting--------")
print("Training data:",X_train.shape)
print("Training scores:",y_train.shape)
print("Validation data:",X_val.shape)
print("Validation scores:",y_val.shape)
print("Test data:",X_test.shape)
print("Test scores:",y_test.shape)
if SHOW_CLEAN_DATA:
    print("\n")
    print("------- Show training data --------")
    print(X_train)
    print(y_train)




