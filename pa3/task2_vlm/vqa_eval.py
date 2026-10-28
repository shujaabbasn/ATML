from __future__ import annotations

import re


#answer normalization ported from the official vqa evaluation code (GT-Vision-Lab/VQA, vqaEval.py)
#manual: apply the same answer normalization throughout, so we use it for targets, references and predictions

CONTRACTIONS={"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've"}
NUMBER_WORDS={"none":"0","zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
ARTICLES=["a","an","the"]
PERIOD_STRIP=re.compile(r"(?!<=\d)(\.)(?!\d)") #a period not inside a number
COMMA_STRIP=re.compile(r"(\d)(\,)(\d)") #comma inside a number like 1,000
PUNCTUATION=[";",r"/","[","]",'"',"{","}","(",")","=","+","\\","_","-",">","<","@","`",",","?","!"]


def process_punctuation(text):
    out=text
    for p in PUNCTUATION:
        if (p+" " in text or " "+p in text) or (re.search(COMMA_STRIP,text) is not None):
            out=out.replace(p,"") #official rule: delete when next to a space or when the text has a number comma
        else:
            out=out.replace(p," ")
    out=PERIOD_STRIP.sub("",out)
    return out


def process_digit_article(text):
    words=[]
    for word in text.lower().split():
        if word in NUMBER_WORDS:
            word=NUMBER_WORDS[word] #"two" -> "2"
        if word not in ARTICLES:
            words.append(word)
    for i in range(len(words)):
        if words[i] in CONTRACTIONS:
            words[i]=CONTRACTIONS[words[i]]
    return " ".join(words)


def normalize_answer(text):
    if text is None:
        return ""
    text=text.replace("\n"," ").replace("\t"," ").strip()
    text=process_punctuation(text)
    text=process_digit_article(text)
    return text


def vqa_soft_accuracy(prediction,references):
    """Manual: s(a) = 1/10 sum_i min(1, 1/3 sum_{j != i} 1[a = a_j]) over the 10 human answers."""
    predicted=normalize_answer(prediction)
    normalized=[]
    for reference in references:
        normalized.append(normalize_answer(reference))
    total=0.0
    for i in range(len(normalized)):
        matches=0
        for j in range(len(normalized)):
            if j!=i and normalized[j]==predicted:
                matches=matches+1
        total=total+min(1.0,matches/3.0)
    return total/len(normalized)
