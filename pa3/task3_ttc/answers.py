from __future__ import annotations

import re

from math_verify import parse, verify


#answer extraction, normalization and equivalence (fixed before final evaluation, same rules for voting and grading)
#  math       math-verify equivalence
#  mathvista  the official mathvista answer-type / choice / precision rules, multiple choice kept as option letters

NO_ANSWER="NO_ANSWER" #every missing extraction goes to this one group, always graded incorrect


def extract_last_boxed(text):
    """README rule: last literal \\boxed{, scan balanced braces, keep nested latex, strip whitespace.
    Absent, empty or unclosed last box -> None. No numeric fallback."""
    if text is None:
        return None
    start=text.rfind("\\boxed{")
    if start==-1:
        return None
    position=start+len("\\boxed{")
    depth=1
    content_start=position
    while position<len(text):
        char=text[position]
        if char=="{":
            depth=depth+1
        elif char=="}":
            depth=depth-1
            if depth==0:
                content=text[content_start:position].strip()
                if content=="":
                    return None
                else:
                    return content
        position=position+1
    return None #unclosed


#---------------- math ----------------

def math_parse(answer):
    """Parse an extracted answer string with math-verify (wrapped in \\boxed{} so the latex extractor finds it)."""
    return parse("\\boxed{"+answer+"}")


def math_equivalent(a,b,parsed_a,parsed_b):
    """Symmetric equivalence of two extracted answers. Identical strings are always equal."""
    if a==b:
        return True
    if len(parsed_a)==0 or len(parsed_b)==0:
        return False #math-verify could not parse one of them, only identical strings match
    try:
        return bool(verify(parsed_a,parsed_b)) or bool(verify(parsed_b,parsed_a))
    except Exception:
        return False


def math_correct(gold,answer):
    if answer is None:
        return False #NO_ANSWER is always incorrect
    try:
        return bool(verify(math_parse(gold),math_parse(answer)))
    except Exception:
        return False


#---------------- mathvista ----------------

def levenshtein(a,b):
    previous=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        current=[i]+[0]*len(b)
        for j in range(1,len(b)+1):
            cost=1
            if a[i-1]==b[j-1]:
                cost=0
            current[j]=min(previous[j]+1,current[j-1]+1,previous[j-1]+cost)
        previous=current
    return previous[len(b)]


def mathvista_key(extraction,meta):
    """Canonical answer key following mathvista's normalize_extracted_answer, but multiple choice -> option letter."""
    if extraction is None:
        return NO_ANSWER
    text=str(extraction).strip()
    if meta["question_type"]=="multi_choice":
        choices=meta["choices"]
        letters=[]
        for i in range(len(choices)):
            letters.append(chr(ord("A")+i))
        found=re.findall(r"\(([a-zA-Z])\)",text) #"(b)" -> "B", official rule
        if len(found)>0:
            text=found[0].upper()
        if len(text)==1 and text.upper() in letters:
            return text.upper()
        for i in range(len(choices)):
            if text.lower()==str(choices[i]).strip().lower():
                return letters[i] #the option text itself
        #official fallback: most similar option text
        best=0
        best_distance=None
        for i in range(len(choices)):
            distance=levenshtein(text,str(choices[i]))
            if best_distance is None or distance<best_distance:
                best=i
                best_distance=distance
        return letters[best]
    elif meta["answer_type"]=="integer":
        try:
            return str(int(float(text)))
        except Exception:
            return "UNPARSED:"+text #a real answer that is not an integer, its own group, never equal to gold
    elif meta["answer_type"]=="float":
        try:
            return str(round(float(text),int(meta["precision"])))
        except Exception:
            return "UNPARSED:"+text
    else:
        return text #list answers, compared as strings (official)


def mathvista_gold_key(gold,meta):
    if meta["question_type"]=="multi_choice":
        return gold.strip().upper() #the export stores the gold option letter
    else:
        return mathvista_key(gold,meta)
