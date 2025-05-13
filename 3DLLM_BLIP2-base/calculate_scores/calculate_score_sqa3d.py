from time import sleep
import copy
from collections import Counter, defaultdict
import re,glob,csv,json
import sys,os
import pickle

import nltk
nltk.download('omw-1.4', download_dir='./')
import random
import numpy as np
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
sys.path.append(os.path.join(os.getcwd()))


articles = ["a", "an", "the"]

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(,)(\d)")
punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText

def get_lemma(ss):
    return [lemmatizer.lemmatize(token) for token in ss.split()]


def simple_ratio(numerator,denominator): 
    num_numerator=sum([1 if token in numerator else 0 for token in denominator])
    num_denominator=len(denominator)
    return num_numerator/num_denominator


def tokens_unigram_f_value(ref: str,pred: str)->float:
    ref_lemma = get_lemma(ref)
    pred_lemma = get_lemma(pred)
    precision = simple_ratio(ref_lemma,pred_lemma)
    recall    = simple_ratio(pred_lemma,ref_lemma)
    return 2*(recall*precision)/(recall+precision) if recall+precision!=0. else 0


def tokens_score(ref: str,pred: str)->float:
    return 1. if ref==pred else 0.


def evals_json(gold_data,preds):
    score_list = ['Top1 (EM)']
    score = {s:[] for s in score_list}
    
    for ins in gold_data:
        question_id=ins['question_id']
        question=ins['question']
        ref_answers=ins['answers']
      
        pred=preds[question_id]

        # top-1
        answer = pred['answer']
        if answer in ref_answers:
            score['Top1 (EM)'].append(1)
        else:
            #scores=[tokens_unigram_f_value(answer,ref) for ref in ref_answers]
            score['Top1 (EM)'].append(0)
        
    rlt={}
    for k,v in score.items():
        assert len(v)==len(gold_data),len(v)
        rlt[k]=np.mean(v)*100
    return rlt

def eval_pycoco(gold_data, preds, use_spice=False):
    score_list = ['Top1 (EM)','Top1 (F-value)','BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    score = {s:[] for s in score_list}
    
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
    ]
    if use_spice:
        scorers.append((Spice(), "SPICE"))

    tokenizer = PTBTokenizer()
    # pycocoeval
    gts = {ins['question_id']:[{'caption':ans} for ans in ins['answers']] for ins in gold_data}
    res = {qid:[{'caption':value['answer']}] for qid,value in preds.items()}
     
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    
    # =================================================
    # Compute scores
    # =================================================
    rlt={}
    for scorer, method in scorers:
        #eprint('computing %s score...'%(scorer.method()))
       
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
         #       print("%s: %0.3f"%(m, sc*100))
                rlt[m]=sc*100
        else:
          #  print("%s: %0.3f"%(method, score*100))
            rlt[method]=score*100
    return rlt

QT=['All', 'What', 'How', 'Can', 'Is', 'Which', 'Other']
def qclass1(question):
    lques = question
    if 'What' in lques:
        return 'What'
    if 'How' in lques:
        return 'How'
    if 'Can' in lques:
        return 'Can'
    if 'Is' in lques:
        return 'Is'
    if 'Which' in lques:
        return 'Which'
    return 'Other'
            
if __name__=="__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder", type=str, help="Folder containing the results", required=True)
    parser.add_argument("--epoch", type=int, help="epoch for evaluation", required=True)

    args = parser.parse_args()

    golds = json.load(open(os.path.join('../data/SQA_test.json')))

    args.folder = "../lavis/output/BLIP2/3DQA/" + args.folder    

    e = args.epoch
    apreds = []
    
    all_eval_json_files = [file for file in os.listdir("%s/result/"%args.folder) if file.startswith("val_%d_vqa_result_rank"%e)]
    all_eval_json_files = sorted(all_eval_json_files, key = lambda x: (int(x.replace("val_%d_vqa_result_rank"%e, "").replace(".json", "")), x))

    for file in all_eval_json_files:       
        apreds.extend(json.load(open("%s/result/%s"%(args.folder, file))))

    preds = dict()
    for pred in apreds:
        q = pred['question_id']
        preds[q] = pred

    scores={}

    pc_feat_root = "/data/sungnyun/dataset/voxelized_features_sam_nonzero_preprocess"  

    golds = [ann for ann in golds if "scene_id" in ann and os.path.exists(os.path.join(pc_feat_root, ann["scene_id"] + ".pt"))]

    preds_={k:{} for k in QT}
    golds_={k:[] for k in QT}

    assert len(golds) == len(list(preds.keys()))

    for (q,g) in enumerate(golds):
        preds[q]['answer'] = preds[q]['answer'].replace(",", '').replace("<pad>", '').replace("</s>", "").strip()
        preds[q]['answer'] = preds[q]['answer'].strip().split('\n')[0]
        # preds[q]['answer'] = processDigitArticle(processPunctuation(preds[q]['answer']))

        g['question_id'] = q
        
        # new_answers = []
        # for ans in g['answers']:
        #     new_answers.append(processDigitArticle(processPunctuation(ans)))
        #     g['answers'] = new_answers 

        preds_[qclass1(g['question'])][q]=preds[q]
        golds_[qclass1(g['question'])].append(g)
        preds_['All'][q]=preds[q]
        golds_['All'].append(g)

    for qt in QT:
        score=evals_json(golds_[qt],preds_[qt])

        score2=eval_pycoco(golds_[qt], preds_[qt])
        score.update(score2)
        scores[f"{qt}"]=score

    print (scores)
