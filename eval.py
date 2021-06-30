from transformers import EncoderDecoderModel
from transformers import AutoTokenizer, BertTokenizerFast
import json
import re

def json_reader(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

""" json_transcripts is a file with the ground truths and 
      original ASR-transcriptions, one json-file on each line. Like this:
      {'GT':'truetrue', 'ASR':'tootoo'}
      {'GT':'other line', 'ASR':"otherly'}
      gt_asr_trans is a path to a file where the ground truths, original ASRs 
      *and* the new translations will be saved.
      stats is a file where the WER, WWER and semantic similarity will be saved.
      The model is an encoderdecoder model from huggingface, and the 
      tokenizer its corresponding tokenizer."""


def translate_transcripts(json_transcripts, gt_asr_trans_f, stats_f, model, tokenizer):
    gt_asr_trans = open(gt_asr_trans_f, 'a', encoding="utf-8")
    stats = open(stats_f, 'a', encoding="utf-8")
    asr_all_wers = []
    asr_all_wwers = []
    asr_all_semsims = []
      
    trans_all_wers = []
    trans_all_wwers = []
    trans_all_semsims = []

    asr_total_wer = 0
    asr_total_errors = 0
    asr_total_words = 0
    asr_total_wwer = 0
    asr_total_semsim = 0

    trans_total_wer = 0
    trans_total_errors = 0
    trans_total_words = 0
    trans_total_semsim = 0
    trans_total_wwer = 0

    ASRs = []
    GTs = []
      
    for t in json_reader(json_transcripts):
      asr = t['ASR']
      gt = t['GT']
      asr = re.sub('[^a-zA-ZåäöÅÄÖ\s]+', '', asr)
      asr = asr.lower()
      gt = re.sub('[^a-zA-ZåäöÅÄÖ\s]+', '', gt)
      gt = gt.lower()
        
      ASRs.append(asr.lower())
      GTs.append(gt.lower())

      # Metrics for the originial ASR output
      asr_wer, asr_errors, asr_words = get_wer(gt.lower(), asr.lower())
      asr_total_wer += asr_wer
      asr_total_errors += asr_errors
      asr_total_words += asr_words
      
      asr_wwer = get_wwer(gt.lower(), asr.lower())
      asr_total_wwer += asr_wwer
      
      asr_semsim = get_semsim(gt.lower(), asr.lower())
      asr_total_semsim += asr_semsim

      asr_all_wers.append(asr_wer)
      asr_all_wwers.append(asr_wwer)
      asr_all_semsims.append(asr_semsim)

    print("Encoding and decoding...")
    M = 0
    i = 0
    while M < len(ASRs):
      ASRs_batch = ASRs[M:M+100]
      generated_ids = tokenizer.batch_encode_plus(ASRs_batch, padding=True, max_length=32, return_tensors="pt")
      greedy_output = model.generate(generated_ids.input_ids, max_length=32)
      decoded_preds = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
      M = M+100
      print(M, "samples encoded and decoded. Calculating metrics for those.")
      for trans in decoded_preds:
        gt = GTs[i]
        asr = ASRs[i]
    
        # Metrics for the "translated" transcriptions
        trans = re.sub('[^a-zA-ZåäöÅÄÖ\s]+', '', trans).lower()
        trans_wer, trans_errors, trans_words = get_wer(gt.lower(), trans)
        trans_total_wer += trans_wer
        trans_total_errors += trans_errors
        trans_total_words += trans_words

        trans_wwer = get_wwer(gt.lower(), trans)
        trans_total_wwer += trans_wwer
      
        trans_semsim = get_semsim(gt.lower(), trans)
        trans_total_semsim += trans_semsim

        trans_all_wers.append(trans_wer)
        trans_all_wwers.append(trans_wwer)
        trans_all_semsims.append(trans_semsim)

      # Write all the transcriptions 
      # (ground truth, first asr, translated)
      # to a json-file

        gt_asr_trans.write("{\"GT\":\"")      
        gt_asr_trans.write(gt.upper() + "\"")

        gt_asr_trans.write(", \"ASR\":\"")
        gt_asr_trans.write(asr.upper() + "\"")
      
        gt_asr_trans.write(", \"TRANS\":\"")
        gt_asr_trans.write(trans.upper() + "\"}\n")
      
        i += 1
    j = len(asr_all_semsims)
    print(j, "samples used for trans")
    print("ASR avg. WER:", asr_total_wer/j)
    print("ASR total WER:", asr_total_errors/asr_total_words)
    print("ASR avg. WWER", asr_total_wwer/j)
    print("ASR avg sem.sim", asr_total_semsim/j)
    print("Trans. avg. WER:", trans_total_wer/j)
    print("Trans. total WER:", trans_total_errors/trans_total_words)
    print("Trans. avg. WWER", trans_total_wwer/j)
    print("Trans. avg sem.sim", trans_total_semsim/j)
    print("Diff in WER:", (trans_total_wer - asr_total_wer)/j)
    print("Diff in WWER:", (trans_total_wwer - asr_total_wwer)/j)
    print("Diff in sem.sim:", (trans_total_semsim - asr_total_semsim)/j)
    

    print("Done. Writing metrics to file.")
    gt_asr_trans.close()
    stats.write(json_transcripts + " used for evaluation" + "\n")
    stats.write("Total avg. ASR WER: " + str(asr_total_wer) + "\n")
    stats.write("Total ASR WWER: " + str(asr_total_wwer) + "\n")
    stats.write("Total ASR sem.sim.: " + str(asr_total_semsim) + "\n")
    
    stats.write("Total translated WER: " + str(trans_total_wer) + "\n")
    stats.write("Total translated WWER: " + str(trans_total_wwer) + "\n")
    stats.write("Total translated sem.sim.: " + str(trans_total_semsim) + "\n")

    stats.write(str(j) + " samples used for trans \n")
    stats.write("ASR avg. WER: " + str(asr_total_wer/j) + "\n")
    stats.write("ASR tot. WER: " + str(asr_total_errors/asr_total_words) + "\n")
    stats.write("ASR avg. WWER: " + str(asr_total_wwer/j) + "\n")
    stats.write("ASR avg sem.sim: " + str(asr_total_semsim/j) + "\n")
    
    stats.write("Trans. avg. WER: " + str(trans_total_wer/j) + "\n")
    stats.write("Trans. tot. WER:" + str(trans_total_errors/trans_total_words) + "\n")
    stats.write("Trans. avg. WWER: " + str(trans_total_wwer/j) + "\n")
    stats.write("Trans. avg sem.sim: " + str(trans_total_semsim/j) + "\n")

    stats.write("Diff in WER: " + str((trans_total_wer - asr_total_wer)/j) + "\n")
    stats.write("Diff in WWER: " + str((trans_total_wwer - asr_total_wwer)/j) + "\n")
    stats.write("Diff in sem.sim: " + str((trans_total_semsim - asr_total_semsim)/j) + "\n")
    
    stats.write("ASR Individual WERs: " + str(asr_all_wers) + "\n")
    stats.write("Translated Individual WERs: " + str(trans_all_wers) + "\n")
    stats.write("ASR Individual WWERs: " + str(asr_all_wwers) + "\n")
    stats.write("Translated Individual WWERs: " + str(trans_all_wwers) + "\n")
    stats.write("ASR Individual SemSims: " + str(asr_all_semsims) + "\n")
    stats.write("Translated Individual SemSims: " + str(trans_all_semsims) + "\n")

    stats.close()
    print("Stats written.")
    
import fasttext
import numpy as np
from numpy import dot
from numpy.linalg import norm
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        vector = [float(token) for token in tokens[1:]]
        data[tokens[0]] = np.array(vector)#
    return data

print("Loading vectors...")
word_vectors = load_vectors('cc.sv.medium.vec')
print("done.")

OOVs = 0

def sentence_vector(words):
    sv = np.zeros(300)
    words = words.split()
    for word in words:
        try:
            sv = np.add(sv, word_vectors[word.lower()])
        except:
            global OOVs
            OOVs += 1
    
    sv = sv/len(words)
    return sv

def get_semsim(s1, s2):
    # returns semantic similarity between 
    # sentence s1 and s2
    vec1 = sentence_vector(s1)
    vec2 = sentence_vector(s2)
    semsim = cosine(vec1, vec2)
    return semsim
    

def cosine(a, b):
    if norm(a)!= 0 and norm(b) != 0:
        return dot(a, b)/(norm(a)*norm(b))
    else:
        return -1

import sys
import numpy


def get_wer(ref, hyp, debug=False):
    '''
    http://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html
    '''
    DEL_PENALTY = 1
    SUB_PENALTY = 1
    INS_PENALTY = 1

    ref = ref.replace(",", "")        
    ref = ref.replace(".", "")
    ref = ref.replace("?", "")

    hyp = hyp.replace(",", "")        
    hyp = hyp.replace(".", "")
    hyp = hyp.replace("?", "")

    hyp = hyp.lower()
    ref = ref.lower()
    #print("asr:", hyp.encode('utf-8'), "gt", ref.encode('utf-8'))
 
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
     
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
         
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
     
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
    
    #return (numSub + numDel + numIns) / (float) (len(r))
    errors = costs[-1][-1]
    words = len(r)
    wer_result = round( errors / (float) (len(r)), 3)
    return wer_result, errors, words


import numpy as np
import argparse
import numpy as np
import codecs
import json

# WWER
def create_self_info(fname):
    words_selfinfo = {}
    lines = 0
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            info = line.split('\t')
            word = info[0]
            idfish = float(info[5])/1000000
        
            if not words_selfinfo.get(word):
                words_selfinfo[word] = idfish
            else:
                oldidfish = words_selfinfo[word]
                newidfish = oldidfish + idfish
                words_selfinfo[word] = newidfish
            lines += 1

    for key, item in words_selfinfo.items():
        words_selfinfo[key] = -np.log(item)
    
    return words_selfinfo, lines 

print("Creating self information...")
words_selfinfo, lines = create_self_info("stats_BLOGGMIX2016.txt")
print("done.", lines, "lines read")
print("value for .", words_selfinfo["."])
print("value for \'hopptränare\'", words_selfinfo["hopptränare"])

def word_cost(word):
  global words_selfinfo
  try:
    cost = words_selfinfo[word]
  except:
    cost = 16.689060597650638
  return cost

def w_editDistance(r, h):
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else: # r[i-1] != h[j-1]
                # r GT, h ASR
                substitute = d[i-1][j-1] + word_cost(r[i-1])
                insert = d[i][j-1] + word_cost(h[j-1])
                delete = d[i-1][j] + word_cost(r[i-1])
                d[i][j] = min(substitute, insert, delete)
    return d

def get_wwer(r, h):
    r = r.split()
    h = h.split()
    # build the matrix
    d = w_editDistance(r, h)
    result = 0
    try:
      result = float(d[len(r)][len(h)]) / len(r) 
    except:
      print("len 0?", r)
    return result

def eval_stats(model_path, orig_transcripts, gt_asr_translations, stats):  
    model = EncoderDecoderModel.from_pretrained(model_path)
    
    tokenizer = BertTokenizerFast.from_pretrained("KB/bert-base-swedish-cased")

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size
    model.config.max_length = 32
    model.config.min_length = 1
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    translate_transcripts(orig_transcripts, gt_asr_translations, stats, model, tokenizer)


# the naming of the stat and transcription-files is "model name and data used when training it_testset_stats"
# b2b is bert2bert, f2b is freeze2bert
# seq is the sequential training, done 3 epochs on generated data
# mixed is the traditional training, done 3 epochs on a mix of generated and authentic data

eval_stats('translator_model', 'testset', 'all_transcripts', 'all_stats')

