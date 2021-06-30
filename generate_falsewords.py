import re
import os

def read_lexicon(lex_file):
	lexicon = {}	
	phones2word = {}
	print("Reading lexicon")
	with open(lex_file) as f:
		for line in f.readlines():
			try:
				word, phones = line.split("\t")
				phones = phones.strip()
				phones2word[phones] = word
				lexicon[word] = phones
			except:
				print("Something went wrong for line", line)

	print("Created phonetic lexicon with", len(phones2word), "words")
	return lexicon, phones2word

def read_phones():
	all_phones = []
	print("Reading phones")
	with open(phone_file) as f:
		for line in f.readlines():
			all_phones.append(line.strip("\n"))
	return all_phones

def create_phone_neighbors(lexicon, fname):
	print("Creating dict of phonetically similar words as well as writing this to file")
	word2simword = {} # This is a dict with words as keys, and a list of phonetically similar words as values

	loops = 0
	created = 0
	for word, orig_phones in lexicon.items():
		word2simword[word] = []
		orig_phones = orig_phones.split(" ")
		for pos, phone in enumerate(orig_phones):
			for replacement_phone in all_phones:
				loops += 1

				if loops%1000000 == 0:
					print(loops, "words checked.", created, "words added to dict.")

				mod_phones = orig_phones.copy()
				mod_phones[pos] = replacement_phone
				mod_phones = " ".join(mod_phones)
				if mod_phones in phones2word:
					simword = phones2word[mod_phones]
					if simword != word:
						word2simword[word].append(simword)
						created += 1

	# Save the dictionary so we don't have to recreate it
	print("Saving the dictionary of phonetically related words")
	dict_file = open(fname, 'w')

	for word, simwords in word2simword.items():
		dict_file.write(word+" "+" ".join(simwords)+"\n")
		
	return word2simword

def read_phone_neighbors(fn):
	word2simword = {}
	with open(fn) as f:
		for line in f.readlines():
			words = line.split()
			word2simword[words[0]] = words[1:]
	return word2simword 


def gen_falsewords():

# Textfiles where the lexicon, phones, and utterances are saved.
# The lexicon is on on the format WORD	P H O N E S, 
# (for example BARNBOKSBILDER	"b A: n`b u k s %b I l d e r)
# The phone_file is a file with one phone on each line
# The utterances are line-separated, with utterance ID as the first thing on each line.
# (for example 003-r4670003-100 BARNBOKSBILDERNA ÄR OVANLIGT VACKRA)
# The NST data follows this format


	lex_file = "data/lexicon.txt" #"data/small_lexicon.txt" #
	phone_file = "data/nonsilence.txt"
	utter_file = "unique_utters.txt"#"smalltraintexts.txt"
	phone_neighbors = "phonetically_similar.txt"

	lexicon, phones2word = read_lexicon()
	all_phones = read_phones(phone_file)
	
	if os.path.exists(phone_neighbors)
		print("Reading phonetical neighbors...")
		word2simword = read_phone_neighbors(phone_neighbors)
		print("Done reading.")
	else:
		print("Creating dict with phonetical neighbors...")
		word2simword = create_phone_neighbors(lexicon, phone_neighbors)
		print("Done.")
	
	print("Creating artifical ASR-outputs")
	processed_utters = {}
	created_utters = 0
	lines_used = 0

	with open(utter_file) as f:
		for line in f.readlines():
			utterance = line			
			utterance = utterance.replace("\\Punkt", ""
					).replace("\\Komma", ""
					).replace("\\Frågetecken", ""
					).replace("\\Utropstecken", "")
			word_list = [re.sub(('[^a-zA-ZåäöÅÄÖÉé]'), "", word) for word in utterance.split()]

			if len(word_list) > 2:
				lines_used += 1
				for pos, word in enumerate(word_list):
					mod_utter = word_list.copy()
					word = word.upper()
				
					try:
						rep_words = word2simword[word]
					except:
						rep_words = []
	
					for rep_word in rep_words:
						mod_utter[pos] = rep_word
						write_utter = " ".join(mod_utter)
						utterance = utterance.strip()
						modfile = open("slashgen_data.json", 'a')
						modfile.write("{\"ASR\":\"" + write_utter.upper() + "\","
									+ "\"GT\":\"" + utterance.upper() + "\"}\n")
						created_utters += 1
			
						if created_utters%100000 == 0:
							print("Created", created_utters, "artificial transcriptions.")

	print(lines_used, "lines used")


# 287 040 in total in train
# 144 994 unique in train 
