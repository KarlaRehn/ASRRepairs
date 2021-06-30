# This reads a CSV-file, turns digits into their written version, and writes to a json-file
# The CSV file for ville has the fields 
# mother tongue,ground truth,native or non-native,transcript without commas,transcript
def letterize_digits(orig_text):
	letterized_text = orig_text.replace("É", "E"
		).replace("11.47", "ELVA FYRTIOSJU"
		).replace("70000000", "SJUTTIO MILJONER"
		).replace("1984", "NITTONHUNDRAÅTTIOFYRA"
		).replace("9999", "NIOTUSEN NIOHUNDRANITTIONIO"					
		).replace("2000", "TVÅTUSEN"
            	).replace("06.45", "KVART I SJU"
		).replace("100", "HUNDRA"
            	).replace("150", "HUNDRAFEMTIO"
            	).replace("400", "FYRAHUNDRA"
            	).replace("800", "ÅTTAHUNDRA"
            	).replace("06.45", "KVART I SJU"
            	).replace("6.45", "KVART I SJU"
            	).replace("10", "TIO"
            	).replace("11", "ELVA"
            	).replace("12", "TOLV"
            	).replace("15", "FEMTON"
            	).replace("18", "ARTON"
            	).replace("19", "NITTON"
            	).replace("20", "TJUGO"
            	).replace("22", "TJUGOTVÅ"
            	).replace("32", "TRETTIOTVÅ"
            	).replace("45", "FYRTIOFEM"
            	).replace("47", "FYRTIOSJU"
            	).replace("50", "FEMTIO"
            	).replace("60", "SEXTIO"
            	).replace("70", "SJUTTIO"
            	).replace("77", "SJUTTIOSJU"
            	).replace("80", "ÅTTIO"
            	).replace("81", "ÅTTIOETT"
            	).replace("95", "NITTIOFEM"
		).replace("99", "NITTIONIO"
            	).replace("1", "ETT"
            	).replace("2", "TVÅ"
            	).replace("3", "TRE"
            	).replace("4", "FYRA"
            	).replace("5", "FEM"
            	).replace("6", "SEX"
		).replace("7", "SJU"
		).replace("8", "ÅTTA"
		).replace("9", "NIO")
	return letterized_text

def ville_to_json():
    filename = "microsoft_transcripts_ville.csv"
    jsonfile = open('orig_microsoft_ville_nospeaker.json', 'a')
    
    with open(filename) as f:
        for line in f.readlines():
            line = line.split(",")  
            
            gt = line[1].upper()
            gt = gt.replace("\"", "")
            gt = letterize_digits(gt)
            
            asr = line[3].upper()
            asr = asr.replace("\"", "")
            asr = letterize_digits(asr)
           
            native_l = line[2]
           
            if asr:
            	jsonfile.write("{\"GT\":\"" + gt + "\", \"ASR\":\"" + asr + "\"}\n")#"\", \"NL\":\"" + native_l + "\"}\n")
            else:
            	print("missing asr? here's the gt:", gt) 
    
    jsonfile.close()
    
def sc_to_json():
    filename = "microsoft_transcripts_sprakcafe.csv"
    jsonfile = open('orig_microsoft_sprakcafe_nospeaker.json', 'a')

    with open(filename) as f:
        for line in f.readlines():
            line = line.split(",")
            gt = line[3].upper()
            gt = gt.replace("\"", "")
            gt = letterize_digits(gt)
            
            
            asr = line[4].upper()
            asr = asr.replace("\"", "")
            asr = letterize_digits(asr)
            
            native_l = line[1]
            
            if asr:
            	jsonfile.write("{\"GT\":\"" + gt + "\", \"ASR\":\"" + asr + "\"}\n") #"\", \"NL\":\"" + native_l + "\"}\n")
            else:
            	print("missing asr? here's the gt:", gt) 
            
    jsonfile.close()

#print("SPRÅKCAFE:")
#sc_to_json()
#print("VILLE:")
#ville_to_json()


