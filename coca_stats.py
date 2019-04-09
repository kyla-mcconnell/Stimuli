from os import walk
from bs4 import BeautifulSoup
import json
import psutil
import shutil
import pickle
from decimal import Decimal
import settings
import re
from multiprocessing import Pool, Manager
import collections
import tqdm
from math import log10
from math import sqrt
from math import log
import os
import sys

# Helper functions to allow multiprocessing
def gscorer(items):
	"""Take a string line (JSON serialized list) as input, return bigram + G-score"""
	global gscores
	bigram, bf, item1_2, item2_2, item1_3, item2_3 = json.loads(items)
	
	score1 = (Decimal(bf) * Decimal(item1_2))/Decimal(item1_3)
	score2 = (Decimal(bf) * Decimal(item2_2))/Decimal(item2_3)
		
	return([bigram, float(score1.ln() + score2.ln())])

# The operations in calculation of Log-likelihood. 
# This allows list comprehension to be used instead of searching through the list one item at a time
order = [1, 1, 1, 1, -1, -1, -1, -1, 1]
def llscorer(items):
	"""Take a string line (JSON serialized list) as input, return bigram + LL-score

	a the frequency of node - collocate pairs
	b number of instances where the node does not co-occur with the collocate
	c number of instances where the collocate does not co-occur with the node
	b the number of words in the corpus minus the number of occurrences of the node and the collocate

	The collocation value is calculated as follows:

	2*( a*log(a) + b*log(b) + c*log(c) + d*log(d)
	- (a+b)*log(a+b) - (a+c)*log(a+c)
	- (b+d)*log(b+d) - (c+d)*log(c+d)
	+ (a+b+c+d)*log(a+b+c+d))"""
	
	global ll_score
	bigram, a,b,c,d = json.loads(items)
	
	a = Decimal(a)
	b = Decimal(b)
	c = Decimal(c)
	d = Decimal(d)
	
	base = [a, b, c, d, a+b, a+c, b+d, c+d, a+b+c+d]
		
	logs = [Decimal(log(x, 10)) for x in base]
	parts = [x*y for x,y in zip(base, logs)]
	parts = [x*float(y) for x,y in zip(order, parts)]
	
	return([bigram, 2*sum(parts)])

def preprocess(filename, queue):
	# print(filename)
	with open(filename, "r") as i:
		doc = i.read()
		
	doc = re.sub("_", "", doc)
	doc = doc.split("\n")
	doc = [word.split("\t") for word in doc]
	doc = [word for word in doc if len(word) == 3]
	doc = [word for word in doc if "@" != word[0]]
	doc = ["_".join([word[0], word[2]]) for word in doc]
	doc = [str(doc[x]).strip() + " " + str(doc[x+1]).strip() for x in range(len(doc)-1) if not (doc[x].endswith("_y") or doc[x+1].endswith("_y"))]		# Drop bigrams that have the full stop in them
		
	queue.put(json.dumps(doc))

def listener(queue):
	f = open("_COCA2.txt", 'w') 
	while 1:
		m = queue.get()
		if m == 'kill':
			break
		f.write(str(m) + '\n')
		f.flush()
	f.close()
	
# Multiprocessing needs this if-statement, otherwise it won't work properly	
if __name__ == "__main__":

	ram_present = psutil.virtual_memory()[0] >> 30
	if ram_present < 7:
		print("WARNING: This is RAM-intensive operation. It cannot continue if you don't have at least 8 GB of RAM.\nExiting...")
		sys.exit(0)		
	
	total, used, free = shutil.disk_usage("\\")
	print("Free drive space: %d/%d GB" % ((free // (2**30)), (total // (2**30))))
	if (free // (2**30)) < 15:
		print("WARNING: This is space-intensive operation. It cannot continue if you don't have at least 15 GB of free space on the same drive as the script.\nExiting...")
		sys.exit(0)	
		
	print("Initializing...")
	files = []
	for dirpath, dirnames, filenames in os.walk("C:/projects/COCA"):
		files.extend([os.path.join(dirpath, file) for file in filenames])
	
	if os.path.isfile("_COCA2.txt"):
		print("A file _COCA2.txt was found in the script's directory.")
		print("This could be the preprocessed corpus. In that case, you can skip the preprocessing.")
		print("Otherwise, this script can delete it and preprocess the corpus.")
		print("Clean the file _COCA2.txt?")
		dec = ""
		while dec.lower() not in set(["y", "n"]):
			dec = input("[y/N]") or "N"
			
		if dec.lower() == "y":
			f = open("_COCA2.txt", 'w+')
			f.close()
			
			manager = Manager()
			queue = manager.Queue()    
			pool = Pool(4)

			#put listener to work first
			watcher = pool.apply_async(listener, (queue,))

			#fire off workers
			jobs = []
			for filename in files:
				job = pool.apply_async(preprocess, (filename, queue))
				jobs.append(job)

			# collect results from the workers through the pool result queue
			for job in tqdm.tqdm(jobs): 
				job.get()

			#now we are done, kill the listener
			queue.put('kill')
			pool.close()
			
		else:
			pass
		
	else:
		f = open("_COCA2.txt", 'w+')
		f.close()		
		manager = Manager()
		queue = manager.Queue()    
		pool = Pool(4)

		#put listener to work first
		watcher = pool.apply_async(listener, (queue,))

		#fire off workers
		jobs = []
		for filename in files:
			job = pool.apply_async(preprocess, (filename, queue))
			jobs.append(job)

		# collect results from the workers through the pool result queue
		for job in tqdm.tqdm(jobs): 
			job.get()

		#now we are done, kill the listener
		queue.put('kill')
		pool.close()

	print("    Bigrams collected")	
	
	print("Preprocessing finished")
	print("Counting scores")
	
	print("    Bigram frequency")
		#- Bigram frequency (bi.freq.NXT)
	coca = open("_COCA2.txt", "r")	
	counter = collections.Counter()


	i = 0
	for line in tqdm.tqdm(coca):
		bgs = json.loads(line)
		bgs = [x.strip() for x in bgs if not x.startswith("##")]
		counter.update(bgs)
		i+=1
		if i == 1:
			break
		
	coca.close()	
	print("        Cropping the bigram dict to items with freq > 4")
	counter = {k:v for k,v in counter.items() if v > 4}
	counter = dict(counter)

	# backup bigram stats file
	print("        Saving")
	backup_out = open("bigrams_debug.json", "w+")
	backup_out.write(json.dumps(counter))	
	backup_out.close()
		
	print("    Making a wordcount from %i bigrams" % (len(counter)))
	w_freq = collections.Counter()
	for item in tqdm.tqdm(counter):	# Get the word frequency for each word
		buffer = counter[item]*item.split()
		w_freq.update(buffer)
		
	w_freq = dict(w_freq)

	backup_out = open("wfreqs_debug.json", "w+")
	backup_out.write(json.dumps(w_freq))
	backup_out.close()
	
	w_count = 0
	for item in w_freq:								# Sum the word frequency to get the total
		w_count += w_freq[item]
		
	print("    TP-D/TP-B")

	#- Direct transitional probability (TPD.bi.NXT)
	#How likely is word x+1 to occur after word x?
	#Backwards transitional probability (TPB.bi.NXT)
	#How likely is word y-1 to occur before word y?

	forward_pairs = {}
	backward_pairs = {}
	
	print("        Dictionarizing word-pair counts")
	#create dict of dicts: how many times is a given word followed by another word and vice versa
	
	for item in tqdm.tqdm(counter): 
		x_word = item.split()[0]
		y_word = item.split()[1]

		# Check the forward freq dictionary, if word x+1 there, add 1 to freq. 
		# If word x+1 not there add it with freq = 1. 
		# If word x not there, create it and add word x+1 as first item with freq = 1.
		if x_word in forward_pairs:
			if y_word in forward_pairs[x_word]:
				forward_pairs[x_word][y_word] += 1
			else:
				forward_pairs[x_word][y_word] = 1
		else:
			forward_pairs[x_word] = {y_word : 1}
			
		# Check the backward freq dictionary, if word y-1 there, add 1 to freq. 
		# If word y-1 not there add it with freq = 1. 
		# If word y not there, create it and add word y-1 as first item with freq = 1.	
		if y_word in backward_pairs:
			if x_word in backward_pairs[y_word]:
				backward_pairs[y_word][x_word] += 1
			else:
				backward_pairs[y_word][x_word] = 1
		else:
			backward_pairs[y_word] = {x_word : 1}

	print("        Pairs collected")

	print("        Counting TPD")	
	# Calculates forward probabilities and saves them as a Decimal(probability)
	# The output variable forward_probs is a dict of dicts, form: {"word_x": {"word_x+1(1)": Decimal(0.73), "word_x+1(2)": Decimal(0.27)}}	
	forward_probs = {}			
	for x_word in tqdm.tqdm(forward_pairs):
		total = Decimal(0)
		for item in forward_pairs[x_word]:
			total += Decimal(forward_pairs[x_word][item])
		forward_probs[x_word] = {}	
		for item in forward_pairs[x_word]:
			forward_probs[x_word][item] = float(Decimal(forward_pairs[x_word][item])/total)

	del forward_pairs

	print("        Counting TPB")						
	# Calculates forward probabilities and saves them as a Decimal(probability)
	# The output variabe forward_probs is a dict of dicts, form: {"word_x": {"word_x+1(1)": Decimal(0.73), "word_x+1(2)": Decimal(0.27)}}	
	backward_probs = {}			
	for x_word in tqdm.tqdm(backward_pairs):
		total = Decimal(0)
		for item in backward_pairs[x_word]:
			total += Decimal(backward_pairs[x_word][item])
		backward_probs[x_word] = {}	
		for item in backward_pairs[x_word]:
			backward_probs[x_word][item] = float(Decimal(backward_pairs[x_word][item])/total)

	del backward_pairs

	backup_out = open("fwd_debug.json", "w+")
	backup_out.write(json.dumps(forward_probs))
	backup_out.close()
				
	backup_out = open("bckw_debug.json", "w+")
	backup_out.write(json.dumps(backward_probs))	
	backup_out.close()

	print("    MI/MI3")				
			
	# - Mutual information score (MI.NXT for word i given word i-1; doesn't look beyond!)
	#log(Bigram_freq/((item1_freq*item2_freq)/WORDCOUNT))
	
	mi_score = {}
	mi3_score = {}
	
	log10_2 = Decimal(log10(2))	# Do not calculate log10(2) every time around
	if settings.mi == "BNC" or settings.mi == "BYU":
		for bigram in tqdm.tqdm(counter):
			
			item1, item2 = bigram.split()
			item1_freq = Decimal(w_freq[item1])
			item2_freq = Decimal(w_freq[item2])
			
			## Used by BNCweb/BYU		
			denom = item1_freq*item2_freq
			score = Decimal(counter[bigram]*Decimal(w_count))/denom
			mi_score[bigram] = float(Decimal(log(score,10))/log10_2)
			
			score3 = Decimal((counter[bigram]**3)*Decimal(w_count))/denom
			mi3_score[bigram] = float(Decimal(log(score3,10))/log10_2)
	
	else:
		## Based on Wiechmann 2008
		item1, item2 = bigram.split()
		item1_freq = Decimal(w_freq[item1])
		item2_freq = Decimal(w_freq[item2])					
		
		score = Decimal(counter[bigram])/((item1_freq*item2_freq)/Decimal(w_count))	
		mi_score[bigram] = float(score.ln())

	backup_out = open("miscore_debug.json", "w+")
	backup_out.write(json.dumps(mi_score))
	backup_out.close()	
	del mi_score
	backup_out = open("mi3score_debug.json", "w+")
	backup_out.write(json.dumps(mi3_score))	
	backup_out.close()	
	del mi3_score

	print("    z-score")				
			
	# - Z score

	# prob = Wi-1/(w_count-Wi)
	# expected = prob * Wi
	# z-score = bigram-expected/sqrt(expected*(1-prob))
	
	z_score = {}	
	for bigram in tqdm.tqdm(counter):
		
		item1, item2 = bigram.split()
		item1_freq = Decimal(w_freq[item1])
		item2_freq = Decimal(w_freq[item2])
		
		## Used by BNCweb/BYU		
		prob = item1_freq/Decimal(w_count-item2_freq)		# probability of item1
		expe = prob*item2_freq								# expected number of bigrams
		numer = Decimal(counter[bigram])-expe				# deviation from the expected number
		denom = Decimal(sqrt(expe*(Decimal(1)-prob)))		# std.deviation (kind of)
		z_score[bigram] = float(numer/denom)
		
	backup_out = open("zscore_debug.json", "w+")
	backup_out.write(json.dumps(z_score))
	backup_out.close()	
	del z_score

	print("    t-score")		
	t_score = {}

	dec_w_count = Decimal(w_count)							# Do not express the word count as a Decimal every time
	for bigram in tqdm.tqdm(counter):
		
		item1, item2 = bigram.split()

		a = Decimal(counter[bigram])
		b = Decimal(w_freq[item1])
		c = Decimal(w_freq[item2])
		
		expe = ((a+b)*(a+c))/dec_w_count
		
		# Based on Gries
		t_score[bigram]= float((a-expe)/Decimal(sqrt(expe)))
		
	backup_out = open("tscore_debug.json", "w+")
	backup_out.write(json.dumps(t_score))		
	backup_out.close()	
	del t_score

	print("    delta-p-score")		
	delta_p21 = {}
	delta_p12 = {}
	
	dec_w_count = Decimal(w_count)							# Do not express the word count as a Decimal every time
	for bigram in tqdm.tqdm(counter):
		
		item1, item2 = bigram.split()

		a = Decimal(counter[bigram])
		b = Decimal(w_freq[item1])
		c = Decimal(w_freq[item2])
		d = dec_w_count
		
		p1 = Decimal(a)/Decimal(a+b)
		p2 = Decimal(c)/Decimal(c+d)
		
		# Based on Gries
		delta_p21[bigram]= float(p1-p2)

		p1 = Decimal(a)/Decimal(a+c)
		p2 = Decimal(b)/Decimal(b+d)
		
		# Based on Gries
		delta_p12[bigram]= float(p1-p2)
		
	backup_out = open("delta_p21_debug.json", "w+")
	backup_out.write(json.dumps(delta_p21))
	backup_out.close()	
	
	backup_out = open("delta_p12_debug.json", "w+")
	backup_out.write(json.dumps(delta_p12))		
	backup_out.close()
	
	del delta_p21
	del delta_p12
	
	print("    Modified Dice-score")	
	# modified Dice coefficient; using a,b,c,d just like LL
	# mod. dice = log2()
	
	dice_score = {}
	for bigram in tqdm.tqdm(counter):
		
		item1, item2 = bigram.split()

		a = Decimal(counter[bigram])
		b = Decimal(w_freq[item1])
		c = Decimal(w_freq[item2])
		
		score = Decimal(2)*(a*a)/(a+b+a+c)
		
		dice_score[bigram]= log(float(score),2)
		
	backup_out = open("dicescore_debug.json", "w+")
	backup_out.write(json.dumps(dice_score))	
	backup_out.close()	
	del dice_score
	
	print("    Log-likelihood")				
	print("        Preparing LL-score calculation")			
	
	lltemp = open("lltemp.bck", "w+")	# We'll use a temp file to save memory
	
	for bigram in tqdm.tqdm(counter):		# Prepare the inputs and save them to a temp file - allow multiprocessing without straining RAM
		item1, item2 = bigram.split()
		
		a = counter[bigram]
		b = w_freq[item1]
		c = w_freq[item2]
		d = w_count-b-c	

		lltemp.write(json.dumps([bigram, a, b, c, d]) + "\n")
		
	lltemp.close()										# Write access no longer needed
	lltemp = open("lltemp.bck", "r")					# Open with read access only
	lt = lltemp.readlines()								# Could be moved to the imap() call, but then the length would be uncertain
	
	print("        Counting")	
	worker = Pool(4)
	ll_score = []
	for i in tqdm.tqdm(worker.imap_unordered(llscorer, lt), total=len(lt)):		# Counting is done on 4 cores, output saved in ll_score
		ll_score.append(i)
	
	del lt
	worker.close()
	worker.join()
	
	print("        Dictionarizing and saving")
	ll_score = dict(ll_score)
	backup_out = open("llscore_debug.json", "w+")	
	backup_out.write(json.dumps(ll_score))
	backup_out.close()		
	del ll_score
	
	lltemp.close()																# We don't need the connection anymore
	try:
		os.remove("lltemp.bck")	
	except:
		print("Couldn't remove the file lltemp.bck, please do it manually")
	print("    G-score")	

		# - Lexical gravity G (G.NXT)

		# log((Fbigr * TypeFreqAfterW1)/Fw1) + log((Fbigr * TypeFreqBeforeW2)/Fw2)

	# G-score needs the number of types following/preceding an item
	fwd_types = {}
	for item in forward_probs:
		fwd_types[item] = len(forward_probs[item])
		
	bckw_types = {}
	for item in backward_probs:
		bckw_types[item] = len(backward_probs[item])
			
	gtemp = open("gscoretemp.bck", "w+")						# We'll use a temp file to save memory
	
	print("        Preparing G-score calculation")
	for bigram in tqdm.tqdm(counter):								# The calculation is prepared as a file with JSON-serialized inputs to the llcounter() function
		item1, item2 = bigram.split()
		item1_3 = w_freq[item1]
		item2_3 = w_freq[item2]
		item1_2 = fwd_types[item1]
		item2_2 = bckw_types[item2]
		bf = counter[bigram]
		gtemp.write(json.dumps([bigram, bf, item1_2, item2_2, item1_3, item2_3]) + "\n")

	gtemp.close()																# We don't need the write access anymore
		
	gtemp = open("gscoretemp.bck", "r")						# Open with read-only
	gt = gtemp.readlines()														# Could be moved to the imap() call, but then the length would be uncertain
	print("        Ready")
	worker = Pool(4)
	g_score = []
	
	for i in tqdm.tqdm(worker.imap_unordered(gscorer, gt), total=len(gt)):		# Counting is done on 4 cores, output saved in g_score
		g_score.append(i)
	
	del gt
	worker.close()
	worker.join()
	print("        Dictionarizing and saving")
	g_score = dict(g_score)
	backup_out = open("gscore_debug.json", "w+")
	backup_out.write(json.dumps(g_score))
	backup_out.close()		
	del g_score
	
	gtemp.close()																# We don't need the connection anymore
	try:
		os.remove("gscoretemp.bck") 	
	except:
		print("Couldn't remove the file gscoretemp.bck, please do it manually")
		
	########### This is a clumsy way of converting the calculated scores into a pandas DataFrame; future versions should get rid of it
	from convert_to_pd import Converter
	worker = Converter()	
	worker.convert()	
	
	exit()	