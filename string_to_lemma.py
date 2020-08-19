from os import walk
from bs4 import BeautifulSoup
import json
import psutil
import shutil
import pickle
from decimal import Decimal
import settings_stats
import re
from multiprocessing import Pool, Manager
import collections
import tqdm
from math import log10
from math import sqrt
from math import log
import os
import sys

path_to_coca = "E:/coca_2019_wlp"

def preprocess(filename, queue):
	with open(filename, "r", errors="replace") as i:
		try:
			doc = i.read()

			#doc = re.sub("_", "", doc)
			doc = re.sub("\t+", "\t", doc)
			doc = doc.split("\n")
			doc = [word.split("\t") for word in doc]

			docs = []
			d = []
			declined = re.compile("@_")
			for word in doc:
				if len(word) == 4: #changed from 3 to 4
					pos = word[3][0]
					d.append([[word[1], word[3]],[word[2], pos]]) #changed from 0 2 1 20 to 1 3 2 pos (defined above)
				#elif word[1].startswith("##"): #changed from 0 to 1
					# d = [["_".join(w).strip(),"_".join(l).strip()] for w,l in d]
					# d = [[x[0] + " " + y[0], x[1] + " " + y[1]] for x,y in zip(d[0:-1], d[1:]) if not (x[0].endswith("_y") or y[0].endswith("_y"))]
					# d = [[x,y] for x,y in d if re.match(declined, x) == None]
					# docs.append(d)
					# d = []

				else:
					pass

			d = [["_".join(w).strip(),"_".join(l).strip()] for w,l in d]
			punctuation = [",", ".", "!", "?", ";", ":"]
			d = [[x[0] + " " + y[0], x[1] + " " + y[1]] for x,y in zip(d[0:-1], d[1:]) if not (x[0] in punctuation) or (y[0] in punctuation)]
			d = [[x,y] for x,y in d if re.match(declined, x) == None]
			docs.append(d)

			docs = [item for sublist in docs for item in sublist]

			queue.put(json.dumps(docs))

		except:
			print("Error: " + filename)


def listener(queue, filename):
	f = open(filename, 'w')
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
	if (free // (2**30)) < 7:
		print("WARNING: This is space-intensive operation. It cannot continue if you don't have at least 15 GB of free space on the same drive as the script.\nExiting...")
		sys.exit(0)

	print("Initializing...")
	files = []
	for dirpath, dirnames, filenames in os.walk(path_to_coca):
		files.extend([os.path.join(dirpath, file) for file in filenames])

	if os.path.isfile("_COCA2_mapping.txt"):
		print("A file _COCA2_mapping.txt was found in the script's directory.")
		print("This could be the preprocessed corpus. In that case, you can skip the preprocessing.")
		print("Otherwise, this script can delete it and preprocess the corpus.")
		print("Clean the file _COCA2_mapping.txt?")
		dec = ""
		while dec.lower() not in set(["y", "n"]):
			dec = input("[y/N]") or "N"

		if dec.lower() == "y":
			f = open("_COCA2_mapping.txt", 'w+')
			f.close()

			manager = Manager()
			queue = manager.Queue()
			pool = Pool(4)

			#put listener to work first
			watcher = pool.apply_async(listener, (queue, "_COCA2_mapping.txt"))

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
		f = open("_COCA2_mapping.txt", 'w+')
		f.close()
		manager = Manager()
		queue = manager.Queue()
		pool = Pool(4)

		#put listener to work first
		watcher = pool.apply_async(listener, (queue, "_COCA2_mapping.txt"))

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

	with open("scores/bigrams.json", "r") as f:
		frequent = json.load(f)
		frequent = set([x for x in frequent])

	coca = open("_COCA2_mapping.txt", "r")
	mapping = {}
	log = open("conversionlog.txt", "w+")

	for line in tqdm.tqdm(coca, total=115):
		bgs = json.loads(line)
		bgs = [[x.strip(), y.strip()] for x,y in bgs if not x.startswith("##")]
		for string, lemma in bgs:
			if string in frequent:
				if string in mapping:
					if mapping[string] == lemma:
						pass
					else:
						log.write('%s : %s versus %s\n' % (string, lemma, mapping[string]))
				else:
					mapping[string] = lemma
			else:
				pass
	del bgs
	coca.close()

	import gc
	gc.collect()

	with open("mapping.json", "w+") as f:
		json.dump(mapping, f)
