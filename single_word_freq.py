import sys
import csv
import tqdm
import psutil
import random
import time
from numpy import array, digitize, zeros, amin, amax, arange, percentile, histogram, argmin, transpose, delete, log, exp, nanmedian
from numpy import random as nrandom
from numpy import sum as nsum
from math import floor, ceil
from nltk import word_tokenize, pos_tag
import matplotlib.pyplot as plt
from copy import deepcopy
from re import match, sub, compile
import json
import os
import pandas as pd

import gc
gc.enable()

score_path = "E:/stimuli_coca_exp19/scores/"
mapping_path = "E:/stimuli_coca_exp19/mapping.json"


class Tagger:
	def __init__(self, iterable_of_tagged):
		self.tagging_map = {}
		for tagged_bigram in iterable_of_tagged:
			tagged_word = tagged_bigram.split()[0].lower().strip()
			untagged_word = sub("_[^ ]+", "", tagged_word)
			tagged_word = untagged_word + "_" + tagged_word.split("_")[1][0]
			if untagged_word in self.tagging_map:
				if tagged_word not in self.tagging_map.get(untagged_word):
					self.tagging_map[untagged_word].append(tagged_word)
			else:
				self.tagging_map[untagged_word] = [tagged_word]
		print(self.tagging_map)

	def tag(self, item_to_tag):
		"""Tag an item. If only one option is know, this is assumed to be the correct option.
		If multiple options are known, the correct one can be picked or specified manually.
		If no option is known, the tags should be written manually."""
		try:
			tags = self.tagging_map[item_to_tag]
			if len(tags) == 1:
				return tags[0]
			elif len(tags) > 1:
				return self.pick_a_tag_from_selection(item_to_tag, tags)
		except:
			return self.tag_manually(item_to_tag)

	def pick_a_tag_from_selection(self, item_to_tag, tags):
		"""Allow the user to select one of multiple offered tags or specify their own tagging"""
		print("_"*10)
		print("There are multiple possible tags for '{}'".format(item_to_tag))
		for index, tag in enumerate(tags):
			print("\t[{}] {}".format(index, tag))
			print("\n\t[m] a manual tag")

		choice = None;
		while choice == None:
			new_choice = input("\nPick the relevant tag\n> ")
			if new_choice.lower() == "m":
				return self.tag_manually(item_to_tag)
			try:
				new_choice = int(new_choice)
				if (new_choice >= 0) and (new_choice < len(tags)):
					choice = new_choice
			except:
				pass

			return tags[choice]

	def tag_manually(self, item_to_tag):
		"""Allow manual tagging of items not found in the simple tagging map.
		Expects the user to input valid tags"""
		print("_"*10)
		print("The item '{}' cannot be tagged semi-automatically. Please tag it manually.".format(item_to_tag))
		#w1, w2 = item_to_tag.split()

		# TODO: input validation/formatting (lowercase?)
		w1_tag = input("{}_".format(item_to_tag))

		return "{}_{}".format(item_to_tag,w1_tag)



class Scorer(object):
	"""Loads all the score lists, can then be used to assign the scores on a per-item base with the score method.
		It is memory-heavy, but could be included in functions which allow interactive collocation input."""

	def __init__(self):
		"""Load the required score files. If they are not present in the folder, throw an exception."""
		import os

		print("\nLoading the saved scores")
		print("    unigram frequency")

		with open(score_path + "wfreqs_lemma.json", "r") as i:
			self.wfreq = json.loads(i.read())

		print("_________________________________")


	def score(self, items):
		"""Score single words. Input is a list items on separate lines. If a bigram is not in the score list, return NA."""

		items_out = []
		for word in items:
			try:
				w1_frq = self.wfreq[word]
			except:
				w1_frq = "NA"

			items_out.append([word, w1_frq])

		return(items_out)

class ScorerLemma(object):
	"""Loads all the score lists, can then be used to assign the scores on a per-item base with the score method.
		It is memory-heavy, but could be included in functions which allow interactive collocation input."""

	def __init__(self):
		"""Load the required score files. If they are not present in the folder, throw an exception."""
		import os

		print("\nLoading the saved scores")
		print("    unigram frequency")

		with open(score_path + "wfreqs_lemma.json", "r") as i:
			self.wfreq = json.loads(i.read())
			self.wfreq_lemma = {}
			for key, value in self.wfreq.items():
				new_key = sub("_[^ ]+", "", key)
				if new_key in self.wfreq_lemma: 
					new_value = self.wfreq_lemma[new_key] + value
					self.wfreq_lemma[new_key] = new_value
				else: 
					self.wfreq_lemma[new_key] = value

		print("_________________________________")


	def score(self, items):
		"""Score single words. Input is a list items on separate lines. If a bigram is not in the score list, return NA."""

		items_out = []
		for word in items:
			try:
				lemma_freq = self.wfreq_lemma[word]
			except:
				lemma_freq = "NA"

			items_out.append([word, lemma_freq])

		return(items_out)
		
if __name__ == "__main__":

	inpath = input("Where is the file to load?\n     ")

	with open(inpath, "r") as infile:
		raw_items = infile.readlines()
		raw_items = [item.replace('\n', '') for item in raw_items]
		print("First read in items: " + str(raw_items))

		#very VERY VERY crude lemmatization, specific to my stimuli list
		items = []
		for item in raw_items:
			if item in ["species_n", "genius_n", "awareness_n", "dress_n", "access_n", "basis_n", "witness_n", "mathmatics_n", "success_n", "business_n", "ms_n", "works_n", "mathematics_n", "changes_n"]:
				items.append(item)
			elif item.endswith("es_n"):
				if item in ["colleagues_n", "languages_n", "values_n", "places_n", "prices_n", "forces_n", "faces_n", "temperatures_n", "candles_n", "everyones_n", "pressures_n", "circumstances_n", "scenes_n"]:
					items.append(item.replace("es_n", "e_n"))
				else:
					items.append(item.replace("es_n", "_n"))
			elif item.endswith("s_n"):
				items.append(item.replace("s_n", "_n"))
			elif item == "led_v":
				items.append("lead_v")
			elif item == "became_v":
				items.append("become_v")
			elif item == "need_v":
				items.append("need_v")
			elif item == "chose_v":
				items.append("choose_v")
			elif item == "found_v":
				items.append("find_v")
			elif item in ["stared_v", "discouraged_v", "changed_v", "saved_v", "ensured_v", "supposed_v", "collapsed_v", "continued_v", "saturated_v", "figured_v", "created_v", "noted_v"]:
				items.append(item.replace("ed_v", "e_v"))
			elif item in ["carried_v", "studied_v"]:
				items.append(item.replace("ied_v", "y_v"))
			elif item == "deteriorating_v":
				items.append("deteriorate_v")
			elif item.endswith("ing_v"):
				if item in ["locating_v", "noticing_v", "defining_v", "deteriorating_v", "consuming_v"]:
					items.append(item.replace("ing_v", "e_v"))
				elif item == "getting_v":
					items.append("get_v")
				else:
					items.append(item.replace("ing_v", "_v"))
			elif item == "written_v":
				items.append("write_v")
			elif item == "given_v":
				items.append("give_v")
			elif item == "heard_v":
				items.append("hear_v")
			elif item == "came_v":
				items.append("come_v")
			elif item == "got_v":
				items.append("get_v")
			elif item == "saw_v":
				items.append("see_v")
			elif item == "went_v":
				items.append("go_v")
			elif item in ["been_v", "are_v", "is_v", "were_v", "was_v"]:
				items.append("be_v")
			elif item == "blew_v":
				items.append("blow_v")
			elif item == "caught_v":
				items.append("catch_v")
			elif item == "ran_v":
				items.append("run_v")
			elif item == "felt_v":
				items.append("feel_v")
			elif item == "does_v":
				items.append("do_v")
			elif item == "meant_v":
				items.append("mean_v")
			elif item == "took_v":
				items.append("take_v")
			elif item == "made_v":
				items.append("make_v")
			elif item in ["has_v", "had_v", "has_v"]:
				items.append("have_v")
			elif item.endswith("ed_v"):
				items.append(item.replace("ed_v", "_v"))
			elif item.endswith("s_v"):
				if item not in []:
					items.append(item.replace("s_v", "_v"))
			else:
				items.append(item)
		print("Lemmatized items: " + str(items))


	ram_present = psutil.virtual_memory()[0] >> 30
	ram_available = psutil.virtual_memory()[1] >> 30

	# Check the RAM installed and available, if sufficient use the default scorer, otherwise use the lite version
	if ram_present > 7 and ram_available > 5:
		pass
	else:
		print("This is a RAM-intensive operation. You need at least 6 GB of free RAM.")
		print("Exiting...")
		sys.exit(0)

	if match("[^_]+_", items[0]) == None:
		with open(mapping_path, "r") as f:
			print("Loading mapping file.")
			mapping = json.load(f)
			tagger = Tagger(mapping)
			print("-----------")
			tagged_items = []
			print("Tagging items.")
			for item in items:
				try:
					tagged_item = tagger.tag(item)
					tagged_items.append(tagged_item)
				except Exception as e:
					print(str(e) + str(item))
	else:
		tagged_items = items

	print("Tagged items: " + str(tagged_items))

	# with open('tagged_list.json', "a") as fp:
	# 	for item in tagged_items:
	# 		fp.write(item + "\n")
	
	scorer = Scorer()
	items = scorer.score(tagged_items)
	print("Third step items: " + str(items))

	scores = pd.DataFrame(items, columns=["tagged_word", "freq"])

	scores["lemma_stem"] = [sub("_[^ ]+","",x) for x in scores["tagged_word"]]
	scores["word"] = [sub("_[^ ]+","",x) for x in raw_items]

	del scorer
	del items

	lemma_scorer = ScorerLemma()
	lemma_scored_items = lemma_scorer.score(scores["lemma_stem"])

	lemmascores = pd.DataFrame(lemma_scorer.score(scores["lemma_stem"]), columns=["lemma_stem", "lemma_freq"])

	scores = scores.merge(lemmascores, on="lemma_stem", how="outer")

	scores = scores[["word", "lemma_stem", "tagged_word", "freq", "lemma_freq"]]

	scores.drop_duplicates().reset_index(drop=True)

	print(scores)

	print("Saving")

	if len(sys.argv) > 2:
		outpath = sys.argv[2]
	else:
		outpath = input("Where should the results be saved?\n    ")

	print("Saving")
	scores.to_csv(outpath, index=False)
	print("Done. Press RETURN to exit")
	wait = input()
	sys.exit(0)
