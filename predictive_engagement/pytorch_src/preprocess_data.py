import json
import random
import re
from nltk.tokenize import TweetTokenizer
import csv


def preprocess_data(data_dir, ifile, ofile):
	'''This method 
		1) takes the json file and processes dialogs with at least one turn including two utterances)
		2) creates an output file in the following format: dialogId\tnum_utterance\tengagement\tquality\tbreadth\t<sp1>utterance</sp1><sp2>utterance<sp2/>... 
		3) if both participants in the dialogue are human then engagement/quality/breadth scores are average of their evaluated scores otherwise they are only human's evaluated scores
		4) binarizing engagement score (eng_score=1  if it is greather than or equal to 3 otherwise 0)
	Params:
		data_dir: the directory that contains the ConvAI dataset
		ifile: the json file containing ConvAI original dataset
		ofile: the processed ConvAI dataset
	'''
	with open(data_dir+ifile) as fr:
		data = json.load(fr)
	num_convs = len(data)

	fw = open(data_dir+ofile, 'w')
	tokenizer = TweetTokenizer()

	dialogue_ids = {}
	dialogue_ids_engagement = {}
	dialogue_ids_quality = {}
	dialogue_ids_breadth = {}

	for i in range(num_convs):
		prev_sp = ''
		prev_sp_talk = ''
		list_talk = []
		user_id = {}
		if data[i]["users"][0]["id"] == 'Bob':
			user_id['Bob'] = data[i]["users"][0]["userType"]
			user_id['Alice'] = data[i]["users"][1]["userType"]

		if data[i]["users"][1]["id"] == 'Bob':
			user_id['Bob'] = data[i]["users"][1]["userType"]
			user_id['Alice'] = data[i]["users"][0]["userType"]

		if len(data[i]["thread"]) != 0:
			prev_sp = data[i]["thread"][0]["userId"]
			text= re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", data[i]["thread"][0]["text"])
			text= ' '.join(tokenizer.tokenize(text)) 
			prev_sp_talk = user_id[data[i]["thread"][0]["userId"]] + '@@'  + text
			if len(data[i]["thread"]) == 1:
				list_talk.append(user_id[data[i]["thread"][0]["userId"]] + '@@' + prev_sp_talk)
			else:
				for t in data[i]["thread"][1:]:
					if t["userId"] != prev_sp:
						t["text"]= re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", t["text"])
						t["text"] = ' '.join(tokenizer.tokenize(t["text"]))
						list_talk.append(prev_sp_talk)
						prev_sp_talk = user_id[t["userId"]] + '@@' + t["text"] 
						prev_sp = t["userId"]
					else:
						t["text"]= re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", t["text"])
						t["text"] = ' '.join(tokenizer.tokenize(t["text"]))
						prev_sp_talk += ' '+ t["text"] 
				list_talk.append(prev_sp_talk)

			conversation = []
			for ind, talk in enumerate(list_talk):
				if '\n' in talk:
						talk = talk.replace('\n',' ')
				speaker = talk.split('@@')[0]
				talk = talk.split('@@')[1]

				if ind %2 ==0:
					conversation.append(' <sp1> '  + talk+' </sp1>')
				else:
					conversation.append(' <sp2> ' + talk+' </sp2>')
			dialogue_ids[data[i]["dialogId"]] =  conversation

			if data[i]["users"][0]["userType"] =="Human" and data[i]["users"][1]["userType"] =="Human":
				dialogue_ids_engagement[(data[i]["dialogId"])] = round(float(data[i]["evaluation"][0]["engagement"] + data[i]["evaluation"][1]["engagement"])/2)
				dialogue_ids_quality[(data[i]["dialogId"])] = round(float(data[i]["evaluation"][0]["quality"] + data[i]["evaluation"][1]["quality"])/2)
				dialogue_ids_breadth[(data[i]["dialogId"])] = round(float(data[i]["evaluation"][0]["breadth"] + data[i]["evaluation"][1]["breadth"])/2)
				
			if data[i]["users"][0]["userType"] =="Bot" and data[i]["users"][1]["userType"] =="Bot":
				print("{} th conversation is between two bots".format(i))
				
			elif data[i]["users"][0]["userType"] =="Human" and data[i]["users"][1]["userType"] =="Bot":
				user_id = data[i]["users"][0]["id"]
				if data[i]["evaluation"][0]["userId"] == user_id:
					dialogue_ids_engagement[data[i]["dialogId"]] = data[i]["evaluation"][0]["engagement"] 
					dialogue_ids_breadth[data[i]["dialogId"]] = data[i]["evaluation"][0]["breadth"] 
					dialogue_ids_quality[data[i]["dialogId"]] = data[i]["evaluation"][0]["quality"] 

				else:
					dialogue_ids_engagement[data[i]["dialogId"]] = data[i]["evaluation"][1]["engagement"] 
					dialogue_ids_breadth[data[i]["dialogId"]] = data[i]["evaluation"][1]["breadth"] 
					dialogue_ids_quality[data[i]["dialogId"]] = data[i]["evaluation"][1]["quality"] 


			elif data[i]["users"][1]["userType"] =="Human" and data[i]["users"][0]["userType"] =="Bot":
				user_id = data[i]["users"][1]["id"]
				if data[i]["evaluation"][0]["userId"] == user_id:
					dialogue_ids_engagement[(data[i]["dialogId"])] = data[i]["evaluation"][0]["engagement"] 
					dialogue_ids_breadth[data[i]["dialogId"]] = data[i]["evaluation"][0]["breadth"]
					dialogue_ids_quality[data[i]["dialogId"]] = data[i]["evaluation"][0]["quality"] 


				else:
					dialogue_ids_engagement[data[i]["dialogId"]] = data[i]["evaluation"][1]["engagement"]
					dialogue_ids_breadth[data[i]["dialogId"]] = data[i]["evaluation"][1]["breadth"]
					dialogue_ids_quality[data[i]["dialogId"]] = data[i]["evaluation"][1]["quality"]

	for d_id, d_turns in dialogue_ids.items():
		d_line = ''
		if len(d_turns) >= 2:
			d_line = str(d_id) + '\t'
			d_line += str(len(d_turns)) +'\t'
			#binarizing the engagement score
			if dialogue_ids_engagement[d_id] >= 3:
				eng_labels = '1\t'
			else:
				eng_labels = '0\t'
			d_line += str(eng_labels)
			d_line += str(dialogue_ids_quality[d_id]) +'\t'
			d_line += str(dialogue_ids_breadth[d_id]) +'\t'
			for t in d_turns:
				d_line += str(t)
			fw.write(d_line+'\n')				

	fw.close()



def sub_AMT_set(data_dir, ifile1, ifile2, ofile):
	'''
	This method:
		1) takes the file containing 50 randomly selected dialogs from ConvAI dataset and subtracts it from original ConvAI data (the 50 randomly selected dialogs has been used as a seperate test set)
	Params:
		data_dir: the directory that contains the ConvAI dataset
		file1: the file containing 50 randomly selected dialogs from ConvAI that their utterances engagement scores have been annotated by Amazon turkers
		file2: the whole ConvAI dataset
		ofile: convs from ConvAI dataset without 50 randomly selected dialogs

	'''
	dialogue_engagements = []
	fw_without50convs= open(data_dir+ofile, 'w')
	data_full= open(data_dir+ifile2, 'r')
	convs = data_full.readlines()

	data_50Convs= open(data_dir+ifile1, 'r')
	lines = data_50Convs.readlines()
	list_50convs = []
	for line in lines:
		list_50convs.append(line.split('\t')[0])

	for conv in convs:
		conv_id = conv.split('\t')[0]
		if conv_id not in list_50convs:
			fw_without50convs.write(conv)
			dialogue_engagements.append(conv.split('\t')[2])
	fw_without50convs.close()
	print('number of labels 0 {}'.format(dialogue_engagements.count('0')))
	print('number of labels 1 {}'.format(dialogue_engagements.count('1')))

	


def split_train_test_valid(data_dir, ifile):
	'''
	This method:
		1) splits the ConvAI dialogues into train/valid/test with portions of 60%/20%/20%
	Params:
		data_dir: the directory that contains the ConvAI dataset
		ifile: the ConvAI dataset 

	'''
	fr = open(data_dir+ifile, 'r')
	lines = fr.readlines()
	fw_train = open(data_dir+ifile+'_train', 'w')
	fw_test = open(data_dir+ifile+'_test', 'w')
	fw_valid = open(data_dir+ifile+'_valid', 'w')
	
	list_convs_label0 = []
	list_convs_label1 = []

	for line in lines:
		if line.split('\t')[2] == '0':
			list_convs_label0.append(line)
		elif line.split('\t')[2] == '1':
			list_convs_label1.append(line)
	split_1_label0 = int(0.6 * len(list_convs_label0))
	split_2_label0 = int(0.8 * len(list_convs_label0))
	split_1_label1 = int(0.6 * len(list_convs_label1))
	split_2_label1 = int(0.8 * len(list_convs_label1))
	train_convs_label0 =  list_convs_label0[:split_1_label0]
	test_convs_label0  =  list_convs_label0[split_1_label0:split_2_label0]
	valid_convs_label0 =  list_convs_label0[split_2_label0:]
	train_convs_label1 =  list_convs_label1[:split_1_label1]
	test_convs_label1  =  list_convs_label1[split_1_label1:split_2_label1]
	valid_convs_label1 =  list_convs_label1[split_2_label1:]

	train_convs = train_convs_label0 + train_convs_label1
	test_convs = test_convs_label0 + test_convs_label1
	valid_convs = valid_convs_label0 + valid_convs_label1
	fw_train.writelines(train_convs)
	fw_test.writelines(test_convs)
	fw_valid.writelines(valid_convs)

	print('the number of convs with label 0 in train dataset is {}'.format(len(train_convs_label0)))
	print('the number of convs with label 1 in train dataset is {}'.format(len(train_convs_label1)))
	print('the number of convs with label 0 in test dataset is {}'.format(len(test_convs_label0)))
	print('the number of convs with label 1 in test dataset is {}'.format(len(test_convs_label1)))
	print('the number of convs with label 0 in valid dataset is {}'.format(len(valid_convs_label0)))
	print('the number of convs with label 1 in valid dataset is {}'.format(len(valid_convs_label1)))
	fw_train.close()
	fw_test.close()
	fw_valid.close()



def create_utts_files(data_dir, ifile, ofile):
	'''This method:
		1) extracts utterance (query-response) pairs from each conversation
			for convs with label 0 the extracted utterances are: utt1 utt2 utt3 utt4 utt5 utt6 ==> (utt1, utt2) , (utt3, utt4) , (utt5, utt6) , ...
			for convs with label 1 the extracted utterances are: utt1 utt2 utt3 utt4 utt5 utt6 ==> (utt1, utt2) , (utt2, utt3) , (utt3, utt4) , ...
		2) heuristically assigns each conversation's engagement score to all its utterances
	Params:
		data_dir: the directory that contains the ConvAI dataset
		ifile: the input file containing conversations
		ofile: a csv file including the extracted query and response pairs from each conversation with their assigned labels 
	'''

	fw = open(data_dir+'ConvAI_utts_'+ofile, 'w')
	fieldnames = ['id','query','reply','label']
	writer = csv.DictWriter(fw, fieldnames=fieldnames)
	writer.writeheader()

	fr = open(data_dir+ifile, 'r')
	lines = fr.readlines()
	num_utts_label0 = 0
	num_utts_label1 = 0
	num_convs_label0 = 0
	num_convs_label1 = 0
	list_utterance_pairs = []
	num_utts = []
	ind_utt = 0
	for line in lines:
		conv_txt = line.split('\t')[5]
		eng_Score = line.split('\t')[2]
		sp1_parts = conv_txt.split('</sp1> <sp2>')
		list_utts = []
		for p1 in sp1_parts:
			if '</sp2>' not in p1:
				utt = p1.split('<sp1>')[1].strip()
				list_utts.append(utt)

			else:
				sp2_parts = p1.split('</sp2> <sp1>')
				for p2 in sp2_parts:
					if '</sp1>' in p2:
						list_utts.append(p2.split('</sp1>')[0].strip())
					elif '</sp2>' in p2:
						list_utts.append(p2.split('</sp2>')[0].strip())

					else:
						list_utts.append(p2.strip())

		if eng_Score == '1':
			for ind in range(len(list_utts)-1):
				writer.writerow({'id':ind_utt,'query':list_utts[ind],'reply':list_utts[ind+1],'label':eng_Score})
				ind_utt+=1
		if eng_Score == '0':
			for ind in range(len(list_utts)-1):
				if ind %2 == 0:
					writer.writerow({'id':ind_utt,'query':list_utts[ind],'reply':list_utts[ind+1],'label':eng_Score})		
					ind_utt+=1
		if int(eng_Score)==0:
			num_utts = int(len(list_utts)/2)	
			for  k in range(num_utts):
				if eng_Score == '0':
					num_utts_label0 +=1
				elif eng_Score == '1':
					num_utts_label1 +=1


		if int(eng_Score)==1:
			num_utts = int(len(list_utts)-1)	
			for  k in range(num_utts):
				if eng_Score == '0':
					num_utts_label0 +=1
				elif eng_Score == '1':
					num_utts_label1 +=1
		if eng_Score == '0':
			num_convs_label0 +=1
		elif eng_Score == '1':
			num_convs_label1 +=1
		

	print('number of utts with label 0 is {}'.format(num_utts_label0))
	print('number of utts with label 1 is {}'.format(num_utts_label1))


if __name__=='__main__':
	data_dir = './../data/'
	preprocess_data(data_dir,'train_full.json','ConvAI_convs_orig')
	sub_AMT_set(data_dir,'50convs_AMT.txt','ConvAI_convs_orig','ConvAI_convs')
	split_train_test_valid(data_dir,'ConvAI_convs')
	create_utts_files(data_dir,'ConvAI_convs_train','train.csv')
	create_utts_files(data_dir,'ConvAI_convs_valid','valid.csv')
	create_utts_files(data_dir,'ConvAI_convs_test','test.csv')

	






