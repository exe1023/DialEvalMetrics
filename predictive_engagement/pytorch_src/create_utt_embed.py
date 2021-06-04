
import argparse
from bert_serving.client import BertClient
import csv
import os
import pickle

#In order to create utterance embeddings, you need to first start BertServer (follow https://github.com/hanxiao/bert-as-service) with following command:
#bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=128 -pooling_strategy=REDUCE_MEAN
#model_dir is the directory that pretrained Bert model has been downloaded

def make_Bert_embeddings(data_dir, fname, f_queries_embed, f_replies_embed, type):
	'''Create embedding file for all queries and replies in the given files
	Param:
		data_dir: the directory of data
		fname: name of the input file containing queries, replies, engagement_score
		f_queries_embed: name of the output file containing the queries bert embeddings
		f_replies_embed: name of the output file containing the replies bert embeddings
		type: indicate train/valid/test set
	'''

	csv_file = open(data_dir + fname)
	csv_reader = csv.reader(csv_file, delimiter=',')

	foutput_q = os.path.join(data_dir + f_queries_embed)
	foutput_r = os.path.join(data_dir + f_replies_embed)
   
	queries,replies = [],[]
	next(csv_reader)
	for row in csv_reader:
		queries.append(row[1].split('\n')[0])
		replies.append(row[2].split('\n')[0])
      
	if os.path.exists(foutput_q) and os.path.exists(foutput_r) :
		print('Bert embedding files for utterances exist!')
		return

	else:
		print("Bert embedding files for utterances do not exist")
		queries_vectors = {}
		replies_vectors = {}
		bc = BertClient()
		has_empty = False
		fwq = open(foutput_q, 'wb')
		for idx, q in enumerate(queries):
			print(str(idx)+'query {}'.format(type))
			if q not in queries_vectors.keys() and q !='':
				queries_vectors[q] = bc.encode([q])
			if q not in queries_vectors.keys() and q =='':
				queries_vectors[q] = bc.encode(['[PAD]'])
				has_empty=True
		if has_empty == False:
			queries_vectors[''] = bc.encode(['[PAD]'])
		pickle.dump(queries_vectors, fwq)

		fwr = open(foutput_r, 'wb')
		has_empty = False
		for idx, r in enumerate(replies):
			print(str(idx)+'reply {}'.format(type))
			if r not in replies_vectors.keys() and r !='':
				replies_vectors[r] = bc.encode([r])
			if r not in replies_vectors.keys() and r =='':
				replies_vectors[r] = bc.encode(['[PAD]'])
				has_empty = True
		if has_empty == False:
			replies_vectors[''] = bc.encode(['[PAD]'])
		pickle.dump(replies_vectors, fwr)



def load_Bert_embeddings(data_dir, f_queries_embed, f_replies_embed):
    '''Load embeddings of queries and replies 
    Param:
		data_dir: the directory of data
		f_queries_embed: name of the input file containing the queries bert embeddings
		f_replies_embed: name of the input file containing the replies bert embeddings
    '''

    print('Loading Bert embeddings of sentences')
    queries_vectors = {}
    replies_vectors = {}

    print('query embedding')
    fwq = open(data_dir  + f_queries_embed, 'rb')
    dict_queries = pickle.load(fwq)
    for query, embeds in dict_queries.items():
    	queries_vectors[query] = embeds[0]
    print('len of embeddings is '+str(len(queries_vectors)))

    print('reply embedding')
    fwr = open(data_dir + f_replies_embed, 'rb')
    dict_replies = pickle.load(fwr)
    for reply, embeds in dict_replies.items():
    	replies_vectors[reply] = embeds[0]
    print('len of embeddings is '+str(len(replies_vectors)))
  
    return queries_vectors, replies_vectors
   

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Parameters for engagement classification')
	parser.add_argument('--data', type=str)
	args = parser.parse_args()	
	data_dir = './../data/'
	pooling = 'mean'
	ifname= 'ConvAI_utts'
	dd_ifname = 'DD_finetune'
	ofname = ''
	#make_Bert_embeddings(data_dir, ifname+'_train.csv', ifname+'_train_queries_embed_'+pooling, ifname+'_train_replies_embed_'+pooling, 'train')
	#make_Bert_embeddings(data_dir, ifname+'_valid.csv', ifname+'_valid_queries_embed_'+pooling, ifname+'_valid_replies_embed_'+pooling, 'valid')
	#make_Bert_embeddings(data_dir, ifname+'_test.csv', ifname+'_test_queries_embed_'+pooling, ifname+'_test_replies_embed_'+pooling, 'test')

	#make_Bert_embeddings(data_dir, 'humanAMT_engscores_utt.csv', 'humanAMT_queries_embed_'+pooling, 'humanAMT_replies_embed_'+pooling, 'testAMT')
	#make_Bert_embeddings(data_dir, dd_ifname+'_train.csv', dd_ifname+'_queries_train_embed_'+pooling, dd_ifname+'_replies_train_embed_'+pooling, 'train')
	#make_Bert_embeddings(data_dir, dd_ifname+'_valid.csv', dd_ifname+'_queries_valid_embed_'+pooling, dd_ifname+'_replies_valid_embed_'+pooling, 'valid')
	#make_Bert_embeddings(data_dir, 'DD_queries_generated_replies.csv', 'DD_queries_embed_'+pooling, 'DD_generated_replies_embed_'+pooling, 'test')

	
	
	make_Bert_embeddings(data_dir, args.data, f'{args.data}_queries_embed_'+pooling, f'{args.data}_replies_embed_'+pooling, 'test')