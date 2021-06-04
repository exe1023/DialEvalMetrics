import argparse
from engagement_classifier import Engagement_cls


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Parameters for engagement classification')
	parser.add_argument('--mlp_hidden_dim', type=int, default=[64, 32, 8],
                    help='number of hidden units in mlp')
	parser.add_argument('--epochs', type=int, default=400,
                    help='number of training epochs')
	parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
	parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
	parser.add_argument('--dropout', type=float, default=0.8,
                    help='dropout rate')
	parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling type for getting sentence embeddings from words embeddings')
	parser.add_argument('--optimizer', type=str, default='Adam',
                    help='optimizer for training model')
	parser.add_argument('--reg', type=float, default=0.001,
                    help='l2 regularizer for training model')
	parser.add_argument('--mode', type=str,
                    help="""train: to train the model 
                    	  test: to test the model on ConvAI test set
                    	  testAMT: to test the model on 297 utterances (50 randomly selected dialogs from ConvAI) annotated by Amazon turkers
                    	  finetune: to finetune the trained model on 300 pairs selected from Daily Dialogue dataset annotated by Amazon turkers 
                    	  predict: to predict engagement scores for query and generated replies (based on attention-based seq-to-seq model) of Daily Dilaogue dataset""")
	parser.add_argument('--data', type=str)
	args = parser.parse_args()

	data_dir = './../data/'
	train_dir = './../model/'


	
	if args.mode == "train":
		print('ConvAI_utts_train_queries_embed_'+args.pooling)
		ftrain,fvalid = ['ConvAI_utts_train.csv','ConvAI_utts_valid.csv']
		#files including queries and replies embeddings in train/valid/test sets 
		ftrain_queries_embed = 'ConvAI_utts_train_queries_embed_'+args.pooling 
		ftrain_replies_embed = 'ConvAI_utts_train_replies_embed_'+args.pooling
		fvalid_queries_embed = 'ConvAI_utts_valid_queries_embed_'+args.pooling
		fvalid_replies_embed = 'ConvAI_utts_valid_replies_embed_'+args.pooling
		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs,\
			                    args.reg, args.lr, args.dropout, args.optimizer,\
			                    ftrain_queries_embed = ftrain_queries_embed, ftrain_replies_embed=ftrain_replies_embed, fvalid_queries_embed= fvalid_queries_embed, fvalid_replies_embed=fvalid_replies_embed)
		eng_cls.prepare_data(data_dir, ftrain, fvalid)
		eng_cls.train()
	
	if args.mode == "test":
		ftest = 'ConvAI_utts_test.csv'
		ftest_queries_embed = 'ConvAI_utts_test_queries_embed_'+args.pooling
		ftest_replies_embed = 'ConvAI_utts_test_replies_embed_'+args.pooling
		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs,\
								args.reg, args.lr, args.dropout, args.optimizer,\
			                    ftest_queries_embed=ftest_queries_embed , ftest_replies_embed=ftest_replies_embed)
		eng_cls.prepare_data(data_dir, ftest=ftest)
		eng_cls.test(ftest)

	if args.mode == "testAMT":
		ftest = 'humanAMT_engscores_utt.csv'
		#humanAMT_engscores_utt.csv: 297 utterances that their engagement scores are annotated by AMT workers
		fhuman_test_queries_embed = 'humanAMT_queries_embed_'+args.pooling #query embeddings of 50 randomly selected conversations (297 utterances) annotated by Amazon turkers
		fhuman_test_replies_embed = 'humanAMT_replies_embed_'+args.pooling #reply embeddings of 50 randomly selected conversations (297 utterances) annotated by Amazon turkers
		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs,\
								args.reg, args.lr, args.dropout, args.optimizer,\
			                    ftest_queries_embed=fhuman_test_queries_embed ,ftest_replies_embed=fhuman_test_replies_embed)
		eng_cls.prepare_data(data_dir,ftest=ftest)
		eng_cls.test(ftest)

	if args.mode == "finetune":
	    #DD_finetune_train.csv and DD_finetune_valid.csv train/valid sets from 300 pairs of Daily Dialogue dataset annotated by Amazon turkers

		ftrain, fvalid, ftest = ['DD_finetune_train.csv','DD_finetune_valid.csv', '']
		#ftrain_queries_embed = 'DD_finetune_queries_train_embed_'+args.pooling
		#ftrain_replies_embed = 'DD_finetune_replies_train_embed_'+args.pooling
		ftrain_queries_embed = 'DD_finetune_train.csv_queries_embed_mean'
		ftrain_replies_embed = 'DD_finetune_train.csv_replies_embed_mean'

		#fvalid_queries_embed = 'DD_finetune_queries_valid_embed_'+args.pooling
		#fvalid_replies_embed = 'DD_finetune_replies_valid_embed_'+args.pooling
		fvalid_queries_embed = 'DD_finetune_valid.csv_queries_embed_mean'
		fvalid_replies_embed = 'DD_finetune_valid.csv_replies_embed_mean'
		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs,\
								args.reg, args.lr, args.dropout, args.optimizer,\
		                        ftrain_queries_embed=ftrain_queries_embed, ftrain_replies_embed=ftrain_replies_embed, fvalid_queries_embed=fvalid_queries_embed, fvalid_replies_embed=fvalid_replies_embed)
		eng_cls.prepare_data(data_dir, ftrain=ftrain, fvalid=fvalid)
		eng_cls.train(finetune=True)

	if args.mode =="predict":
		#The file including queries and generated replies
		ftest = 'DD_queries_generated_replies.csv'
		#ftest_queries_embed = 'DD_queries_embed_'+args.pooling
		ftest_queries_embed = 'DD_queries_generated_replies.csv_queries_embed_mean'
		#ftest_replies_embed = 'DD_generated_replies_embed_'+args.pooling
		ftest_replies_embed = 'DD_queries_generated_replies.csv_replies_embed_mean'

		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs, \
								args.reg, args.lr, args.dropout, args.optimizer,\
			                    ftest_queries_embed=ftest_queries_embed ,ftest_replies_embed=ftest_replies_embed)
		eng_cls.prepare_data(data_dir, ftest=ftest)
		eng_cls.generate_eng_score('DD_replies.txt','DD_queries_genreplies_eng_{}.txt'.format(args.pooling))


		#The file including queries and Human-written(ground-truth) replies
		ftest = 'DD_queries_groundtruth_replies.csv'
		#ftest_queries_embed = 'DD_queries_embed_'+args.pooling
		ftest_queries_embed = 'DD_queries_groundtruth_replies.csv_queries_embed_mean'

		#ftest_replies_embed = 'DD_groundtruth_replies_embed_'+args.pooling
		ftest_replies_embed = 'DD_queries_groundtruth_replies.csv_replies_embed_mean'
		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs, \
								args.reg, args.lr, args.dropout, args.optimizer,\
			                    ftest_queries_embed=ftest_queries_embed ,ftest_replies_embed=ftest_replies_embed)
		eng_cls.prepare_data(data_dir, ftest=ftest)
		eng_cls.generate_eng_score('DD_replies.txt','DD_queries_groundtruth_eng_{}.txt'.format(args.pooling))
	
	if args.mode == 'infer':
		#The file including queries and generated replies
		ftest = args.data
		ftest_queries_embed = f'{args.data}_queries_embed_'+args.pooling
		ftest_replies_embed = f'{args.data}_replies_embed_'+args.pooling
		eng_cls = Engagement_cls(train_dir, args.batch_size, args.mlp_hidden_dim, args.epochs, \
								args.reg, args.lr, args.dropout, args.optimizer,\
			                    ftest_queries_embed=ftest_queries_embed ,ftest_replies_embed=ftest_replies_embed)
		eng_cls.prepare_data(data_dir, ftest=ftest)
		eng_cls.generate_eng_score('', f'{args.data}_score.txt')
	