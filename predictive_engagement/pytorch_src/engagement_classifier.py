import random
import numpy as np
import torch 
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import torch.nn as nn
import os 
import csv

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

class Engagement_cls():
    '''This class classifies each query and response pairs as 0(not engaging) or 1 (engaging)
    '''
    def __init__(self, train_dir, batch_size, mlp_hidden_dim, num_epochs,\
                regularizer = 0.01, lr=1e-4, dropout = 0.1, optimizer="Adam",\
                ftrain_queries_embed=None, ftrain_replies_embed=None, fvalid_queries_embed=None, fvalid_replies_embed=None, ftest_queries_embed=None ,ftest_replies_embed=None):
        print('***************model parameters********************')
        print('mlp  layers {}'.format(mlp_hidden_dim))
        print('learning rate {}'.format(lr))
        print('drop out rate {}'.format(dropout))
        print('batch size {}'.format(batch_size))
        print('optimizer {}'.format(optimizer))
        print('regularizer {}'.format(regularizer))
        print('***************************************************')
        print(ftrain_queries_embed)
        print(ftrain_replies_embed)
        print(fvalid_queries_embed)
        print(fvalid_replies_embed)
        print(ftest_queries_embed)
        print(ftest_replies_embed)

        self.train_dir = train_dir
        self.batch_size = batch_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.lr = lr
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.optim = optimizer
        self.reg= regularizer
        self.ftrain_queries_embed = ftrain_queries_embed
        self.ftrain_replies_embed =ftrain_replies_embed
        self.fvalid_queries_embed= fvalid_queries_embed
        self.fvalid_replies_embed = fvalid_replies_embed
        self.ftest_queries_embed = ftest_queries_embed
        self.ftest_replies_embed = ftest_replies_embed

    def load_Bert_embeddings(self, data_dir, f_queries_embed, f_replies_embed):
        '''Load sentences Bert embeddings into dictionary 
        '''
        print('Loading Bert embeddings of sentences')
        queries_vectors = {}
        replies_vectors = {}
        fwq = open(data_dir+f_queries_embed, 'rb')
        dict_queries = pickle.load(fwq)
        for query, embeds in dict_queries.items():
            queries_vectors[query] = embeds[0]

        fwr = open(data_dir + f_replies_embed, 'rb')
        dict_replies = pickle.load(fwr)
        for reply, embeds in dict_replies.items():
            replies_vectors[reply] = embeds[0]
        print('number of loaded embeddings is {} {}'.format(len(queries_vectors), len(replies_vectors)))
        return queries_vectors, replies_vectors
       
    
    def prepare_data(self, data_dir, ftrain=None, fvalid=None, ftest=None):
        '''Load train/valid/test utterance pairs and get their embeddings
        '''
        self.data_dir = data_dir 
        if ftrain != None:
            csv_file = open(data_dir + ftrain)
            csv_reader_train = csv.reader(csv_file, delimiter=',')
            self.train_queries,self.train_replies,self.train_labels = [],[],[]
            next(csv_reader_train)
            for row in csv_reader_train:
                self.train_queries.append(row[1].split('\n')[0])
                self.train_replies.append(row[2].split('\n')[0])
                self.train_labels.append(int(row[3]))
            print('size of train_queries {}'.format(len(self.train_queries)))
            self.train_size = len(self.train_queries)
            self.train_queries_embeds, self.train_replies_embeds= self.load_Bert_embeddings(data_dir, self.ftrain_queries_embed, self.ftrain_replies_embed)

        if fvalid != None:
            csv_file = open(data_dir + fvalid)
            csv_reader_valid = csv.reader(csv_file, delimiter=',')
            self.valid_queries,self.valid_replies,self.valid_labels= [],[],[]
            next(csv_reader_valid)
            for row in csv_reader_valid:
                self.valid_queries.append(row[1].split('\n')[0])
                self.valid_replies.append(row[2].split('\n')[0])
                self.valid_labels.append(int(row[3]))
            print('size of valid_queries {}'.format(len(self.valid_queries)))
            self.valid_size = len(self.valid_queries)
            self.valid_queries_embeds, self.valid_replies_embeds= self.load_Bert_embeddings(data_dir, self.fvalid_queries_embed, self.fvalid_replies_embed)


        if ftest != None:
            print(self.ftest_queries_embed)
            print(self.ftest_replies_embed)
            csv_file = open(data_dir + ftest)
            csv_reader_test = csv.reader(csv_file, delimiter=',')

            self.test_queries,self.test_replies,self.test_labels = [],[],[]
            next(csv_reader_test)
            for row in csv_reader_test:
                self.test_queries.append(row[1].split('\n')[0])
                self.test_replies.append(row[2].split('\n')[0])
                self.test_labels.append(int(row[3]))
            self.test_size = len(self.test_queries)
            self.test_queries_embeds, self.test_replies_embeds= self.load_Bert_embeddings(data_dir, self.ftest_queries_embed, self.ftest_replies_embed)

        filename = self.train_dir + "log_train.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fw =open(filename, "a")
        self.fw.write('***************model parameters******************** \n')
        self.fw.write('mlp layers {} \n'.format(self.mlp_hidden_dim))
        self.fw.write('learning rate {}\n'.format(self.lr))
        self.fw.write('drop out rate {}\n'.format(self.dropout))
        self.fw.write('batch size {}\n'.format(self.batch_size))
        self.fw.write('optimizer {}\n'.format(self.optim))
        self.fw.write('regularizer {}'.format(self.reg))
        self.fw.write('***************************************************\n')
        
        
        
    def shuffle_data(self, type='train'):
        '''Shuffle queries/replies/engagement scores for train/valid/test sets 
        '''
        if type=='train':
            train_indexes = [i for i in range(self.train_size)] 
            random.shuffle(train_indexes)
            shuffled_queries  = []
            shuffled_replies = []
            shuffled_labels = []
            shuffled_replies_len = []
            shuffled_replies_num_diverse= []

            for i in train_indexes:
               shuffled_queries.append(self.train_queries[i])   
               shuffled_replies.append(self.train_replies[i])
               shuffled_labels.append(self.train_labels[i])

            self.train_queries = shuffled_queries
            self.train_replies = shuffled_replies
            self.train_labels = shuffled_labels

        elif type=='valid':
            valid_indexes = [i for i in range(self.valid_size)] 
            random.shuffle(valid_indexes)
            shuffled_queries  = []
            shuffled_replies = []
            shuffled_labels = []

            for i in valid_indexes:
               shuffled_queries.append(self.valid_queries[i])   
               shuffled_replies.append(self.valid_replies[i])
               shuffled_labels.append(self.valid_labels[i])

            self.valid_queries = shuffled_queries
            self.valid_replies = shuffled_replies
            self.valid_labels = shuffled_labels

        elif type=='test':
            test_indexes = [i for i in range(self.test_size)] 
            random.shuffle(test_indexes)
            shuffled_queries  = []
            shuffled_replies = []
            shuffled_labels = []

            for i in test_indexes:
               shuffled_queries.append(self.test_queries[i])   
               shuffled_replies.append(self.test_replies[i])
               shuffled_labels.append(self.test_labels[i])

            self.test_queries = shuffled_queries
            self.test_replies = shuffled_replies
            self.test_labels = shuffled_labels

        
    def train(self, early_stop=50, finetune=False):
    
        model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        max_auc = 0
        no_improve_epoch = 0
        no_improve_in_previous_epoch = False

        if finetune==False:
            model_name ='best_model' 
        if finetune==True:
            model_name ='best_model_finetuned'
            #load pretrained model
            model.load_state_dict(torch.load(self.train_dir + 'best_model.pt'))
            info = torch.load(self.train_dir + 'best_model.info')
            print('the parameters of the best trained model is ')
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print (name, param.data, param.shape)
        print(self.lr)
        if self.optim=='SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.reg)
        if self.optim=='Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.reg)
        if self.optim=='RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.lr, weight_decay=self.reg)

        plot_train_auc = []
        plot_valid_auc = []
        plot_valid_loss = []
        plot_train_loss = []
        plot_ep = []
        step=0
        #Shuffle valid data once since original file first has all the utterances with engagement score=0 and then all the utterances with engagement score=1
        self.shuffle_data('valid') 

        for e in range(self.num_epochs):
            print('***********************************************')
            print(e)
            if no_improve_in_previous_epoch:
                no_improve_epoch += 1
                if no_improve_epoch >= early_stop:
                    break
            else:
                no_improve_epoch = 0
            no_improve_in_previous_epoch = True

            train_loss = []
            train_auc = []
            nonzero_total= 0
            list_preds = torch.tensor([self.train_size])
            list_grtuth = torch.tensor([self.train_size])
            if torch.cuda.is_available():
                list_preds = list_preds.cuda()
                list_grtuth = list_grtuth.cuda()
            self.shuffle_data('train')

            for stidx in range(0, self.train_size, self.batch_size):
                step+=1
                model.train()
                model.zero_grad()
                x_q = self.train_queries[stidx:stidx + self.batch_size]
                x_r = self.train_replies[stidx:stidx + self.batch_size]
                y = torch.tensor(self.train_labels[stidx:stidx + self.batch_size]).long()
                
                if torch.cuda.is_available():
                    y = y.cuda()
                nonzero = torch.nonzero(y).size(0)
                nonzero_total +=nonzero
                model_output = model(x_q, x_r, self.train_queries_embeds, self.train_replies_embeds)

                pred_eval = torch.argmax(model_output, 1)
                

                list_preds = torch.cat((list_preds, pred_eval), dim=0)
                list_grtuth = torch.cat((list_grtuth, y), dim=0)
                
                #calculate weights for each class
                try:
                    weight = torch.tensor([y.shape[0]/(2*(y.shape[0]- nonzero)), y.shape[0]/(2*nonzero)])
                    if torch.cuda.is_available():
                        weight = weight.cuda()
                except:
                    weight = None
                #weighted loss function due bacuase of imbalanced data 
                loss_function = nn.CrossEntropyLoss(weight)
                loss = loss_function(model_output, y)
                train_loss.append(loss.data) 
                loss.backward()
                optimizer.step()
            print('number of nonzero in train is {}'.format(nonzero_total))
            #calculate the evaluation metric and loss value for train data
            train_auc = roc_auc_score(list_grtuth[1:].detach().cpu().numpy(), list_preds[1:].detach().cpu().numpy())
            train_loss = torch.mean(torch.stack(train_loss))
            # train_loss = np.mean(train_loss)
        
            #evaluate trained model on valid data
            val_loss = []
            val_auc = []
            nonzero_total = 0
            list_preds_v = torch.tensor([self.valid_size])
            list_grtuth_v = torch.tensor([self.valid_size])
            if torch.cuda.is_available():
                list_preds_v = list_preds_v.cuda()
                list_grtuth_v = list_grtuth_v.cuda()
            for stidx in range(0, self.valid_size, self.batch_size):
                model.eval()
                val_x_q = self.valid_queries[stidx:stidx + self.batch_size]
                val_x_r = self.valid_replies[stidx:stidx + self.batch_size]
                val_y = torch.tensor(self.valid_labels[stidx:stidx + self.batch_size]).long()
                
                if torch.cuda.is_available():
                    val_y = val_y.cuda()
                nonzero = torch.nonzero(val_y).size(0)
                nonzero_total +=nonzero
                model_output = model(val_x_q, val_x_r, self.valid_queries_embeds, self.valid_replies_embeds)
                val_pred = torch.argmax(model_output, 1)
                list_preds_v = torch.cat((list_preds_v, val_pred), dim=0)
                list_grtuth_v = torch.cat((list_grtuth_v, val_y), dim=0)

                weight = torch.tensor([val_y.shape[0]/(2*(val_y.shape[0]- nonzero)), val_y.shape[0]/(2*nonzero)])
                if torch.cuda.is_available():
                    weight = weight.cuda()
                loss_function = nn.CrossEntropyLoss(weight)
                v_loss = loss_function(model_output, val_y)

                val_loss.append(v_loss.data)

            val_auc = roc_auc_score(list_grtuth_v[1:].detach().cpu().numpy(), list_preds_v[1:].detach().cpu().numpy())
            # val_loss = np.mean(val_loss)
            val_loss = torch.mean(torch.stack(val_loss))

            print('number of nonzero in valid is {}'.format(nonzero_total))
            
            st_improv = ''
            if val_auc > max_auc:
                st_improv = '*'
                torch.save({'step': step, 'epoch': e, 'train_loss': train_loss, 'train_auc': train_auc, 'val_loss': val_loss, 'val_auc': val_auc }, self.train_dir+model_name+'.info')
                torch.save(model.state_dict(), self.train_dir+model_name+'.pt')
                max_auc = val_auc
                no_improve_in_previous_epoch = False
                
            print('epcoh {:02} - train_loss {:.4f}  - train_auc {:.4f} val_loss {:.4f}  - val_auc {:.4f}{}'.format(
                        e, train_loss, train_auc, val_loss, val_auc, st_improv))
            self.fw.write('epcoh {:02} - train_loss {:.4f} - train_auc {:.4f} val_loss {:.4f}  - val_auc {:.4f}{} \n'.format(
                            e, train_loss, train_auc, val_loss, val_auc, st_improv))

            plot_train_auc.append(train_auc)
            plot_valid_auc.append(val_auc)
            plot_train_loss.append(train_loss)
            plot_valid_loss.append(val_loss)
            plot_ep.append(e)
        
        print('#############################################')
        model.load_state_dict(torch.load(self.train_dir +  model_name+'.pt'))
        info = torch.load(self.train_dir + model_name+'.info')

        print('the parameters of the best trained model is ')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data, param.shape)


        print('Done!')
        plt.figure(0)          
        l1 = plt.plot(plot_ep,plot_train_auc,'-r', label='Train auc')
        l2 = plt.plot(plot_ep,plot_valid_auc,'-b', label='Valid auc')
        plt.legend(loc='upper left')
        plt.xlabel("train and valid acc for model")
        plt.savefig(self.train_dir + 'model_auc.jpg')
            
        plt.figure(1)  
        l1 = plt.plot(plot_ep,plot_train_loss,'-r', label='Train loss')
        l2 = plt.plot(plot_ep,plot_valid_loss,'-b', label='Valid loss')
        plt.legend(loc='upper left')
        plt.xlabel("train and valid loss for model")
        plt.savefig(self.train_dir + 'model_loss.jpg')

        

         
    def test(self, fname):
        '''Test the trained model on test set
        '''
        if not os.path.isfile(self.train_dir+'best_model.pt'):
            print('There is not any trained model to be tested!\nPlease first try to train the model.')
            return 

        model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(self.train_dir+'best_model.pt'))
        info = torch.load(self.train_dir + 'best_model.info')
        model.eval()
        print('begining of test')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data, param.shape)

        self.shuffle_data('test') 
        test_loss = []
        test_auc = []
        nonzero_total= 0
        step = 0
        list_preds_t = torch.tensor([self.test_size])
        list_grtuth_t = torch.tensor([self.test_size])
        if torch.cuda.is_available():
                list_preds_t = list_preds_t.cuda()
                list_grtuth_t = list_grtuth_t.cuda()
        for stidx in range(0, self.test_size, self.batch_size):
            step+=1
            x_q = self.test_queries[stidx:stidx + self.batch_size]
            x_r = self.test_replies[stidx:stidx + self.batch_size]
            y = torch.tensor(self.test_labels[stidx:stidx + self.batch_size]).long()
            if torch.cuda.is_available():
                y = y.cuda()
            nonzero = torch.nonzero(y).size(0)
            nonzero_total +=nonzero
            model_output = model(x_q, x_r,  self.test_queries_embeds, self.test_replies_embeds)
            pred_eval = torch.argmax(model_output, 1)
            list_preds_t = torch.cat((list_preds_t, pred_eval), dim=0)
            list_grtuth_t = torch.cat((list_grtuth_t, y), dim=0)
            print('batch {} has {} nonzero points and {} zero points overall {} points '.format(step, nonzero, y.shape[0]- nonzero, y.shape[0]))
            weight = torch.tensor([y.shape[0]/(2*(y.shape[0]- nonzero)), y.shape[0]/(2*nonzero)])
            if torch.cuda.is_available():
                weight = weight.cuda()
            loss_function = nn.CrossEntropyLoss(weight)
            loss = loss_function(model_output, y)
            test_loss.append(loss.data)  
        print('number of nonzero in test is {}'.format(nonzero_total))

        test_auc = roc_auc_score(list_grtuth_t[1:].detach().cpu().numpy(), list_preds_t[1:].detach().cpu().numpy())
        print(classification_report(list_grtuth_t[1:].detach().cpu().numpy(), list_preds_t[1:].detach().cpu().numpy()))
        # test_loss = np.mean(test_loss)
        test_loss = torch.mean(torch.stack(test_loss))

        print('Test set: test_loss: {} -- test_auc: {}'.format(test_loss, test_auc))


    def generate_eng_score(self, fname_ground_truth, ofile):
        '''for all pairs of queries and replies predicts engagement scores
        Params:
            fname_ground_truth: file includes the queries and their ground-truth replies
            foname: file includes the queries, ground truth replies, generated replies (from self.test_replies) and engagement_score of queries and generated replies with following format:
                query===groundtruth_reply===generated_reply===engagement_score of query and generated_reply

        '''

        if not os.path.isfile(self.train_dir+'best_model_finetuned.pt'):
            print('There is not any finetuned model on DD dataset to be used!\nPlease first try to finetune trained model.')
            return
        model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(self.train_dir +  'best_model_finetuned.pt'))
        info = torch.load(self.train_dir + 'best_model_finetuned.info')
        model.eval()

        fw_pred_labels = open(self.data_dir + ofile, 'w')
        #fr_groundtruth_replies = open(self.data_dir + fname_ground_truth, 'r')
        #groundtruth_replies =fr_groundtruth_replies.readlines() 

        print('begining of prediction')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print (name, param.data, param.shape)
        for stidx in range(0, self.test_size, self.batch_size):
            x_q = self.test_queries[stidx:stidx + self.batch_size]
            x_r = self.test_replies[stidx:stidx + self.batch_size]
            #x_groundtruth_r = groundtruth_replies[stidx:stidx + self.batch_size]
            model_output = model(x_q, x_r, self.test_queries_embeds, self.test_replies_embeds)
            pred_eng = torch.nn.functional.softmax(model_output, dim=1)
            for ind in range(len(x_q)):
                #fw_pred_labels.write(x_q[ind]+'==='+x_groundtruth_r[ind].split('\n')[0]+'==='+x_r[ind]+'==='+str(pred_eng[ind][1].item())+'\n')
                fw_pred_labels.write(x_q[ind]+'==='+x_r[ind]+'==='+str(pred_eng[ind][1].item())+'\n')
            
        print('The engagingness score for specified replies has been predicted!')


    def get_eng_score(self, query, q_embed, reply, r_embed, model):
        '''for a pair of query and reply predicts engagement scores
        Params:
            query: input query
            q_embed: embeddings of query
            reply: input reply
            r_embed: embeddings of reply
           
        '''
        if not os.path.isfile(self.train_dir+'best_model_finetuned.pt'):
            print('There is not any finetuned model on DD dataset to be used!\nPlease first try to finetune trained model.')
            return
            
        model = BiLSTM(mlp_hidden_dim=self.mlp_hidden_dim, dropout=self.dropout)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(self.train_dir +  'best_model_finetuned.pt'))
        info = torch.load(self.train_dir + 'best_model_finetuned.info')
        model.eval()

        model_output = model(query, reply, q_embed, r_embed)
        pred_eng = torch.nn.functional.softmax(model_output, dim=1)
        return pred_eng

 
 


class  BiLSTM(nn.Module):
    '''The engagement classification model is a three layer mlp classifier with having tanh as activation functions which takes the embeddings of query and reply as input and pass their average into the mlp classifier
    '''
    def __init__(self, mlp_hidden_dim=[128], dropout=0.2):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        num_classes=2
        self.mlp_hidden_0 = nn.Linear(768, mlp_hidden_dim[0], bias=True)
        self.mlp_hidden_1 = nn.Linear(mlp_hidden_dim[0], mlp_hidden_dim[1], bias=True)
        self.mlp_hidden_2 = nn.Linear(mlp_hidden_dim[1], mlp_hidden_dim[2], bias=True)
        self.mlp_out = nn.Linear(mlp_hidden_dim[2], num_classes, bias=True)


    def forward(self, queries_input, replies_input,  queries_embeds, replies_embeds):

        for ind, q in enumerate(queries_input):
            if q not in queries_embeds.keys():
                print('the query {} embedding has not been found in the embedding file'.format(q))
        X_q = torch.tensor([queries_embeds[q] for q in queries_input])

        for ind, r in enumerate(replies_input):
            if r not in replies_embeds.keys():
                print('the reply {} embedding has not been found in the embedding file'.format(r))
        X_r = torch.tensor([replies_embeds[r] for r in replies_input])
        
        if torch.cuda.is_available():
            X_q, X_r = X_q.cuda(), X_r.cuda()
        mlp_input=X_q.add(X_r)
        mlp_input = torch.div(mlp_input,2)

        mlp_h_0 = torch.tanh(self.mlp_hidden_0(mlp_input))
        mlp_h_0= self.dropout(mlp_h_0)
    
        mlp_h_1 = torch.tanh(self.mlp_hidden_1(mlp_h_0))
        mlp_h_1= self.dropout(mlp_h_1)

        mlp_h_2 = torch.tanh(self.mlp_hidden_2(mlp_h_1))
        mlp_h_2= self.dropout(mlp_h_2)

        mlp_out= self.mlp_out(mlp_h_2)
        return mlp_out
