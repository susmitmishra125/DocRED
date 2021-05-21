import sys
import math
import nltk
import numpy as np
import os
import json
import copy
import random
import time
import datetime
os.chdir("/home/mtech1/19CS60R28/susmit/DocRed_hongwang600/DocRed")

from nltk.tokenize import WordPunctTokenizer
from pytorch_transformers import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from models.bert import Bert

os.chdir("/home/mtech1/19CS60R28/susmit/DocRed_hongwang600")
RANDOM_SEED = 42
in_path='data'
out_path='prepro_data'
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
PRE_TRAINED_MODEL_NAME='bert-base-uncased'
model_name='bert'
BATCH_SIZE=4
EPOCH = 10

n_gpu = torch.cuda.device_count()
if not os.path.exists(out_path):
    os.mkdir(out_path)

MAX_LEN=512
SEP='[SEP]'
MASK = '[MASK]'
CLS = "[CLS]"
bert = Bert(BertModel, PRE_TRAINED_MODEL_NAME)

train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

def logging(*msg, print_=True, log_=True):
    for i in range(0,len(msg)):
        if(i==len(msg)-1):
            end='\n'
        else:
            end=' '
        if print_:
            print(msg[i],end=end)
        if log_:
            with open(os.path.join(os.path.join("log", model_name+'.txt')), 'a+') as f_log:
                f_log.write(str(msg[i])+end)
                f_log.close()
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seeds(RANDOM_SEED)

def load_ckp(checkpoint_fpath, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_fpath)
    except OSError as e:
        return -1,-1,-1.0,0,model,optimizer 
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['best_epoch_idx'], checkpoint['best_epoch_seed'], checkpoint['best_dev_acc'], checkpoint['epoch'], model, optimizer


def preprocess(data_file_name, max_length = 512, is_training = True, suffix=''):
    ori_data=json.load(open(data_file_name))[0:100]
    max_sent_count=0#maximum number of sentences in a doc across the dataset
    list_sent_ids=[]
    list_attention=[]#this stores attention of docs
    list_sent_mask=[]#this will be used in the batch multliplication for getting the embeddings of each sentence
    # (len(list_sent_ids),max_sent_count,max_length)
    
    labels=[]
    i=0
    for doc in ori_data:
        i=i+1
        sys.stdout.write("\r%d/%d docs"%(i,len(ori_data)))
        # this dict is used to take care of multiple relations with same head and tail
        head_tail_index={}
        max_sent_count=max(max_sent_count,len(doc['sents']))
        for label in doc['labels']:
            idx_list=[]
            head=doc['vertexSet'][label['h']]
            tail=doc['vertexSet'][label['t']]
            if (label['h'],label['t']) in head_tail_index:
                labels[head_tail_index[(label['h'],label['t'])]]+=label['evidence']
                continue
            else:
                head_tail_index[(label['h'],label['t'])]=len(list_sent_ids)
            for entity in head:
                if (entity['sent_id'],entity['pos'][0],'[unused0]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][0],'[unused0]'))
                if (entity['sent_id'],entity['pos'][1]+1,'[unused1]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][1]+1,'[unused1]'))
            for entity in tail:
                if (entity['sent_id'],entity['pos'][0],'[unused2]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][0],'[unused2]'))
                if (entity['sent_id'],entity['pos'][1]+1,'[unused3]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][1]+1,'[unused3]'))
            idx_list.sort(key=lambda tup:(tup[0],tup[1]),reverse=True)
            temp_doc=copy.deepcopy(doc)
            for loc in idx_list:
                temp_doc['sents'][loc[0]].insert(loc[1],loc[2])

            sent_combine=[]
            for sent in temp_doc['sents']:
                sent_combine=sent_combine+sent
            sent_ids,sent_attention_mask,sent_start_ids=bert.subword_tokenize_to_ids(sent_combine)
            list_sent_ids.append(sent_ids[0])
            list_attention.append(sent_attention_mask[0])
            labels.append(label['evidence'])
            
            
            sent_mask=[]
            l=1# we start from index 1 because we skip CLS token
            for sent in temp_doc['sents']:
                sent_mask.append([0]*max_length)
                j=l
                while(j<min(max_length-2,l+len(sent))):
                    sent_mask[-1][j]=1
                    j+=1
                l+=len(sent)
                if(l>=max_length-2):
                    break
            list_sent_mask.append(sent_mask)
            
    logging('')
    evi_labels = np.zeros((len(labels),max_sent_count),dtype = np.int64)
    for i in range(len(labels)):
        evi_labels[i][labels[i]]=1 #if evidence present then 1
    print("max_sent_cout",max_sent_count)
    for i in range(len(list_sent_mask)):
        # the label for pad sentence is 2
        evi_labels[i][len(list_sent_mask[i]):max_sent_count]=2
        # to pad sentences with arrays of 1s
        list_sent_mask[i]=list_sent_mask[i]+[[1]*max_length]*(max_sent_count-len(list_sent_mask[i]))
    list_sent_ids=np.asarray(list_sent_ids,dtype=np.int64)
    list_attention=np.asarray(list_attention,dtype=np.int64)
    list_sent_mask=np.asarray(list_sent_mask,dtype=np.int64)
    
    logging("Started saving")
    
    logging("Number of instances: {}".format(list_sent_ids.shape[0]))
    np.save(os.path.join(out_path,suffix+'_sent_ids.npy'),list_sent_ids)
    np.save(os.path.join(out_path,suffix+'_sent_attention.npy'),list_attention)
    np.save(os.path.join(out_path,suffix+'_sent_mask.npy'),list_sent_mask)
    np.save(os.path.join(out_path,suffix+'_evidence_labels.npy'),evi_labels)
    logging("completed saving\n")

preprocess(train_annotated_file_name, max_length = 512, is_training = False, suffix='train')
preprocess(dev_file_name, max_length = 512, is_training = False, suffix='dev')


class Docred_dataset(Dataset):
    def __init__(self,sent_ids,sent_attention,sent_mask,evi_target,max_len=512):
        self.sent_ids=torch.from_numpy(sent_ids)
        self.sent_attention=torch.from_numpy(sent_attention)
        self.sent_mask=torch.from_numpy(sent_mask)
        self.evi_target=torch.from_numpy(evi_target)
        self.no_samples=evi_target.shape[0]
    def __len__(self):
        return evi_target.shape[0]
    def __getitem__(self,index):
        return {
            'sent_ids':self.sent_ids[index],
            'sent_attention':self.sent_attention[index],
            'sent_mask':self.sent_mask[index],
            'targets':self.evi_target[index]
        }

sent_ids=np.load(os.path.join(out_path,'train'+'_sent_ids.npy'))
sent_attention=np.load(os.path.join(out_path,'train'+'_sent_attention.npy'))
sent_mask=np.load(os.path.join(out_path,'train'+'_sent_mask.npy'))
evi_target=np.load(os.path.join(out_path,'train'+'_evidence_labels.npy'))

dataset=Docred_dataset(sent_ids=sent_ids,sent_attention=sent_attention,sent_mask=sent_mask,evi_target=evi_target,max_len=MAX_LEN)
dataloader=DataLoader(dataset=dataset, batch_size=BATCH_SIZE,num_workers=2)

class EvidenceClassifier(nn.Module):
    def __init__(self):
        super(EvidenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dense = nn.Linear(self.bert.config.hidden_size, 3)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input_ids, attention_mask,sent_mask,is_training=False):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        sent_mask=sent_mask.float()
        output=torch.bmm(sent_mask,last_hidden_state)/sent_mask.sum(axis=2)[...,None]
        logits = self.dense(output)
        if is_training:
            return self.logsoftmax(logits)
        else:
            return self.softmax(logits)

def train(input_ids,sent_attention,sent_mask,evi_target):
    train_size=input_ids.shape[0]
    batch_size=BATCH_SIZE
    batch_count=int(math.ceil(train_size)/batch_size)
    model=EvidenceClassifier()
    logging(model)
    
    if torch.cuda.is_available():
        model.cuda
        model = torch.nn.DataParallel(model)
    criterion = nn.NLLLoss(reduction='mean',ignore_index=2)
    optimizer = AdamW(model.parameters(),lr=1e-05,correct_bias=False)
    
    logging(optimizer)
    
    best_dev_acc = -1
    best_epoch_idx = -1
    best_epoch_seed = -1
    start_epoch = 0
    ckp_path=os.path.join('checkpoint',model_name+'_checkpoint.pt')
    best_epoch_idx,best_epoch_seed,best_dev_acc,start_epoch,model,optimizer=load_ckp(ckp_path, model, optimizer)
    
    train_dataset=Docred_dataset(sent_ids=input_ids,sent_attention=sent_attention,sent_mask=sent_mask,evi_target=evi_target,max_len=MAX_LEN)
    train_dataloader=DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,num_workers=2)
    for epoch_idx in range(start_epoch, EPOCH):
        model.train()
        model.zero_grad()
        logging('Epoch:', epoch_idx + 1)
        cur_seed = RANDOM_SEED + epoch_idx + 1
        set_random_seeds(cur_seed)
        
        start_time = datetime.datetime.now()
        train_loss_val = 0
        is_best = False
        
        for i,data in enumerate(tqdm(train_dataloader)):
            batch_sent_ids = data['sent_ids']
            batch_sent_attention = data['sent_attention']
            batch_sent_mask = data['sent_mask']
            batch_evi_targets=data['targets']
            
            if torch.cuda.is_available():
                batch_sent_ids = batch_sent_ids.cuda()
                batch_sent_ids = batch_sent_attention.cuda()
                batch_sent_mask = batch_sent_mask.cuda()
                batch_evi_targets = batch_evi_targets.cuda()
            
            outputs = model(batch_sent_ids,batch_sent_ids,batch_sent_mask,is_training=True)
            loss = criterion(outputs.reshape((outputs.shape[0]*outputs.shape[1],outputs.shape[2])),batch_evi_targets.reshape((batch_evi_targets.shape[0]*batch_evi_targets.shape[1])))
            loss.backward()
            train_loss_val+=loss.item()
        train_loss_val/=batch_count
        end_time = datetime.datetime.now()
        logging('Training_loss: ',train_loss_val)
        logging('Time: ',end_time-start_time)
    logging("*"*50)