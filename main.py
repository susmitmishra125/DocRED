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
import shutil
# os.chdir("/home/mtech1/19CS60R28/susmit/DocRed_hongwang600/DocRed")

from nltk.tokenize import WordPunctTokenizer
from pytorch_transformers import BertModel, BertTokenizer, BertPreTrainedModel, AdamW
import torch
import torch.autograd as autograd
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append("/home/mtech1/19CS60R28/susmit/DocRed_hongwang600/DocRed/models")
# from bert import Bert
# from torchviz import make_dot
torch.backends.cudnn.deterministic = True

def logging(*msg):
    for i in range(0,len(msg)):
        if(i==len(msg)-1):
            end='\n'
        else:
            end=' '
        print(msg[i],end=end)
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


def load_ckp(checkpoint_fpath, model, optimizer):
    try:
        checkpoint = torch.load(checkpoint_fpath)
    except OSError as e:
        return -1,-1,-1.0,0,model,optimizer 
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['best_epoch_idx'], checkpoint['best_epoch_seed'], checkpoint['best_dev_acc'], checkpoint['epoch'], model, optimizer


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir+'/checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
            best_fpath = best_model_dir+'/best_model.pt'
            shutil.copyfile(f_path, best_fpath)

def preprocess(data_file_name, max_length = 512, is_training = True, suffix=''):
    """
    inserts unused tokens and saves them in a dict
    """
    ori_data=json.load(open(data_file_name))[0:10]
    input_sent = []
    labels=[]
    i=0
    for doc in tqdm(ori_data):
        i=i+1
        # sys.stdout.write("\r%d/%d docs"%(i,len(ori_data)))
        # this dict is used to take care of multiple relations with same head and tail
        head_tail_index={}
        for label in doc['labels']:
            idx_list=[]
            head=doc['vertexSet'][label['h']]
            tail=doc['vertexSet'][label['t']]
            if (label['h'],label['t']) in head_tail_index:
                labels[head_tail_index[(label['h'],label['t'])]]+=label['evidence']
                continue
            else:
                head_tail_index[(label['h'],label['t'])]=len(input_sent)
            for entity in head:
                if (entity['sent_id'],entity['pos'][0],'[unused0]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][0],'[unused0]'))
                if (entity['sent_id'],entity['pos'][1],'[unused1]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][1],'[unused1]'))
            for entity in tail:
                if (entity['sent_id'],entity['pos'][0],'[unused2]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][0],'[unused2]'))
                if (entity['sent_id'],entity['pos'][1],'[unused3]') not in idx_list:
                    idx_list.append((entity['sent_id'],entity['pos'][1],'[unused3]'))
            idx_list.sort(key=lambda tup:(tup[0],tup[1]),reverse=True)
            temp_doc=copy.deepcopy(doc)
            for loc in idx_list:
                temp_doc['sents'][loc[0]].insert(loc[1],loc[2])

            input_sent.append(temp_doc['sents'])
            labels.append(label['evidence'])
            
    data={}
    data['input_sent']=input_sent
    data['labels']=labels
    logging("Number of instances: {}".format(len(input_sent)))
    logging("Started saving")
    out_file = open(os.path.join(out_path,suffix+'_data.json'),"w")
    json.dump(data,out_file,indent=2)
    out_file.close()
    logging("completed saving\n")



def get_max(cur_samples):
    max_doc_len = 0
    max_sent_count = 0
    batch_sent_ids = []
    i=0
    sent_attention = []
    for doc in cur_samples:
        s=2 # because CLS and SEP token will be added
        doc_sent_ids = []
        for sent in doc:
            sent_ids = []
            for token in sent:
                if token.startswith('[unused'):
                    sent_ids += [int(token[-2])+1]
                else:
                    sent_ids+=tz.convert_tokens_to_ids(tz.tokenize(token))
            s += len(sent_ids)
            doc_sent_ids.append(sent_ids)
        batch_sent_ids.append(doc_sent_ids)
        s=min(s,512)
        max_doc_len = max(max_doc_len,s)
        max_sent_count = max(max_sent_count,len(doc))
        i+=1
    return max_doc_len, max_sent_count, batch_sent_ids


def get_batch_data(cur_samples_doc,cur_samples_evi = None):
    """
    input is a list of docs
    each doc is list of sentences
    each sent is a list of tokens
    Returns dictionary of training samples and labels as numpy array
    """
    no_samples = len(cur_samples_doc)
    max_doc_len,max_sent_count, batch_sent_ids= get_max(cur_samples_doc)
    input_ids = np.zeros((no_samples,max_doc_len))
    sent_attention = np.zeros((no_samples,max_doc_len))
    sent_mask = np.zeros((no_samples,max_sent_count, max_doc_len))# required for averaging the sentence embeddings
    evi_target = np.zeros((no_samples,max_sent_count))
    for i in range(no_samples):
        doc=[101] # CLS token
        lower_index = 1
        j=0
        for sent in batch_sent_ids[i]:
            if(lower_index>510):
                break
            doc+=sent
            sent_mask[i][j][lower_index:min(lower_index+len(sent),511)]=1.0
            lower_index+=len(sent)
            if(lower_index>510):
                doc=doc[:511]
            j+=1
        doc.append(102) # SEP token
        if j<max_sent_count:
            sent_mask[i][j:,:]=1.0 # for pad sentences
        # sent_ids,attention,start_ids = bert.subword_tokenize_to_ids(doc,max_doc_len)
        # input_ids[i],sent_attention[i]=sent_ids[0],attention[0]
        for j in range(len(doc)):
            input_ids[i][j]=doc[j]
        sent_attention[i][:len(doc)] = 1
        # _,_,_ = bert.subword_tokenize_to_ids(doc,max_doc_len+2)
        if cur_samples_evi != None :
            evi_target[i][cur_samples_evi[i]]=1
            if j<max_sent_count:
                evi_target[i][j:]=2 # this will be used later to remove loss from this sample
    return {
        'sent_ids':input_ids,
        'sent_attention':sent_attention,
        'sent_mask':sent_mask,
        'targets':evi_target,
    }


class EvidenceClassifier(nn.Module):
    def __init__(self):
        super(EvidenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dense = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)
    def forward(self, input_ids, attention_mask,sent_mask,is_training=False):
        last_hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        # sent_mask=sent_mask.float()
        output=torch.bmm(sent_mask,last_hidden_state)/sent_mask.sum(axis=2)[...,None]
        logits = self.dense(output) # shape b,k,1
        logits = logits.view((output.shape[0],output.shape[1]))
        # logits = logits.squeeze() # shape b,k
        # note the sigmod layer will be inside the BCEWithLogitsLoss
        if is_training:
            return logits
        else:
            return self.sigmoid(logits)
        #     return self.logsoftmax(logits)
        # else:
        #     return self.softmax(logits)

def train(train_data,dev_data,model_file):
    input_sent = train_data['input_sent']
    target = train_data['labels']
    dev_input_ids = dev_data['input_sent']
    dev_target = dev_data['labels']
    train_size=len(input_sent)
    batch_size=BATCH_SIZE
    batch_count=int(math.ceil(train_size)/batch_size)
    model=EvidenceClassifier()
    weights = torch.tensor([1.0])
    
    # logging(model)
    
    if torch.cuda.is_available():
        # model.cuda
        model = nn.DataParallel(model, device_ids = [0,1])
        model.to(f'cuda:{model.device_ids[0]}')

        weights = weights.cuda()
    # criterion = nn.NLLLoss(weight=weights, reduction='mean',ignore_index=2)
    criterion = nn.BCEWithLogitsLoss(reduction='mean',pos_weight= weights)
    optimizer = AdamW(model.parameters(),lr=1e-05,correct_bias=False)
    
    logging(optimizer)
    
    best_dev_acc = -1
    best_epoch_idx = -1
    best_epoch_seed = -1
    start_epoch = 0
    ckp_path=os.path.join('checkpoint',model_name+'_checkpoint.pt')
    best_epoch_idx,best_epoch_seed,best_dev_acc,start_epoch,model,optimizer=load_ckp(ckp_path, model, optimizer)
    
    for epoch_idx in range(start_epoch, EPOCH):
        model.train()
        model.zero_grad()
        logging('Epoch:', epoch_idx + 1)
        cur_seed = RANDOM_SEED + epoch_idx + 1
        set_random_seeds(cur_seed)
        random.shuffle(cur_shuffled_train_data)
        start_time = datetime.datetime.now()
        train_loss_val = 0
        is_best = False
        
        for batch_idx in tqdm(range(0,batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(train_size,batch_start+batch_size)
            data = get_batch_data(input_sent[batch_start:batch_end],target[batch_start:batch_end])
            batch_sent_ids = torch.tensor(data['sent_ids']).to(torch.int64)
            batch_sent_attention = torch.tensor(data['sent_attention']).to(torch.int64)
            batch_sent_mask = torch.tensor(data['sent_mask']).float()
            batch_evi_targets = torch.tensor(data['targets']).to(torch.int64)
            
            if torch.cuda.is_available():
                batch_sent_ids = batch_sent_ids.cuda()
                batch_sent_attention = batch_sent_attention.cuda()
                batch_sent_mask = batch_sent_mask.cuda()
                batch_evi_targets = batch_evi_targets.cuda()
            batch_sent_ids = autograd.Variable(batch_sent_ids)
            batch_sent_attention = autograd.Variable(batch_sent_attention)
            batch_sent_mask = autograd.Variable(batch_sent_mask)
            batch_evi_targets = autograd.Variable(batch_evi_targets)

            outputs = model(batch_sent_ids,batch_sent_attention,batch_sent_mask,is_training=True)
            y_pred = outputs.view((outputs.shape[0]*outputs.shape[1]))
            labels = batch_evi_targets.view((batch_evi_targets.shape[0]*batch_evi_targets.shape[1])).float()
            y_pred[labels==2] = -np.Inf
            labels[labels==2] = 0

            loss = criterion(y_pred,labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            # if (batch_idx + 1) % update_freq == 0:
            optimizer.step()
            # model.zero_grad()
            train_loss_val +=loss.item()
        train_loss_val /=batch_count
        end_time = datetime.datetime.now()
        logging('\nTraining_loss: ',train_loss_val)
        logging('Time: ',end_time-start_time)
        logging('\nDev_Results\n')

        acc,F1 = predict(dev_data,model)
        logging('Dev acc:',round(acc,3))
        logging('Dev F1:',round(F1,3))
        if F1 > best_dev_acc:
            best_epoch_idx=epoch_idx+1
            best_epoch_seed=cur_seed
            logging("model saved ...")
            best_dev_acc=F1
            torch.save(model.state_dict(),model_file)
            is_best=True
        checkpoint = {
        'best_epoch_idx':best_epoch_idx,
        'best_epoch_seed':best_epoch_seed,
        'best_dev_acc':best_dev_acc,
        'epoch':epoch_idx+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        save_ckp(checkpoint, is_best, 'checkpoint', 'checkpoint') # uncomment this when running for long time
        logging('\ncheckpoint updated\n\n')
        if epoch_idx+1-best_epoch_idx>=early_stop_count:
            break

    logging("*"*50)
    logging('Best epoch',best_epoch_idx)
    logging('Best Epoch Seed:', best_epoch_seed)

def test



def predict(dev_data,model,threshold=0.5):
    dev_input_ids = dev_data['input_sent']
    dev_target =dev_data['labels'] 
    dev_size=len(dev_input_ids)
    batch_size=BATCH_SIZE_PRED
    batch_count=int(math.ceil(dev_size/batch_size))
    logging("Evidence Threshold:",threshold)
    model.eval()
    set_random_seeds(RANDOM_SEED)
    
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for batch_idx in tqdm(range(0,batch_count)):
        batch_start = batch_idx * batch_size
        batch_end = min(dev_size,batch_start+batch_size)
        data = get_batch_data(dev_input_ids[batch_start:batch_end], dev_target[batch_start:batch_end])

        batch_sent_ids = torch.tensor(data['sent_ids']).to(torch.int64)
        batch_sent_attention = torch.tensor(data['sent_attention']).to(torch.int64)
        batch_sent_mask = torch.tensor(data['sent_mask']).float()
        batch_evi_targets = torch.tensor(data['targets']).to(torch.int64)
        
        if torch.cuda.is_available():
            batch_sent_ids = batch_sent_ids.cuda()
            batch_sent_attention = batch_sent_attention.cuda()
            batch_sent_mask = batch_sent_mask.cuda()
            batch_evi_targets = batch_evi_targets.cuda()
        with torch.no_grad():
            outputs = model(batch_sent_ids,batch_sent_attention,batch_sent_mask,is_training=False)
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                if(batch_evi_targets[i][j]==2):
                    continue
                if outputs[i][j]>=threshold and batch_evi_targets[i][j]==1:
                    true_positive+=1
                elif outputs[i][j]>=threshold and batch_evi_targets[i][j]==0:
                    false_positive+=1
                elif outputs[i][j]<=threshold and batch_evi_targets[i][j]==1:
                    false_negative+=1
                else:
                    true_negative+=1
    logging('true_positive',true_positive)
    logging('false_positive',false_positive)
    logging('false_negative',false_negative)
    logging('true_negative',true_negative)

    preciscion = true_positive/(true_positive+false_positive+1e-05)
    recall = true_positive/(true_positive+false_negative+1e-05)
    F1 = 2*recall*preciscion/(preciscion+recall+1e-05)
    acc = true_positive+true_negative
    model.train()
    return acc,F1



if __name__ == "__main__":
    RANDOM_SEED = 42
    set_random_seeds(RANDOM_SEED)

    in_path='data'
    out_path='prepro_data'
    os.environ['CUDA_VISIBLE_DEVICES']="0,1"
    PRE_TRAINED_MODEL_NAME='bert-base-uncased'
    save_model_file = os.path.join('checkpoint', 'model.h5py')
    model_name='bert'
    BATCH_SIZE=16# for training
    BATCH_SIZE_PRED=64
    EPOCH = 100
    update_freq = 2
    early_stop_count = 5

    n_gpu = torch.cuda.device_count()
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    MAX_LEN=512
    SEP='[SEP]'
    MASK = '[MASK]'
    CLS = "[CLS]"
    # bert = Bert(BertModel, PRE_TRAINED_MODEL_NAME)
    tz = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME) # tokenizer
    train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
    dev_file_name = os.path.join(in_path, 'dev.json') # but his will be used for testing purposes
    # test_file_name = os.path.join(in_path, 'test.json') # not used because labels are not available
    # preprocessing
    preprocess(train_annotated_file_name, max_length = 512, is_training = True, suffix='train')
    preprocess(dev_file_name, max_length = 512, is_training = False, suffix='test') # will use this as test data


    train_data = json.load(open(os.path.join(out_path,'train_data.json'),'r'))

    test_data = json.load(open(os.path.join(out_path,'test_data.json'),'r'))

    # training
    train(train_data = train_data,dev_data = dev_data, model_file=save_model_file)

