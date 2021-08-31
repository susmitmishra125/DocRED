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

def preprocess(ori_data,is_training = True, suffix='',create_nan=True):
		"""
		inserts unused tokens and saves them in a dict
		"""
		input_sent = []
		labels=[]
		# this step is to add nan relation evidences
		if create_nan:
			i=0
			for i in range(len(ori_data)):
					label_list=[]
					for label in ori_data[i]['labels']:
							label_list.append((label['h'],label['t']))
					s=0
					label_set=set(label_list)
					for j in range(0,len(ori_data[i]['vertexSet'])):
							for k in range(0,len(ori_data[i]['vertexSet'])):
									if s<len(label_list) and j!=k and ((j,k) not in label_set):
											temp={'h':j,'t':k,'evidence':[]}
											ori_data[i]['labels'].append(temp)
											# for dev and test set we are adding all pairs but for training dataset we add only equal number of nan relation sentences
											if is_training:
												s+=1
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

						input_sent.append(copy.deepcopy(temp_doc['sents']))
						labels.append(copy.deepcopy(label['evidence']))
						
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

def gen_data(ori_data,is_training,filename,model_file):
	doc_index=0
	if is_training:
		modified_list = []
		for doc in ori_data:
			doc_index=doc_index+1
			sys.stdout.write("\r%d/%d docs"%(doc_index,len(ori_data)))

			label_list=[]
			sent_numpy = np.array(doc['sents'],dtype=object)
			for labels in doc['labels']:
				label_list.append((labels['h'],labels['t']))
				temp_doc={}
				temp_doc['labels']=[]
				temp_doc['title']=doc['title']
				temp_doc['vertexSet']=[doc['vertexSet'][labels['h']],doc['vertexSet'][labels['t']]]

				temp_label={}
				temp_label['r']=labels['r']
				temp_label['h']=0
				temp_label['t']=1
				temp_label['evidence']=labels['evidence']

				for i in range(2):
					for j in range(len(temp_doc['vertexSet'][i])):
						if temp_doc['vertexSet'][i][j]['sent_id'] not in temp_label['evidence']:
							temp_label['evidence'].append(temp_doc['vertexSet'][i][j]['sent_id']) # adding the sentences which contains the entity
				temp_label['evidence'].sort()
				temp_doc['sents']=sent_numpy[labels['evidence']].tolist()
				for i in range(2):
					for j in range(len(temp_doc['vertexSet'][i])):
						temp_doc['vertexSet'][i][j]['sent_id'] = temp_label['evidence'].index(temp_doc['vertexSet'][i][j]['sent_id'])
				temp_doc['labels'].append(copy.deepcopy(temp_label))
				temp_doc['rel']=1
				modified_list.append(copy.deepcopy(temp_doc))
			
			label_set=set(label_list)
			# to add nan relations
			for i in range(len(doc['vertexSet'])):
				for j in range(len(doc['vertexSet'])):
					if i!=j and (i,j) not in label_set:
						temp_doc={}
						temp_doc['sents']=doc['sents']
						temp_doc['labels']=[]
						temp_doc['title']=doc['title']
						temp_doc['vertexSet']=[doc['vertexSet'][i],doc['vertexSet'][j]]
						temp_doc['rel']=0
						modified_list.append(copy.deepcopy(temp_doc))
		logging('\nstarted saving '+filename)
		out_file = open(os.path.join(out_path,filename+'.json'),"w")
		json.dump(modified_list,out_file,indent=2)
		out_file.close()
		logging('completed saving '+filename)
	else:
		model = EvidenceClassifier()
		
		if torch.cuda.is_available():
				model.cuda()
				model = nn.DataParallel(model)
		model.load_state_dict(torch.load(model_file))
		modified_list=[]
		doc_index=0
		for doc in ori_data:
			doc_index=doc_index+1
			sys.stdout.write("\r%d/%d docs"%(doc_index,len(ori_data)))
			label_list=[]
			for i in range(len(doc['labels'])):
				modified_list.append(evd_sent(doc,i,model,threshold = 0.3))
				label_list.append((doc['labels'][i]['h'],doc['labels'][i]['t']))
			label_set=set(label_list)
			for i in range(len(doc['vertexSet'])):
				for j in range(len(doc['vertexSet'])):
					if i!=j and (i,j) not in label_set:
						temp_doc={}
						temp_doc['sents']=doc['sents']
						temp_doc['title']=doc['title']
						temp_doc['labels']=[]
						temp_doc['vertexSet']=[doc['vertexSet'][i],doc['vertexSet'][j]]
						temp_doc['rel']=0
						modified_list.append(copy.deepcopy(temp_doc))
		
		logging('\nstarted saving '+filename)
		out_file = open(os.path.join(out_path,filename+'.json'),"w")
		json.dump(modified_list,out_file,indent=2)
		out_file.close()
		logging('completed saving '+filename)

def evd_sent(doc,label_index,model,threshold=0.5):
	"""
	doc is the dict for a document
	label_index is the index of the label to be processed
	returns doc of same format for the input doc and relation with only evidence sentences
	"""
	label = doc['labels'][label_index]
	head=doc['vertexSet'][label['h']]
	tail=doc['vertexSet'][label['t']]


	input_sent = []
	labels = []
	idx_list = []
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

	input_sent.append(copy.deepcopy(temp_doc['sents']))
	labels.append(copy.deepcopy(label['evidence']))

	data={}
	data['input_sent']=input_sent
	data['labels']=labels

	# model loading

	evd_output = predict(data = data,model = model,threshold = 0.3,return_output=True).cpu()
	model.eval()

	sent_numpy = np.array(doc['sents'],dtype=object)
	temp_doc={}
	temp_doc['vertexSet']=[doc['vertexSet'][label['h']],doc['vertexSet'][label['t']]]
	for i in range(2):
		for j in range(len(temp_doc['vertexSet'][i])):
			evd_output[0][temp_doc['vertexSet'][i][j]['sent_id']] = 1 # including the sentences which contain the entity
	pred_evd = np.where(evd_output[0]>=threshold)[0]

	for i in range(2):
		for j in range(len(temp_doc['vertexSet'][i])):
			temp_doc['vertexSet'][i][j]['sent_id'] = int(np.where(pred_evd == temp_doc['vertexSet'][i][j]['sent_id'])[0][0])
	temp_doc['sents'] = sent_numpy[pred_evd].tolist()
	temp_doc['title'] = doc['title']
	temp_doc['labels'] = []
	temp_label = {}
	temp_label['r'] = label['r']
	temp_label['h'] = 0
	temp_label['t'] = 1
	temp_label['evidence']=label['evidence']
	temp_doc['labels'].append(temp_label)
	temp_doc['rel']=1

	return temp_doc

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
		weights = torch.tensor([10.0])
		
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
		# best_epoch_idx,best_epoch_seed,best_dev_acc,start_epoch,model,optimizer=load_ckp(ckp_path, model, optimizer)
		
		for epoch_idx in range(start_epoch, EPOCH):
				model.train()
				model.zero_grad()
				logging('Epoch:', epoch_idx + 1)
				cur_seed = RANDOM_SEED + epoch_idx + 1
				set_random_seeds(cur_seed)
				# random.shuffle(cur_shuffled_train_data)
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

				acc,F1 = predict(dev_data,model,threshold = 0.3)
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
				# save_ckp(checkpoint, is_best, 'checkpoint', 'checkpoint') # uncomment this when running for long time
				logging('\ncheckpoint updated\n\n')
				if epoch_idx+1-best_epoch_idx>=early_stop_count:
						break

		logging("*"*50)
		logging('Best epoch',best_epoch_idx)
		logging('Best Epoch Seed:', best_epoch_seed)

def test(test_data,model_file,threshold = 0.5,save_output=False,save_suffix='_output'):
		model = EvidenceClassifier()
		
		if torch.cuda.is_available():
				model.cuda()
				model = nn.DataParallel(model)
		model.load_state_dict(torch.load(model_file))
		# logging(model)
		acc,F1 = predict(test_data,model,threshold,save_output,save_suffix)
		print('\n\n')
		# for i in range(9):
		#     threshold = 0.1*(i+1)
		#     acc,F1 = predict(test_data,model,threshold)
		#     logging('Test acc:',round(acc,3))
		#     logging('Test F1:',round(F1,3))
		#     print('\n\n')

def predict(data,model,threshold=0.5,save_output=False,save_suffix='_output', return_output = False):
		dev_input_ids = data['input_sent']
		dev_target =data['labels'] 
		dev_size=len(dev_input_ids)
		batch_size=BATCH_SIZE_PRED
		batch_count=int(math.ceil(dev_size/batch_size))
		# logging("Evidence Threshold:",threshold)
		model.eval()
		set_random_seeds(RANDOM_SEED)
		
		true_positive = 0
		false_positive = 0
		false_negative = 0
		true_negative = 0

		output_list=[]
		# for batch_idx in tqdm(range(0,batch_count)):
		for batch_idx in (range(0,batch_count)):
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
						if save_output:
							output_list=output_list+outputs.tolist()
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
		# logging('true_positive',true_positive)
		# logging('false_positive',false_positive)
		# logging('false_negative',false_negative)
		# logging('true_negative',true_negative)
		# logging('gt_pos',true_positive+false_negative)
		# logging('pred_pos',true_positive+false_positive)
		# logging('correct_pos',true_positive)
		preciscion = true_positive/(true_positive+false_positive+1e-05)
		recall = true_positive/(true_positive+false_negative+1e-05)
		# logging('recall',round(recall,3))
		# logging('prec',round(preciscion,3))
		F1 = 2*recall*preciscion/(preciscion+recall+1e-05)
		# logging('F1',round(F1,3))
		acc = true_positive+true_negative


		if save_output:
			json.dump(output_list,open(os.path.join(out_path,save_suffix+'.json'),'w'),indent=2)
		model.train()
		if return_output:
			return outputs
		return acc,F1

if __name__ == "__main__":
		job_mode = 'gen_data'
		do_preprocessing = True
		RANDOM_SEED = 42
		set_random_seeds(RANDOM_SEED)

		in_path='data'
		out_path='prepro_data'
		os.environ['CUDA_VISIBLE_DEVICES']="0,1"
		PRE_TRAINED_MODEL_NAME='bert-base-uncased'
		save_model_file = os.path.join('checkpoint', 'model.h5py')
		model_name='bert'
		BATCH_SIZE=4# 16 for training
		BATCH_SIZE_PRED=32# 64 
		EPOCH = 100
		update_freq = 2
		early_stop_count = 5

		n_gpu = torch.cuda.device_count()
		if not os.path.exists(out_path):
				os.mkdir(out_path)

		MAX_LEN=512
		SEP='[SEP]'
		MASK = '[MASK]'
		CLS = '[CLS]'
		# bert = Bert(BertModel, PRE_TRAINED_MODEL_NAME)
		tz = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME) # tokenizer
		train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
		dev_file_name = os.path.join(in_path, 'dev.json') # dev but this will be used for testing purposes
		# test_file_name = os.path.join(in_path, 'test.json') # not used because labels are not available
		# preprocessing
		train_data = json.load(open(train_annotated_file_name))[:30]
		train_size = len(train_data)
		dev_data = train_data[:train_size//10] # choosing 10% of train data into dev 
		train_data = train_data[train_size//10:]
		test_data = json.load(open(dev_file_name))[:30]


		logging('Train size:',len(train_data))
		logging('Dev size:',len(dev_data))
		logging('Test size:',len(test_data))
		
		if do_preprocessing and job_mode=='train':
				preprocess(train_data, is_training = True, suffix='train',create_nan=True)
				preprocess(dev_data, is_training = False, suffix='dev',create_nan=True)
		elif do_preprocessing and job_mode=='test':
				preprocess(test_data, is_training = False, suffix='test', create_nan=True) # will use this as test data
		# elif do_preprocessing and job_mode == 'gen_data':
				# preprocess(dev_data,is_training = False,suffix='dev',create_nan = False)
				# preprocess(test_data,is_training = False,suffix='test',create_nan = False)



		# training
		if job_mode =='train':
				train_data = json.load(open(os.path.join(out_path,'train_data.json'),'r'))
				dev_data = json.load(open(os.path.join(out_path,'dev_data.json'),'r'))
				train(train_data = train_data,dev_data = dev_data, model_file=save_model_file)
		# testing
		elif job_mode=='test':
				test_data = json.load(open(os.path.join(out_path,'test_data.json'),'r'))
				logging('Started Testing')
				test(test_data = test_data,model_file = save_model_file, threshold = 0.3)
		elif job_mode == 'gen_data':
				gen_data(ori_data = train_data,is_training=True,filename="modified_train_data",model_file=save_model_file)
				gen_data(ori_data = dev_data,is_training=False,filename="modified_dev_data",model_file=save_model_file)
				gen_data(ori_data = test_data,is_training=False,filename="modified_test_data",model_file=save_model_file)
		# model = EvidenceClassifier()
		
		# if torch.cuda.is_available():
		# 		model.cuda()
		# 		model = nn.DataParallel(model)
		# model.load_state_dict(torch.load(save_model_file))
		# print(evd_sent(doc = dev_data[0],label_index=0,model=model,threshold=0.3))
						
