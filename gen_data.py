"""
This file is to create modified datasets with only evidence sentences and positive realtions
"""
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

def evidence_sent(doc):
	"""
	this function takes dict of a list of sentences and the relation
	returns the list of indices of list of sentences that are having evidences 
	"""


def gen_dataset_without_evidence(ori_data, suffix = ''):
	"""
	this function takes the dev dataset and creates another dataset with pairings between all entities
	which then needs to be changed to dataset with positive evidence
	"""
	modified_list=[]

	for doc in ori_data:
		head_tail_index=set()
		sent_numpy = np.array(doc['sents'],dtype=object)
		for labels in doc['labels']:
			temp_doc={}
			temp_doc['title'] = doc['title']
			temp_doc['sents'] = doc['sents']
			temp_doc['labels'] = []
			temp_doc['vertexSet'] = [doc['vertexSet'][labels['h']],doc['vertexSet'][labels['t']]]
			temp_label={}
			temp_label['r']=labels['r']
			temp_label['h']=0
			temp_label['t']=1
			temp_label['evidence']=labels['evidence']
			
			temp_doc['labels'].append(copy.deepcopy(temp_label))

			head_tail_index.add((labels['h'],labels['t']))
			# note that i have not added the evidence key pair
			modified_list.append(copy.deepcopy(temp_doc))
		for i in range(len(doc['vertexSet'])):
			for j in range(len(doc['vertexSet'])):
				if i==j:
					continue
				if (i,j) in head_tail_index:
					continue
				temp_doc = {}
				temp_doc['title'] = doc['title']
				temp_doc['sents'] = doc['sents']
				temp_doc['labels'] = []
				temp_doc['vertexSet'] = [doc['vertexSet'][i],doc['vertexSet'][j]]

				temp_label = {}
				temp_label['h']=0
				temp_label['t']=1
				temp_label['evidence']=[]
				temp_doc['labels'].append(copy.deepcopy(temp_label))

				modified_list.append(copy.deepcopy(temp_doc))
	print('started saving...')
	out_file = open(os.path.join(out_path,suffix+'.json'),'w')
	json.dump(modified_list,out_file,indent=2)
	print('completed saving...')


def gen_dataset(data,suffix = '',is_training = False):
	if is_training:
		modified_list = []
		# print(data)
		s=0
		i=-1
		labels_count=0
		evidence_outlier = 0
		for doc in data:
			i+=1
			j=-1
			sent_numpy = np.array(doc['sents'],dtype=object)
			for labels in doc['labels']:
				j+=1
				labels['evidence'].sort()
				labels_count+=1
				if len(labels['evidence'])==0:
					evidence_outlier+=1
					continue# if the evidence set is empty then we dont add in the training dataset
				temp_doc={}
				temp_doc['sents']=sent_numpy[labels['evidence']].tolist()
				temp_doc['labels']=[]
				temp_doc['title']=doc['title']
				temp_doc['vertexSet']=[doc['vertexSet'][labels['h']],doc['vertexSet'][labels['t']]]
				temp_label={}
				temp_label['r']=labels['r']
				temp_label['h']=0
				temp_label['t']=1
				
				temp_doc['labels'].append(temp_label)
				for instance in copy.deepcopy(temp_doc['vertexSet']):
					flag=0
					k=len(instance)-1
					# print('len instance',len(instance))
					while k>=0:
						# print("i",i,"j",j,"k",k)
						if instance[k]['sent_id'] not in labels['evidence']:
							instance.pop(k)
						else:
							instance[k]['sent_id'] = labels['evidence'].index(instance[k]['sent_id'])
							flag=1
						k-=1
					if flag==0:
						evidence_outlier+=1
						break
				# print('vertexSet',temp_doc['vertexSet'])
				if flag:
					modified_list.append(copy.deepcopy(temp_doc))
		print('evidence_outlier:',evidence_outlier)
		print('labels_count:',labels_count)
		out_file=open(os.path.join(out_path,'train_annotated_modified.json'),'w')
		print('\nstarted saving')
		json.dump(modified_list,out_file,indent=2)
		print('\nsave completed')
	else:
		print("nothing yet")


if __name__ == "__main__":
	in_path='data'
	out_path='prepro_data'
	train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
	dev_file_name = os.path.join(in_path, 'dev.json')

	train_data = json.load(open(train_annotated_file_name))[:10]
	train_size = len(train_data)
	dev_data = train_data[:train_size//10] # choosing 10% of train data into dev data
	train_data = train_data[train_size//10:]
	test_data = json.load(open(dev_file_name))[:10]

	# for generating the train dataset usign gold labels
	gen_dataset(data = train_data,suffix = 'train',is_training = True)
	# for generating the docred dataset without all pairings
	gen_dataset_without_evidence(dev_data, suffix = 'dev_dataset_modified_without_evidence')
	gen_dataset_without_evidence(test_data, suffix = 'test_dataset_modified_without_evidence')
	

