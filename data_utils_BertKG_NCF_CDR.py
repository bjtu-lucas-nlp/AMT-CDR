"""
@Paper: AMT-CDR: A Deep Adversarial Multi-channel Transfer Network for Cross-domain Recommendation
@author: Kezhi Lu, Qian Zhang
@time: June 20th, 2023
"""
import numpy as np
import pandas as pd 
import scipy.sparse as sp
import collections
import itertools
import torch.utils.data as data
from transformers import BertTokenizer,DataCollatorForLanguageModeling
import config

def load_all(test_num=100):
	train_data = pd.read_csv(
		config.train_rating,   #config.train_rating,
		sep='\t', header=0, names=['user', 'item', 'review'],
		usecols=[0, 1, 4], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data_review = train_data['review'].tolist()
	train_input_ids,train_token_type_ids,train_attention_masks,masked_labels,train_next_sentence_labels = text_2_bertTokens(train_data_review)
	train_data = train_data.values[:,:2].tolist()	# 取user和item列

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = []
	with open(config.test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			u = eval(arr[0])[0]
			test_data.append([u, eval(arr[0])[1]])
			for i in arr[1:]:
				test_data.append([u, int(i)])
			line = fd.readline()
	return train_data, test_data, user_num, item_num, train_mat, train_input_ids,train_token_type_ids,train_attention_masks,masked_labels,train_next_sentence_labels

def construct_kg(kg_np):
	print('constructing knowledge graph ...')
	kg = collections.defaultdict(list)
	triples_num = 0
	for head, relation, tail in kg_np:
		kg[head].append((tail, relation))
		kg[tail].append((head, relation))
		triples_num += 1
	if 8421 in kg:
		print('Finish constructing knowledge graph triples: ', str(triples_num), ' demo item 8421: ', kg[8421])

	return kg

def load_kg(kg_path, source_train_data, target_train_data):
	print('reading KG file ...')
	# reading kg file
	kg_np = np.loadtxt(kg_path, dtype=np.int32)
	n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
	n_relation = len(set(kg_np[:, 1]))
	n_relation += 1
	kg = construct_kg(kg_np)
	user_history_dict = dict()
	for i, triple in source_train_data.iterrows():  # Source dataset
		user = triple[0]
		item = triple[1]
		if user not in user_history_dict:
			user_history_dict[user] = []
		user_history_dict[user].append(item)
	for i, triple in target_train_data.iterrows():  # Target dataset
		user = triple[0]
		item = triple[1]
		if user not in user_history_dict:
			user_history_dict[user] = []
		user_history_dict[user].append(item)
	for user in user_history_dict:
		user_inter_items = user_history_dict[user]
		is_in_kg = False
		for item in user_inter_items:
			if item in kg:
				is_in_kg = True
				break
		if is_in_kg == False:
			if len(user_inter_items) == 1:
				kg[user_inter_items[0]].append((user_inter_items[0], n_relation - 1))
			else:
				zuhe_list = list(itertools.combinations(user_inter_items, 2))
				for element in zuhe_list:
					kg[element[0]].append((element[1], n_relation - 1))
					kg[element[1]].append((element[0], n_relation - 1))
	return n_entity, n_relation, kg, user_history_dict

def get_ripple_set(kg, user_history_dict, args):
	print('constructing ripple set ...')
	if 834 in user_history_dict:
		print("user_history_dict demo user: ", '834: ', user_history_dict[834])
	if 18 in user_history_dict:
		print("user_history_dict demo user: ", '18: ', user_history_dict[18])
	ripple_set = collections.defaultdict(list)
	for user in user_history_dict:
		for h in range(args.n_hop):
			memories_h = []
			memories_r = []
			memories_t = []
			if h == 0:
				tails_of_last_hop = user_history_dict[user]
			else:
				tails_of_last_hop = ripple_set[user][-1][2]

			for entity in tails_of_last_hop:
				for tail_and_relation in kg[entity]:
					memories_h.append(entity)
					memories_r.append(tail_and_relation[1])
					memories_t.append(tail_and_relation[0])
			if len(memories_h) == 0:
				print("User: ", user, " hop: ", str(h), " tails_of_last_hop: ", tails_of_last_hop, " memories_h: ",
					  memories_h, " memories_r: ", memories_r, " memories_t: ", memories_t, " ripple_set[user], ",
					  ripple_set[user])
				ripple_set[user].append(ripple_set[user][-1])
			else:
				# sample a fixed-size 1-hop memory for each user
				replace = len(memories_h) < args.n_memory
				indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
				memories_h = [memories_h[i] for i in indices]
				memories_r = [memories_r[i] for i in indices]
				memories_t = [memories_t[i] for i in indices]
				ripple_set[user].append((memories_h, memories_r, memories_t))
	return ripple_set

class DMTN_Data(data.Dataset):
	def __init__(self, source_list, target_list, mode = None, data_source = None, data_version = "V1"):
		super(DMTN_Data, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.mode = mode
		self.data_source = data_source
		self.data_version = data_version
		if mode == "Train":
			if data_source == "source_target":
				if data_version in ["V1_sent", "V1_sent_T2"]:
					self.features = pd.concat([source_list[1][['user', 'item']], target_list[1][['user', 'item']]],
											  axis=0).values.tolist()
					self.labels = pd.concat([source_list[1]['rating'], target_list[1]['rating']],
											axis=0).values.tolist()
					self.reviews = source_list[4] + target_list[4]
					self.domain_labels = [0] * len(source_list[4]) + [1] * len(target_list[4])
				else:
					self.features = pd.concat([source_list[1][['user','item']],target_list[1][['user','item']]],axis=0).values.tolist()
					self.labels = pd.concat([source_list[1]['rating'],target_list[1]['rating']],axis=0).values.tolist()
					self.input_ids = source_list[4] + target_list[4]
					self.token_type_ids = source_list[5] + target_list[5]
					self.attention_masks = source_list[6] + target_list[6]
					self.masked_labels = source_list[7] + target_list[7]
					self.next_sentence_labels = source_list[8] + target_list[8]
					self.domain_labels = [0] * len(source_list[4]) + [1] * len(target_list[4])
			elif data_source == "Source":
				if data_version in ["V1_sent", "V1_sent_T2"]:
					self.features = source_list[1][['user', 'item']].values.tolist()
					self.labels = source_list[1]['rating'].values.tolist()
					self.reviews = source_list[4]
					self.domain_labels = [0] * len(self.labels)
				else:
					self.features = source_list[1][['user', 'item']].values.tolist()
					self.labels = source_list[1]['rating'].values.tolist()
					self.input_ids = source_list[4]
					self.token_type_ids = source_list[5]
					self.attention_masks = source_list[6]
					self.masked_labels = source_list[7]
					self.next_sentence_labels = source_list[8]
					self.domain_labels = [0] * len(self.labels)
			elif data_source == "Target":
				if data_version in ["V1_sent","V1_sent_T2"]:
					self.features = target_list[1][['user', 'item']].values.tolist()
					self.labels = target_list[1]['rating'].values.tolist()
					self.reviews = target_list[4]
					self.domain_labels = [1] * len(self.labels)
				else:
					self.features = target_list[1][['user', 'item']].values.tolist()
					self.labels = target_list[1]['rating'].values.tolist()
					self.input_ids = target_list[4]
					self.token_type_ids = target_list[5]
					self.attention_masks = target_list[6]
					self.masked_labels = target_list[7]
					self.next_sentence_labels = target_list[8]
					self.domain_labels = [1] * len(self.labels)
		elif mode == "Vali":
			if data_source == "Source":
				if data_version in ["V1_sent","V1_sent_T2"]:
					self.reviews = source_list[5]
				self.features = source_list[2][['user','item']].values.tolist()
				self.labels = source_list[2]['rating'].values.tolist()
				self.domain_labels = [0] * len(self.labels)
			elif data_source == "Target":
				if data_version in ["V1_sent","V1_sent_T2"]:
					self.reviews = target_list[5]
				self.features = target_list[2][['user','item']].values.tolist()
				self.labels = target_list[2]['rating'].values.tolist()
				self.domain_labels = [1] * len(self.labels) # lucas 230327 added domain label
		elif mode == "Test":
			if data_source == "Source":
				if data_version in ["V1_sent","V1_sent_T2"]:
					self.reviews = source_list[6]
				self.features = source_list[3][['user', 'item']].values.tolist()
				self.labels = source_list[3]['rating'].values.tolist()
				self.domain_labels = [0] * len(self.labels)
			elif data_source == "Target":
				if data_version in ["V1_sent","V1_sent_T2"]:
					self.reviews = target_list[6]
				self.features = target_list[3][['user', 'item']].values.tolist()
				self.labels = target_list[3]['rating'].values.tolist()
				self.domain_labels = [1] * len(self.labels)

	def ng_sample(self):
		assert self.mode, 'no need to sampling when testing'
		self.features_ng = []
		self.input_ids_ng = []
		self.token_type_ids_ng = []
		self.attention_masks_ng = []
		self.train_labels_ng = []
		self.next_sentence_labels_ng = []

		input_id_str = "||".join(list(map(str,[101,102]+[0]*510)))
		token_type_id_str = "||".join(list(map(str, [0, 0]+[0]*510)))
		attention_mask_str = "||".join(list(map(str, [1, 1]+[0]*510)))
		train_label_str = "||".join(list(map(str, [-100, -100]+[-100]*510)))
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])
				self.input_ids_ng.append(input_id_str)
				self.token_type_ids_ng.append(token_type_id_str)
				self.attention_masks_ng.append(attention_mask_str)
				self.train_labels_ng.append(train_label_str)
				self.next_sentence_labels_ng.append(0)
		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features_ps + self.features_ng
		self.labels_fill = labels_ps + labels_ng

		self.input_ids_fill = self.input_ids + self.input_ids_ng
		self.token_type_ids_fill = self.token_type_ids + self.token_type_ids_ng
		self.attention_masks_fill = self.attention_masks + self.attention_masks_ng
		self.train_labels_fill = self.train_labels + self.train_labels_ng
		self.next_sentence_labels_fill = self.next_sentence_labels + self.next_sentence_labels_ng


	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		if self.mode == "Train":
			if self.data_version in ["V1_sent","V1_sent_T2"]:
				user = int(self.features[idx][0])
				item = int(self.features[idx][1])
				label = int(self.labels[idx])
				review = self.reviews[idx]
				domain_label = self.domain_labels[idx]
				return user, item, label, review, domain_label
			else:
				user = int(self.features[idx][0])
				item = int(self.features[idx][1])
				label = int(self.labels[idx])
				bert_input_id = self.input_ids[idx]
				bert_token_type_id = self.token_type_ids[idx]
				bert_attention_mask = self.attention_masks[idx]
				bert_train_label = self.masked_labels[idx]
				bert_next_sentence_label = self.next_sentence_labels[idx]
				domain_label = self.domain_labels[idx]
				return user, item, label,bert_input_id,bert_token_type_id,bert_attention_mask,bert_train_label,bert_next_sentence_label, domain_label
		else:
			if self.data_version in ["V1_sent","V1_sent_T2"]:
				user = int(self.features[idx][0])
				item = int(self.features[idx][1])
				label = int(self.labels[idx])
				review = self.reviews[idx]
				domain_label = self.domain_labels[idx]
				return user, item, label, review, domain_label
			else:
				user = int(self.features[idx][0])
				item = int(self.features[idx][1])
				label = int(self.labels[idx])
				domain_label = self.domain_labels[idx]
				return user, item ,label, domain_label
