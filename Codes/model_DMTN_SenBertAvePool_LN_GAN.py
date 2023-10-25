import torch
import torch.nn as nn
from eval_BertKG_NCF_CDR import LossMSE
from sentence_transformers import SentenceTransformer, util
from torch.autograd import Function
import math

# ------LN layer------#
class LNLayer(nn.Module):
	def __init__(self, hidden_size1, hidden_size, layer_norm_eps):
		super().__init__()
		self.dense = nn.Linear(hidden_size1, hidden_size * 2)
		self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
		self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.act1 = nn.ReLU()

	def forward(self, hidden_states, input_tensor):
		hidden_out = self.dense(hidden_states)
		hidden_out = self.act1(hidden_out)
		hidden_out = self.dense2(hidden_out)
		hidden_out = self.LayerNorm(hidden_out + input_tensor)
		return hidden_out

class EncoderLayer(nn.Module):
	def __init__(self, num_input, num_dim):
		super(EncoderLayer, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(num_input, num_dim),
			nn.ReLU()
		)

	def forward(self, input_data):
		coding = self.encoder(input_data)
		return coding

# ------GRL------#
class ReverseLayerF(Function):
	@staticmethod
	def forward(ctx, x, p):
		ctx.p = p
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.p
		return output, None

# ------Classifier for GRL------#
class Classifier(nn.Module):
	def __init__(self, num_dim_s_2, num_dim_hidden):
		super(Classifier, self).__init__()
		self.encoder = EncoderLayer(num_dim_s_2, num_dim_hidden)
		self.classifier = nn.Sequential(
			nn.Linear(num_dim_hidden, 2),
			nn.Sigmoid()
        )
	def forward(self, input_data, p):
		embeds = self.encoder(input_data)
		embeds_revsers = ReverseLayerF.apply(embeds, p)
		label = self.classifier(embeds_revsers)
		return label

# ------Model------#
class DMTN_SenBert_LN_TransForKG_3GAN(nn.Module):
	def __init__(self, user_num, item_num, args, kg_relation_num, GMF_model=None, MLP_model=None, deviceid = 0):
		super(DMTN_SenBert_LN_TransForKG_3GAN, self).__init__()
		self.factor_num = args.factor_num
		self.num_layers = args.num_layers
		self.dropout = args.dropout
		# ------Parameters of KG------#
		self.dim = args.dim
		self.n_dim = args.n_dim
		self.kg_dropout = args.kg_dropout
		self.n_hop = args.n_hop
		self.kge_weight = args.kge_weight
		self.l2_weight = args.l2_weight
		self.lr = args.lr
		self.n_memory = args.n_memory
		self.item_update_mode = args.item_update_mode
		self.using_all_hops = args.using_all_hops
		self.n_entity = item_num
		self.n_relation = kg_relation_num
		self.ablation_model = args.ablation_model

		self.sen_bert = SentenceTransformer('all-MiniLM-L6-v2')
		self.hidden_size1 = 384 # 256
		self.hidden_size = self.hidden_size1# self.factor_num
		self.output_LN = LNLayer(self.hidden_size1, self.hidden_size1, 1e-12)

		self.embed_user_BERT = nn.Embedding(user_num, self.hidden_size1)  # factor_num, 768
		self.embed_item_BERT = nn.Embedding(item_num, self.hidden_size1)  # factor_num, 768

		self.user_semantic_classifier = Classifier(self.hidden_size1, 10)  # 768, 10
		if self.ablation_model not in ['DMTN-No-KGL', 'DMTN-No-KGL-SRL', 'DMTN-No-KGL-IRL']:
			self.user_kg_classifier = Classifier(self.dim, 10)

		self.model = args.model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model
		print("AMT-CDR user_num: ", user_num, "--item_num: ", item_num)
		self.embed_user_GMF = nn.Embedding(user_num, self.factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, self.factor_num)
		self.user_GMF_classifier = Classifier(self.factor_num, 10)

		self.embed_user_MLP = nn.Embedding(
				user_num, self.factor_num * (2 ** (self.num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
				item_num, self.factor_num * (2 ** (self.num_layers - 1)))
		self.user_NCF_classifier = Classifier(self.factor_num + self.factor_num * (2 ** (self.num_layers - 1)), 10)
		# layer1：256-128；layer2: 128-64; layer3: 64-32
		MLP_modules = []
		for i in range(self.num_layers):
			input_size = self.factor_num * (2 ** (self.num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		# ------AMT-CDR add prediction size for ablation study------#
		if self.ablation_model == 'DMTN-No-KGL': #
			self.predict_size = self.factor_num * 2 + self.hidden_size1
		elif self.ablation_model == 'DMTN-No-SRL':
			self.predict_size = self.factor_num * 2 + self.dim
		elif self.ablation_model == 'DMTN-No-IRL':
			self.predict_size = self.hidden_size1 + self.dim
		elif self.ablation_model == 'DMTN-No-KGL-SRL':
			self.predict_size = self.factor_num * 2
		elif self.ablation_model == 'DMTN-No-KGL-IRL':
			self.predict_size = self.hidden_size1
		elif self.ablation_model == 'DMTN-No-SRL-IRL':
			self.predict_size = self.dim
		else:
			self.predict_size = self.factor_num * 2 + self.hidden_size1 + self.dim

		print("AMT-CDR KG initial--n_entity: ", self.n_entity, "--dim: ",self.dim)
		print("AMT-CDR KG initial--n_relation: ", self.n_relation, "--dim: ", self.dim * self.dim)
		self.entity_emb = nn.Embedding(self.n_entity, self.dim)
		self.relation_emb = nn.Embedding(self.n_relation, self.dim)
		self.kg_user_emb = nn.Embedding(user_num, self.dim)  # factor_num, 768

		self.query_h = nn.Linear(self.dim, self.dim * self.n_dim)  # 直接将多头的矩阵拼接了，不用拆分为几个头计算
		self.key_r = nn.Linear(self.dim, self.dim * self.n_dim)
		self.value_t = nn.Linear(self.dim, self.dim * self.n_dim)
		self.target = nn.Linear(self.dim * self.n_dim, self.dim)
		self.dropout = nn.Dropout(self.kg_dropout)
		self.kg_output_LN = LNLayer(self.dim, self.dim, 1e-12)
		self.kg_rip_LN = LNLayer(self.dim, self.dim, 1e-12)
		self.tar_item = nn.Linear(self.dim, self.dim)

		self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
		self.criterion = nn.BCELoss()
		# self.rating_loss = nn.BCEWithLogitsLoss()
		self.rating_loss = LossMSE()
		# self.rating_loss = nn.MSELoss()

		print("AMT-CDR final prediction layer--predict_size: ", self.predict_size)
		self.predict_layer = nn.Linear(self.predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
		nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_user_BERT.weight, std=0.01)
		nn.init.normal_(self.embed_item_BERT.weight, std=0.01)
		nn.init.normal_(self.entity_emb.weight, std=0.01)
		nn.init.normal_(self.relation_emb.weight, std=0.01)
		nn.init.xavier_uniform_(self.query_h.weight)
		nn.init.xavier_uniform_(self.key_r.weight)
		nn.init.xavier_uniform_(self.value_t.weight)
		nn.init.xavier_uniform_(self.target.weight)
		nn.init.xavier_uniform_(self.tar_item.weight)

		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.predict_layer.weight,
								a=1, nonlinearity='sigmoid')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()

	def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
		# base_loss = self.criterion(scores, labels.float())
		rating_loss = self.rating_loss(scores, labels)

		if self.ablation_model not in ['DMTN-No-KGL', 'DMTN-No-KGL-SRL', 'DMTN-No-KGL-IRL']:
			kge_loss = 0
			l2_loss = 0
			for hop in range(self.n_hop):
				l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
				l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
				l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
			l2_loss = self.l2_weight * l2_loss
			loss = rating_loss + kge_loss + l2_loss
		else:
			loss = rating_loss
			kge_loss = 0
			l2_loss = 0
		return dict(base_loss=rating_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

	def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, item_embeddings, user_embeddings):
		for hop in range(self.n_hop):
			h = h_emb_list[hop]	# (batch_size, n_memory, dim)
			r = r_emb_list[hop]	# (batch_size, n_memory, dim)
			t = t_emb_list[hop]	# (batch_size, n_memory, dim)
			h_R = torch.matmul(h, r.transpose(-1,-2))	# (batch_size, n_memory, n_memory)
			h_R_att = h_R / math.sqrt(self.dim)
			att_p = nn.Softmax(dim=-1)(h_R_att)
			hRt = torch.matmul(att_p, t)
			rip_hRt_emb = hRt.mean(dim=1)
			user_embeddings = self.kg_rip_LN(rip_hRt_emb, user_embeddings)
		return user_embeddings, item_embeddings

	def _update_item_embedding(self, item_embeddings, o):
		if self.item_update_mode == "replace":
			item_embeddings = o
		elif self.item_update_mode == "plus":
			item_embeddings = item_embeddings + o
		elif self.item_update_mode == "replace_transform":
			item_embeddings = self.transform_matrix(o)
		elif self.item_update_mode == "plus_transform":
			item_embeddings = self.transform_matrix(item_embeddings + o)
		else:
			raise Exception("Unknown item updating mode: " + self.item_update_mode)
		return item_embeddings

	def predict(self, item_embeddings, o_list):
		y = o_list[-1]
		if self.using_all_hops:
			for i in range(self.n_hop - 1):
				y += o_list[i]

		scores = (item_embeddings * y).sum(dim=1)
		return torch.sigmoid(scores)  # 最终映射到[0,1]之间

	def forward(self, user, item, labels, p,
        kg_memories_h: list,
        kg_memories_r: list,
        kg_memories_t: list,
		bert_train_status, sentences):
		user_sem_dom_label = None
		user_NCF_dom_label = None
		if self.ablation_model not in ['DMTN-No-IRL', 'DMTN-No-KGL-IRL', 'DMTN-No-SRL-IRL']:
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF

			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

			# AMT-CDR added GRL layer for user embeddings in NCF
			emb_user_NCF = torch.cat((embed_user_GMF, embed_user_MLP), -1)
			user_NCF_dom_label = self.user_NCF_classifier(emb_user_NCF, p)

		h_emb_list = []
		r_emb_list = []
		t_emb_list = []
		user_kg_dom_label = None
		if self.ablation_model not in ['DMTN-No-KGL', 'DMTN-No-KGL-SRL', 'DMTN-No-KGL-IRL']:
			item_embeddings = self.entity_emb(item)
			kg_user_embeddings = self.kg_user_emb(user)
			for i in range(self.n_hop):
				h_emb_list.append(self.entity_emb(kg_memories_h[i]))
				r_emb_list.append(self.relation_emb(kg_memories_r[i]))
				t_emb_list.append(self.entity_emb(kg_memories_t[i]))
			kg_user_embs, item_embeddings = self._key_addressing(
				h_emb_list, r_emb_list, t_emb_list, item_embeddings, kg_user_embeddings
			)
			user_kg_dom_label = self.user_kg_classifier(kg_user_embs, p)
			item_embeddings = self.tar_item(item_embeddings)
			users_emb_LN = self.kg_output_LN(kg_user_embeddings, kg_user_embs)
			output_KG = item_embeddings * users_emb_LN

		if self.ablation_model not in ['DMTN-No-SRL', 'DMTN-No-KGL-SRL', 'DMTN-No-SRL-IRL']:
			embed_sent_BERT = self.sen_bert.encode(sentences, convert_to_tensor=True)
			embed_user_sBERT_LN = self.output_LN(embed_sent_BERT, self.embed_user_BERT(user))
			user_sem_dom_label = self.user_semantic_classifier(embed_user_sBERT_LN, p)

			embed_item_sBERT_LN = self.output_LN(embed_sent_BERT, self.embed_item_BERT(item))
			interaction_BERT = embed_user_sBERT_LN * embed_item_sBERT_LN

		if self.ablation_model == 'DMTN-No-KGL':
			concat = torch.cat((output_GMF, output_MLP,interaction_BERT), -1)
		elif self.ablation_model == 'DMTN-No-SRL':
			concat = torch.cat((output_GMF, output_MLP, output_KG), -1)
		elif self.ablation_model == 'DMTN-No-IRL':
			concat = torch.cat((interaction_BERT, output_KG), -1)
		elif self.ablation_model == 'DMTN-No-KGL-SRL':
			concat = torch.cat((output_GMF, output_MLP), -1)
		elif self.ablation_model == 'DMTN-No-KGL-IRL':
			concat = interaction_BERT
		elif self.ablation_model == 'DMTN-No-SRL-IRL':
			concat = output_KG
		else:
			concat = torch.cat((output_GMF, output_MLP,interaction_BERT, output_KG), -1)
		prediction = self.predict_layer(concat)
		return_loss_dict = self._compute_loss(prediction.view(-1), labels, h_emb_list, t_emb_list, r_emb_list)
		return_loss_dict["scores"] = prediction.view(-1)
		return return_loss_dict, [user_sem_dom_label, user_kg_dom_label, user_NCF_dom_label]