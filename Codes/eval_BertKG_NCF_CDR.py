"""
@Paper: AMT-CDR: A Deep Adversarial Multi-channel Transfer Network for Cross-domain Recommendation
@author: Kezhi Lu, Qian Zhang
@time: June 20th, 2023
"""
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import CrossEntropyLoss

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def metrics(model, test_loader, top_k, device_ids):
	HR, NDCG = [], []
	for user, item, label in test_loader:
		user = user.cuda(device=device_ids[0])
		item = item.cuda(device=device_ids[0])

		predictions = model(user, item, False)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)

def get_batchUsers_kg_ripples(args, batch_users, ripple_set_source, ripple_set_target, device_id):
	memories_h, memories_r, memories_t = [], [], []
	for i in range(args.n_hop):
		h, r, t = [], [], []
		for user in batch_users:
			if int(user) in ripple_set_source:
				h.append(ripple_set_source[int(user)][i][0])
				r.append(ripple_set_source[int(user)][i][1])
				t.append(ripple_set_source[int(user)][i][2])
			else:
				if len(ripple_set_target) !=0:
					h.append(ripple_set_target[int(user)][i][0])
					r.append(ripple_set_target[int(user)][i][1])
					t.append(ripple_set_target[int(user)][i][2])
		memories_h.append(torch.LongTensor(h))
		memories_r.append(torch.LongTensor(r))
		memories_t.append(torch.LongTensor(t))

	memories_h = list(map(lambda x: x.cuda(device=device_id), memories_h))
	memories_r = list(map(lambda x: x.cuda(device=device_id), memories_r))
	memories_t = list(map(lambda x: x.cuda(device=device_id), memories_t))
	return memories_h, memories_r, memories_t

class LossMSE(nn.Module):
	def __init__(self):
		super(LossMSE, self).__init__()

	def forward(self, pred, real):
		pred_loss=pred.clone()
		pred_loss[real==0]=0
		diffs = torch.add(real, -pred_loss)
		n = len(torch.nonzero(real))
		mse = torch.sum(diffs.pow(2)) / n
		return mse

class LossOrth(nn.Module):
	def __init__(self):
		super(LossOrth, self).__init__()

	def forward(self, input1, input2):
		batch_size = input1.size(0)
		input1 = input1.view(batch_size, -1)
		input2 = input2.view(batch_size, -1)
		input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
		input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
		input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
		input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
		diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
		return diff_loss

class LossMAE(nn.Module):
	def __init__(self):
		super(LossMAE, self).__init__()

	def forward(self, pred, real):
		pred_loss=pred.clone()
		pred_loss[real==0]=0
		diffs = torch.add(real, -pred_loss)
		n = len(torch.nonzero(real))
		mae = torch.sum(torch.abs(diffs)) / n
		return mae

def evaluation(args, model, test_loader, ripple_set, device_id):
	rmse_vali_epoch = 0
	mae_vali_epoch = 0
	loss_recon = LossMSE().to(device_id)
	loss_mae = LossMAE().to(device_id)
	with torch.no_grad():
		for user, item, label in test_loader:
			user = user.cuda(device=device_id)
			item = item.cuda(device=device_id)
			label = label.cuda(device=device_id)
			memories_h, memories_r, memories_t = get_batchUsers_kg_ripples(args, user, ripple_set, device_id)
			return_loss_dict = model(user, item, label, memories_h, memories_r, memories_t, False)
			predict_score = return_loss_dict["scores"]
			predict_score_re = torch.clamp(predict_score, 1, 5)
			loss_target_vali = loss_recon(predict_score_re, label)
			loss_target_mae = loss_mae(predict_score_re, label)
			rmse_vali_epoch += loss_target_vali.item()
			mae_vali_epoch += loss_target_mae.item()
		rmse_vali_epoch = math.sqrt(rmse_vali_epoch / len(test_loader))
		mae_vali_epoch = mae_vali_epoch / len(test_loader)
	return rmse_vali_epoch, mae_vali_epoch

def evaluation_V2(args, model, test_loader, ripple_set, device_id):
	rmse_vali_epoch = 0
	mae_vali_epoch = 0
	loss_recon = LossMSE().to(device_id)
	loss_mae = LossMAE().to(device_id)
	predict_result = torch.FloatTensor().to(device_id)
	label_result = torch.FloatTensor().to(device_id)
	with torch.no_grad():
		for user, item, label in test_loader:
			user = user.cuda(device=device_id)
			item = item.cuda(device=device_id)
			label = label.cuda(device=device_id)
			memories_h, memories_r, memories_t = get_batchUsers_kg_ripples(args, user, ripple_set, device_id)
			return_loss_dict = model(user, item, label, memories_h, memories_r, memories_t, False)
			predict_score = return_loss_dict["scores"]
			for i in range(len(predict_score)):
				if predict_score[i] < 1:
					predict_score[i] = 1
				elif predict_score[i] > 5:
					predict_score[i] = 5
			loss_target_vali = loss_recon(predict_score, label)
			loss_target_mae = loss_mae(predict_score, label)
			predict_result = torch.cat((predict_score,predict_result),0)
			label_result = torch.cat((label, label_result),0)

			rmse_vali_epoch += loss_target_vali.item()
			mae_vali_epoch += loss_target_mae.item()
		rmse_vali_epoch = math.sqrt(rmse_vali_epoch / len(test_loader))
		mae_vali_epoch = mae_vali_epoch / len(test_loader)
	return rmse_vali_epoch, mae_vali_epoch, predict_result, label_result

def evaluation_v3(args, model, test_loader, ripple_set_source, ripple_set_target, device_id, p, data_version = "V1"):
	rmse_vali_epoch = 0
	mae_vali_epoch = 0
	loss_recon = LossMSE().to(device_id)
	loss_mae = LossMAE().to(device_id)
	domain_sem_clssifier_loss = 0
	domain_kg_classifier_loss = 0
	domain_ncf_classifier_loss = 0
	dom_classifier_loss = 0
	domain_loss = CrossEntropyLoss().to(device_id)
	with torch.no_grad():
		if data_version in ["V1_sent","V1_sent_T2"]:
			for user, item, label, review, domain_labels in test_loader:
				user = user.cuda(device=device_id)
				item = item.cuda(device=device_id)
				label = label.cuda(device=device_id)
				domain_labels = domain_labels.cuda(device=device_id)
				if args.data_version == "V1":
					memories_h, memories_r, memories_t = get_batchUsers_kg_ripples(args, user, ripple_set_source,
																				   ripple_set_target,
																				   device_id)
				else:
					memories_h, memories_r, memories_t = get_batchUsers_kg_ripples(args, user, ripple_set_source,
																				   ripple_set_target, device_id)
				return_loss_dict, [user_sem_dom_label, user_kg_dom_label, user_NCF_dom_label] = model(user, item,
																									  label, p,
																									  memories_h,
																									  memories_r,
																									  memories_t,
																									  False, review)
				predict_score = return_loss_dict["scores"]
				predict_score_re = torch.clamp(predict_score, 1, 5)
				loss_target_vali = loss_recon(predict_score_re, label)
				loss_target_mae = loss_mae(predict_score_re, label)
				rmse_vali_epoch += loss_target_vali.item()
				mae_vali_epoch += loss_target_mae.item()
				if args.ablation_model not in ['DMTN-No-SRL', 'DMTN-No-KGL-SRL', 'DMTN-No-SRL-IRL']:
					domain_sem_clssifier_loss += domain_loss(user_sem_dom_label, domain_labels).item()
				if args.model in ["DMTN_SenBert_LN_KG_2GAN", "DMTN_SenBert_LN_TransForKG_2GAN"]:
					if args.ablation_model not in ['DMTN-No-KGL', 'DMTN-No-KGL-SRL', 'DMTN-No-KGL-IRL']:
						domain_kg_classifier_loss += domain_loss(user_kg_dom_label, domain_labels).item()
				elif args.model in ["DMTN_SenBert_LN_TransForKG_3GAN"]:
					if args.ablation_model not in ['DMTN-No-IRL', 'DMTN-No-KGL-IRL', 'DMTN-No-SRL-IRL']:
						domain_ncf_classifier_loss += domain_loss(user_NCF_dom_label, domain_labels).item()
		else:
			for user, item, label, domain_labels in test_loader:
				user = user.cuda(device=device_id)
				item = item.cuda(device=device_id)
				label = label.cuda(device=device_id)
				domain_labels = domain_labels.cuda(device = device_id)
				memories_h, memories_r, memories_t = get_batchUsers_kg_ripples(args, user, ripple_set_source, ripple_set_target, device_id)
				return_loss_dict, [user_sem_dom_label] = model(user, item, label, p, memories_h, memories_r, memories_t, False)
				predict_score = return_loss_dict["scores"]
				predict_score_re = torch.clamp(predict_score, 1, 5)
				loss_target_vali = loss_recon(predict_score_re, label)
				loss_target_mae = loss_mae(predict_score_re, label)
				rmse_vali_epoch += loss_target_vali.item()
				mae_vali_epoch += loss_target_mae.item()
				if args.ablation_model not in ['DMTN-No-SRL', 'DMTN-No-KGL-SRL', 'DMTN-No-SRL-IRL']:
					domain_sem_clssifier_loss += domain_loss(user_sem_dom_label, domain_labels).item()

		rmse_vali_epoch = math.sqrt(rmse_vali_epoch / len(test_loader))
		mae_vali_epoch = mae_vali_epoch / len(test_loader)
		if args.ablation_model not in ['DMTN-No-SRL', 'DMTN-No-KGL-SRL', 'DMTN-No-SRL-IRL']:
			dom_classifier_loss += domain_sem_clssifier_loss / len(test_loader)
		if args.ablation_model not in ['DMTN-No-KGL', 'DMTN-No-KGL-SRL', 'DMTN-No-KGL-IRL']:
			if args.model in ["DMTN_SenBert_LN_KG_2GAN", "DMTN_SenBert_LN_TransForKG_2GAN"]:
				dom_classifier_loss += domain_kg_classifier_loss / len(test_loader)
		if args.ablation_model not in ['DMTN-No-IRL', 'DMTN-No-KGL-IRL', 'DMTN-No-SRL-IRL']:
			if args.model in ['DMTN_SenBert_LN_TransForKG_3GAN']:
				dom_classifier_loss += domain_ncf_classifier_loss / len(test_loader)
	return rmse_vali_epoch, mae_vali_epoch, dom_classifier_loss