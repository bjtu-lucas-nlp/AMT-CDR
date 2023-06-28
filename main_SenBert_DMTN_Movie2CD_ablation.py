"""
@Paper: AMT-CDR: A Deep Adversarial Multi-channel Transfer Network for Cross-domain Recommendation
@author: Kezhi Lu, Qian Zhang
@time: June 20th, 2023
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import model_DMTN_SenBertAvePool_LN_GAN
import config
import eval_BertKG_NCF_CDR
import data_utils_BertKG_NCF_CDR
from transformers import BertConfig
import statistics
#---------------------------- PREPARE PARAMETERS ----------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
parser.add_argument("--num_ng", type=int, default = 4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--n_dim', type=int, default=8, help='number of dimension for entity and relation in Transformer')
parser.add_argument('--kg_dropout', type=float, default=0.1, help="dropout ratio for the kg layer")
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True, help='whether using outputs of all hops or just the last hop when making prediction')
#---------------------------- PREPARE PARAMETERS for ABLATION STUDY----------------------------#
# 1: DMTN-No-KGL: DMTN without KG; 2: DMTN-No-SRL: DMTN without semantic Bert; 3: DMTN-No-IRL: DMTN without NCF;
# 4: DMTN-No-KGL-SRL: DMTN without KG and Bert; 5: DMTN-No-KGL-IRL: DMTN without KG and NCF; 6: DMTN-No-SRL-IRL: DMTN without BERT and NCF;
parser.add_argument('--ablation_model', type=str, default='DMTN', help='select model for normal experiment-DMTN or ablation study')
parser.add_argument('--model', type=str, default='DMTN_SenBert_LN_TransForKG_3GAN', help='select model for the experiment')
parser.add_argument('--data_version', type=str, default='V1', help='select dataset version for the experiment')
args = parser.parse_args()

#---------------------------- CHOOSE GPU DEVICES----------------------------#
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("AMT-CDR--cuda infor: count: ",torch.cuda.device_count(),"----")
device_ids = [0]
cudnn.benchmark = True

#------Function: list to torch longtensor------#
def list_2_npLongTensor(cur_list):
	return_list = []
	for i_list in cur_list:
		return_list.append([item.numpy() for item in i_list])
	return torch.LongTensor(return_list)

#------Function: get system time------#
def get_time():
	cur_time = time.asctime(time.localtime(time.time()))
	return cur_time

#------Function: get users' n_hop ripples------#
def get_batchUsers_kg_ripples(args, batch_users, ripple_set_source, ripple_set_target = None, device_id = 0):
	# users = torch.LongTensor(batch_users)
	# items = torch.LongTensor(batch_items)
	# labels = torch.LongTensor(batch_labels)
	memories_h, memories_r, memories_t = [], [], []
	# memories_h: (n_hops, batch_size--users, n_memory--head_items)
	for i in range(args.n_hop):
		if ripple_set_target == None:
			memories_h.append(torch.LongTensor([ripple_set_source[int(user)][i][0] for user in batch_users]))
			memories_r.append(torch.LongTensor([ripple_set_source[int(user)][i][1] for user in batch_users]))
			memories_t.append(torch.LongTensor([ripple_set_source[int(user)][i][2] for user in batch_users]))
		else:
			h, r, t = [], [], []
			for user in batch_users:
				if int(user) in ripple_set_source:
					h.append(ripple_set_source[int(user)][i][0])
					r.append(ripple_set_source[int(user)][i][1])
					t.append(ripple_set_source[int(user)][i][2])
				else:
					h.append(ripple_set_target[int(user)][i][0])
					r.append(ripple_set_target[int(user)][i][1])
					t.append(ripple_set_target[int(user)][i][2])
			memories_h.append(torch.LongTensor(h))
			memories_r.append(torch.LongTensor(r))
			memories_t.append(torch.LongTensor(t))
	memories_h = list(map(lambda x: x.cuda(device=device_id), memories_h))
	memories_r = list(map(lambda x: x.cuda(device=device_id), memories_r))
	memories_t = list(map(lambda x: x.cuda(device=device_id), memories_t))
	# print("--get_batchUsers_kg_ripples--to cuda finished--")
	return memories_h, memories_r, memories_t

#---------------------------- Main Function ----------------------------#
if __name__=="__main__":
	# ------PREPARE DATASET------#
	source_data_list, target_data_list, kg_data_list = torch.load("Data/formatData_Movie2CD_SenBert_V1T2.pt")

	# ------PREPARE dataloader for train, vali, test datasets------#
	train_dataset = data_utils_BertKG_NCF_CDR.DMTN_Data(source_data_list, target_data_list, "Train", "source_target", args.data_version)
	print("batch_size: ", args.batch_size, " num_ng: ", args.num_ng, " args.test_num_ng: ", args.test_num_ng,
		  " factor_num: ", args.factor_num, " num_layers: ", args.num_layers, " epochs: ", args.epochs)
	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

	vali_source_dataset = data_utils_BertKG_NCF_CDR.DMTN_Data(
		source_data_list, target_data_list, "Vali", "Source", args.data_version)
	vali_source_loader = data.DataLoader(vali_source_dataset,
										 batch_size=args.batch_size, shuffle=False, num_workers=0)

	vali_target_dataset = data_utils_BertKG_NCF_CDR.DMTN_Data(
		source_data_list, target_data_list, "Vali", "Target", args.data_version)
	vali_target_loader = data.DataLoader(vali_target_dataset,
										 batch_size=args.batch_size, shuffle=False, num_workers=0)

	test_source_dataset = data_utils_BertKG_NCF_CDR.DMTN_Data(
		source_data_list, target_data_list, "Test", "Source", args.data_version)
	test_source_loader = data.DataLoader(test_source_dataset,
										 batch_size=args.batch_size, shuffle=False, num_workers=0)

	test_target_dataset = data_utils_BertKG_NCF_CDR.DMTN_Data(
		source_data_list, target_data_list, "Test", "Target", args.data_version)
	test_target_loader = data.DataLoader(test_target_dataset,
										 batch_size=args.batch_size, shuffle=False, num_workers=0)

	# ------statistics for user, item, kg entity, kg relation------#
	user_num = source_data_list[0].shape[0]
	item_num = source_data_list[0].shape[1] + target_data_list[0].shape[1]
	kg_entity_num = kg_data_list[0]
	kg_relation_num = kg_data_list[1]
	ripple_set = kg_data_list[2]
	ripple_set_source = kg_data_list[3]
	ripple_set_target = kg_data_list[4]
	print("AMT-CDR --item num: ",item_num, "--kg_entity_num: ",kg_entity_num, "--kg_relation_num: ",kg_relation_num)

	# ----------------------------- CREATE MODEL ----------------------------------
	rmse_source_10, rmse_target_10, mae_source_10, mae_target_10 = [], [], [], []
	active_domain_loss_step = 0
	num_batch = len(train_loader)
	dann_epoch = np.floor(active_domain_loss_step / num_batch * 1.0)
	# ------loss for GRLayer------#
	loss_adver = nn.CrossEntropyLoss().cuda(device=device_ids[0])
	K_folds = 10
	for ii in range(K_folds):
		GMF_model = None
		MLP_model = None
		model = model_DMTN_SenBertAvePool_LN_GAN.DMTN_SenBert_LN_TransForKG_3GAN(user_num, item_num, args, kg_relation_num,
																			 GMF_model, MLP_model, device_ids[0])
		model.cuda(device=device_ids[0])
		# optimizer = optim.SGD(model.parameters(), lr=args.lr)
		optimizer = optim.Adam(model.parameters(), lr=args.lr)

		# ------------------------------TRAINING --------------------------------
		count, best_hr = 0, 0
		epoch_int = 0
		min_test = [0, 0, 0, 0]
		min_rmse_vali_source = 100
		min_rmse_vali_target = 100
		min_mae_vali_source = 100
		min_mae_vali_target = 100
		best_source_epoch = 0
		best_target_epoch = 0
		print("Start training: ",get_time())
		for epoch in range(args.epochs):
			model.train()
			start_time = time.time()
			# train_loader.dataset.ng_sample()
			print("Epoch: ", epoch_int)
			print("Cur time: ", get_time())
			epoch_int += 1
			batch_int = 0
			for user, item, label, review, domain_labels in train_loader:
				# ------train data for knowledge graph------#
				batch_int += 1
				print("Epoch: ",str(epoch_int)," Batch: ",str(batch_int))
				memories_h, memories_r, memories_t = get_batchUsers_kg_ripples(args, user, ripple_set, None,device_ids[0])
				user = user.cuda(device=device_ids[0])
				item = item.cuda(device=device_ids[0])
				labels = label.float().cuda(device=device_ids[0])
				domain_labels = torch.LongTensor(domain_labels).cuda(device = device_ids[0])

				model.zero_grad()
				p = float(batch_int + (epoch - dann_epoch) * num_batch / (args.epochs - dann_epoch) / num_batch)
				p = 2. / (1. + np.exp(-10 * p)) - 1

				# ------training and return loss------#
				return_loss_dict, [user_sem_dom_label, user_kg_dom_label, user_NCF_dom_label] = model(user, item, labels, p,
																				  memories_h, memories_r,
																				  memories_t, True, review)
				loss = return_loss_dict["loss"]
				# ------Classification loss during adversarial training------#
				if batch_int > active_domain_loss_step + 1:
					if user_sem_dom_label is not None:
						loss_user_sem_dom = loss_adver(user_sem_dom_label, domain_labels)
						loss += loss_user_sem_dom
					if user_kg_dom_label is not None:
						loss_user_kg_dom = loss_adver(user_kg_dom_label, domain_labels)
						loss += loss_user_kg_dom
					if user_NCF_dom_label is not None:
						loss_user_NCF_dom = loss_adver(user_NCF_dom_label, domain_labels)
						loss += loss_user_NCF_dom
				print("-------AMT-CDR----loss: ", loss)
				loss.backward()
				optimizer.step()
				count += 1
			# ------Evaluation on Vali and Test------#
			model.eval()
			RMSE_source_vali_epoch, MAE_source_vali_epoch, s_vali_dom_cla_loss = eval_BertKG_NCF_CDR.evaluation_v3(args, model, vali_source_loader, ripple_set, set(), device_ids[0], p, args.data_version)
			RMSE_target_vali_epoch, MAE_target_vali_epoch, t_vali_dom_cla_loss = eval_BertKG_NCF_CDR.evaluation_v3(args, model, vali_target_loader, ripple_set, set(), device_ids[0], p, args.data_version)
			RMSE_source_test_epoch, MAE_source_test_epoch, s_test_dom_cla_loss = eval_BertKG_NCF_CDR.evaluation_v3(args, model, test_source_loader, ripple_set, set(), device_ids[0], p, args.data_version)
			RMSE_target_test_epoch, MAE_target_test_epoch, t_test_dom_cla_loss = eval_BertKG_NCF_CDR.evaluation_v3(args, model, test_target_loader, ripple_set, set(), device_ids[0], p, args.data_version)
			if epoch == 1:
				min_test = [RMSE_target_test_epoch, MAE_target_test_epoch, RMSE_source_test_epoch, MAE_source_test_epoch]

			if MAE_source_vali_epoch <= min_mae_vali_source:
				best_source_epoch = epoch_int
				min_mae_vali_source = MAE_source_vali_epoch
				min_rmse_vali_source = RMSE_source_vali_epoch
				min_test[2], min_test[3] = RMSE_source_test_epoch, MAE_source_test_epoch
			if MAE_target_vali_epoch <= min_mae_vali_target:
				best_target_epoch = epoch_int
				min_mae_vali_target = MAE_target_vali_epoch
				min_test[0], min_test[1] = RMSE_target_test_epoch, MAE_target_test_epoch

			if MAE_source_vali_epoch > min_mae_vali_source and MAE_target_vali_epoch > min_mae_vali_target:
				print(
					'Iter %d, Para: kge_weight:%f, l2_weight:%f, batch_size: %d, learning rate: %f. \n Target RMSE: %f, MAE: %f.\t Source RMSE: %f, MAE: %f.' % \
					(ii + 1, args.kge_weight, args.l2_weight, args.batch_size, args.lr, min_test[0], min_test[1], min_test[2], min_test[3]))
				rmse_target_10.append(min_test[0])
				mae_target_10.append(min_test[1])
				rmse_source_10.append(min_test[2])
				mae_source_10.append(min_test[3])
				break

			if epoch == args.epochs - 1:
				print('Epoch finished.')
				print(
					'Para: kge_weight: %f, l2_weight:%f, batch_size: %d, learning rate: %f. \n Target RMSE: %f, MAE: %f.\t Source RMSE: %f, MAE: %f.' % \
					(args.kge_weight, args.l2_weight, args.batch_size, args.lr, RMSE_target_test_epoch, MAE_target_test_epoch, RMSE_source_test_epoch,
					 MAE_source_test_epoch))
				rmse_target_10.append(RMSE_target_test_epoch)
				mae_target_10.append(MAE_target_test_epoch)
				rmse_source_10.append(RMSE_source_test_epoch)
				mae_source_10.append(MAE_source_test_epoch)
			elapsed_time = time.time() - start_time
			print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
					time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

		print("End. Best Source epoch {:04d}: RMSE source test = {:.4f}, MAE source test = {:.4f}".format(best_source_epoch, min_test[2], min_test[3]))
		print("End. Best Target epoch {:04d}: RMSE target test = {:.4f}, MAE target test = {:.4f}".format(best_target_epoch, min_test[0], min_test[1]))

	# ------Save statistic results------#
	torch.save(model.state_dict(), "Data/save_DMTN_epoch_Movie2CD_" + str(args.lr) + str(args.model) + args.ablation_model + args.data_version + '.pth')

	target_mae_fin_mean = statistics.mean(mae_target_10)
	target_mae_fin_std = statistics.stdev(mae_target_10)
	target_rmse_fin_mean = statistics.mean(rmse_target_10)
	target_rmse_fin_std = statistics.stdev(rmse_target_10)
	print('FINAL Target MAE: mean: %f, std: %f, RMSE: mean: %f, std: %f ' % (target_mae_fin_mean, target_mae_fin_std, target_rmse_fin_mean, target_rmse_fin_std))

	source_mae_fin_mean = statistics.mean(mae_source_10)
	source_mae_fin_std = statistics.stdev(mae_source_10)
	source_rmse_fin_mean = statistics.mean(rmse_source_10)
	source_rmse_fin_std = statistics.stdev(rmse_source_10)
	print('FINAL Source MAE: mean: %f, std: %f, RMSE: mean: %f, std: %f ' % (source_mae_fin_mean, source_mae_fin_std, source_rmse_fin_mean, source_rmse_fin_std))

	with open("Data/save_DMTN_epoch_Movie2CD_" + str(args.lr) + '_' +str(args.batch_size) + str(args.model) + args.ablation_model + args.data_version +'.txt', 'w') as f:
		for ii in range(K_folds):
			f.write(str(rmse_target_10[ii]) + '\t' + str(mae_target_10[ii]) + '\t' + str(rmse_source_10[ii]) + '\t' + str(mae_source_10[ii]) + str(args.model) + args.ablation_model + '\n')

