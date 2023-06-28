"""
@Paper: AMT-CDR: A Deep Adversarial Multi-channel Transfer Network for Cross-domain Recommendation
@author: Kezhi Lu, Qian Zhang
@time: June 20th, 2023
"""

# ------model name ------#
model = 'DMTN_SenBert_LN_TransForKG_3GAN'
# assert model in ['BertKG_NCF', 'DMTN_BertAvePool', 'DMTN_BertAvePool_DotPro', 'DMTN_BertAvePool_LN']

# ------------------------------Please choose different data path depending on the CDR-Task--------------------------------

# AMT-CDR Book --> Movie
# source_base_path = 'Data/KB-Am-FinalData-output/AmBook/FinalDataset/'
# target_base_path = 'Data/KB-Am-FinalData-output/AmMovie/FinalDataset/'
# kg_base_path = 'Data/KB-Am-FinalData-output/finalKB/'
# source_train_rating_path = source_base_path + 'source_train_ratings.tsv'
# source_train_review_path = source_base_path + 'source_train_reviews.tsv'
# source_vali_rating_path = source_base_path + 'source_vali_ratings.tsv'
# source_test_rating_path = source_base_path + 'source_test_ratings.tsv'
#
# target_train_rating_path = target_base_path + 'target_train_ratings.tsv'
# target_train_review_path = target_base_path + 'target_train_reviews.tsv'
# target_vali_rating_path = target_base_path + 'target_vali_ratings.tsv'
# target_test_rating_path = target_base_path + 'target_test_ratings.tsv'
# kg_triples = kg_base_path + 'finalKB_fromFB+YAGO_format.tsv'

# # AMT-CDR Book --> CD
# source_base_path = 'Data/KB-Am-FinalData-output/AmBook/FinalDataset/'
# target_base_path = 'Data/KB-Am-FinalData-output/AmCDs_LFM1b/FinalDataset/'
# kg_base_path = 'Data/KB-Am-FinalData-output/finalKB/'
# source_train_rating_path = source_base_path + 'source_train_ratings.tsv'
# source_train_review_path = source_base_path + 'source_train_reviews.tsv'
# source_vali_rating_path = source_base_path + 'source_vali_ratings.tsv'
# source_test_rating_path = source_base_path + 'source_test_ratings.tsv'
#
# target_train_rating_path = target_base_path + 'target_train_ratings.tsv'
# target_train_review_path = target_base_path + 'target_train_reviews.tsv'
# target_vali_rating_path = target_base_path + 'target_vali_ratings.tsv'
# target_test_rating_path = target_base_path + 'target_test_ratings.tsv'
# kg_triples = kg_base_path + 'finalKB_fromFB+YAGO_format.tsv'

# AMT-CDR Movie --> CD
source_base_path = '../Data/KB-Am-FinalData-output/AmMovie/FinalDataset/'
target_base_path = '../Data/KB-Am-FinalData-output/AmCDs_LFM1b/FinalDataset/'
kg_base_path = '../Data/KB-Am-FinalData-output/finalKB/'
source_train_rating_path = source_base_path + 'target_train_ratings.tsv'
source_train_review_path = source_base_path + 'target_train_reviews.tsv'
source_vali_rating_path = source_base_path + 'target_vali_ratings.tsv'
source_test_rating_path = source_base_path + 'target_test_ratings.tsv'

target_train_rating_path = target_base_path + 'target_train_ratings.tsv'
target_train_review_path = target_base_path + 'target_train_reviews.tsv'
target_vali_rating_path = target_base_path + 'target_vali_ratings.tsv'
target_test_rating_path = target_base_path + 'target_test_ratings.tsv'
kg_triples = kg_base_path + 'finalKB_fromFB+YAGO_format.tsv'