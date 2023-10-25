# AMT-CDR: A Deep Adversarial Multi-channel Transfer Network for Cross-domain Recommendation
Souce code and data of AMT-CDR

## Data:
We shared our data through the following link:
https://drive.google.com/drive/folders/1yDGRGuszbEK2TnVxydGDlbJBWMh4Iqpx?usp=sharing

## Train and evaluate AMT-CDR**

- nohup python3 main_SenBert_DMTN_Book2Movie_ablation.py --batch_size=8 --lr=0.0001 --factor_num=16 --dim=32 --n_dim=12 --n_hop=2 --dropout=0 --n_memory=32 --ablation_model=DMTN --model=DMTN_SenBert_LN_TransForKG_3GAN --data_version=V1_sent_T2 > log_Book2Movie.txt 2>&1 &
.<br>

- nohup python3 main_SenBert_DMTN_Book2CD_ablation.py --batch_size=8 --lr=0.0001 --factor_num=16 --dim=32 --n_dim=12 --n_hop=2 --dropout=0 --n_memory=32 --ablation_model=DMTN --model=DMTN_SenBert_LN_TransForKG_3GAN --data_version=V1_sent_T2 > log_Book2CD.txt 2>&1 &
.<br>

- nohup python3 main_SenBert_DMTN_Movie2CD_ablation.py --batch_size=8 --lr=0.0001 --factor_num=16 --dim=32 --n_dim=12 --n_hop=2 --dropout=0 --n_memory=32 --ablation_model=DMTN --model=DMTN_SenBert_LN_TransForKG_3GAN --data_version=V1_sent_T2 > log_Movie2CD.txt 2>&1 &
.<br>

