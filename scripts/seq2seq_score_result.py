import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


df_score = pd.read_csv('data/seq_split_interdm_pred.csv')

df = pd.read_csv('data/seq_split_interdm.csv')

assert(df_score.shape[0] == df.shape[0])

df_score['inter_flag'] = df['inter_flag'].values
df_score['contact_flag'] = df['contact_flag'].values

df_score.to_csv('data/seq_split_interdm_pred_flag.csv', index=False)

# df['score'] = df_score['score'].values
# df['loss'] = df_score['loss'].values


df = pd.read_csv('seq_split_interdm_pred_flag.csv')
score = df['score'].values
loss = df['loss'].values


inter_flag = df['inter_flag'].values
contact_flat = df['contact_flag'].values

intra = (inter_flag == 0)
inter = (inter_flag == 1)

inc = (contact_flat == 0)
ic = (contact_flat == 1)


pl.figure()

pl.hist(score[intra & inc], bins=np.arange(20)*10-100, alpha=0.3)
pl.hist(score[intra & ic], bins=np.arange(20)*10-100, alpha=0.3)

pl.figure()

pl.hist(score[inter & inc], bins=np.arange(20)*10-100, alpha=0.3)
pl.hist(score[inter & ic], bins=np.arange(20)*10-100, alpha=0.3)





df = pd.read_csv('seq_split_interdm_pred.csv')
score = df['score'].values
loss = df['loss'].values

pl.hist(score, bins=np.arange(20)*10-100, alpha=0.3)

df = pd.read_csv('seq_split_interdm_pred_flag_nr90_15_sample.csv')
score = df['score'].values
loss = df['loss'].values

pl.hist(score, bins=np.arange(40)*10-100, alpha=0.3)



