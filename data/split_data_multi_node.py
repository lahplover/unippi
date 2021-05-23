import pandas as pd
import numpy as np


df = pd.read_csv('seq_split_train_shuffle.csv')

step = int(df.shape[0] / 16) + 1
for i in range(16):
    df2 = df[i*step:(i+1)*step]
    df2.to_csv(f'multinode/seq_split_train_shuffle_{i}.csv', index=False)

step = int(df.shape[0] / 16) + 1
for i in range(16):
    df2 = df[i*step:(i+1)*step]
    df2.to_csv(f'multinode/seq_split_nr90_train_{i}.csv', index=False)


df = pd.read_csv('seq_unalign_indel_all_cut_train.csv')
df = df.sample(frac=1.0)

step = int(df.shape[0] / 16) + 1
for i in range(16):
    df2 = df[i*step:(i+1)*step]
    df2.to_csv(f'multinode/seq_unalign_indel_all_cut_train_{i}.csv', index=False)


df = pd.read_csv('uniref50.csv')

df = df.sample(frac=1.0)
df.index = np.arange(df.shape[0])

n = int(df.shape[0]*0.7)

df_train = df[:n]

df_test = df[n:]
df_test.index = np.arange(df_test.shape[0])

df_test.to_csv('uniref50_test.csv', index=False)

step = int(df_train.shape[0] / 17) + 1
for i in range(17):
    df2 = df[i*step:(i+1)*step]
    df2.to_csv(f'../unippi/data/multinode/uniref50_train_shuffle_{i}.csv', index=False)


###
for i in range(16):
    df = pd.read_csv(f'uniref50_train_shuffle_{i}.csv')
    df2 = df[(df['seq_len'] <= 256)]
    df2 = df2.sample(frac=1.0)
    df2.index = np.arange(df2.shape[0])
    df2.to_csv(f'uniref50_train_shuffle_{i}_0-256.csv', index=False)

    df2 = df[(df['seq_len'] > 256) & (df['seq_len'] <= 512)]
    df2 = df2.sample(frac=1.0)
    df2.index = np.arange(df2.shape[0])
    df2.to_csv(f'uniref50_train_shuffle_{i}_256-512.csv', index=False)

    df2 = df[(df['seq_len'] > 512) & (df['seq_len'] <= 1000)]
    df2 = df2.sample(frac=1.0)
    df2.index = np.arange(df2.shape[0])
    df2.to_csv(f'uniref50_train_shuffle_{i}_512-1000.csv', index=False)

    print(i)


for i in range(16):
    df = pd.read_csv(f'uniref50_train_shuffle_{i}.csv')
    df2 = df[(df['seq_len'] <= 256)]
    df2 = df2.sample(frac=0.001)
    df2.index = np.arange(df2.shape[0])
    df2.to_csv(f'test_uniref50_train_shuffle_{i}_0-256.csv', index=False)

    df2 = df[(df['seq_len'] > 256) & (df['seq_len'] <= 512)]
    df2 = df2.sample(frac=0.001)
    df2.index = np.arange(df2.shape[0])
    df2.to_csv(f'test_uniref50_train_shuffle_{i}_256-512.csv', index=False)

    df2 = df[(df['seq_len'] > 512) & (df['seq_len'] <= 1000)]
    df2 = df2.sample(frac=0.001)
    df2.index = np.arange(df2.shape[0])
    df2.to_csv(f'test_uniref50_train_shuffle_{i}_512-1000.csv', index=False)

    print(i)


for i in range(16):
    df1 = pd.read_csv(f'uniref50_train_shuffle_{i}_0-256.csv')
    df2 = pd.read_csv(f'uniref50_train_shuffle_{i}_256-512.csv')
    df3 = pd.read_csv(f'uniref50_train_shuffle_{i}_512-1000.csv')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df.to_csv(f'uniref50_train_shuffle_{i}_iter.csv', index=False)

    print(i)


