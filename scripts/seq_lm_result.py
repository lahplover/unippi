import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


# df = pd.read_csv('seq_unalign_indel_all_cut_train.csv', usecols=['pfam_acc', 'seq_len'])
# a = df['pfam_acc'].value_counts()
# a.to_csv('seq_unalign_indel_all_cut_train_pfam_count.csv')


df_fam = pd.read_csv('seq_unalign_indel_all_cut_train_pfam_count.csv')
# df_fam.sort_values('fam')
fam_num_dict = {x: y for x, y in zip(df_fam['fam'], df_fam['num'])}

df_fam_id = pd.read_csv('pfamA_percentage_identity.csv')
fam_id_dict = {x: y for x, y in zip(df_fam_id['fam'], df_fam_id['percent_id'])}

df_seq = pd.read_csv('seq_unalign_indel_all_cut_dev_sample2.csv', nrows=100000)


def proc_df(df):
    grouped = df.groupby(by='fam')
    df_fam_acc = grouped.mean()
    df_fam_acc['fam'] = df_fam_acc.index
    df_fam_acc.index = np.arange(df_fam_acc.shape[0])

    fam_num = [fam_num_dict[x] for x in df_fam_acc['fam']]
    df_fam_acc['num'] = fam_num
    pcid = [fam_id_dict[x] for x in df_fam_acc['fam']]
    df_fam_acc['pcid'] = pcid

    df_fam_acc['log_num'] = np.log10(df_fam_acc['num'])

    # df_fam_acc.to_csv('seq_lm_pred_fam_acc.csv', index=False)
    return df_fam_acc



def plot_df(df_fam_acc):
    import seaborn as sns
    sns.set()
    sns.scatterplot(x="log_num", y="acc", hue="pcid", data=df_fam_acc)
    # sns.scatterplot(x="log_num", y="acc", hue="pcid", marker='.', data=df_fam_acc)
    pl.xlabel('log(num_seq_in_family)')
    pl.ylabel('accuracy')

    # fam = df_fam_acc['fam'].values
    # num = df_fam_acc['num'].values
    # acc = df_fam_acc['acc'].values
    # pcid = df_fam_acc['pcid'].values
    #
    # pl.figure()
    # pl.plot(df_fam_acc['num'], df_fam_acc['acc'], 'b.')
    # pl.xscale('log')
    #
    # bins = [1, 10, 100, 1000, 10000, 100000]
    # acc_bin = []
    # for i in range(len(bins)-1):
    #     ind = (num > bins[i]) & (num < bins[i+1])
    #     acc_bin.append(acc[ind].mean())
    # acc_bin = np.array(acc_bin)
    # bins = np.array([5, 50, 500, 5000, 50000])
    #
    # pl.plot(bins, acc_bin, 'ro')
    # pl.plot([1, 100000], [0.5, 0.5])
    #
    # ind = (acc > 0.8)
    # fam_good = df_fam_acc['fam'][ind].values


# read prediction result from seq_lm model
df = pd.read_csv('seq_lm_prediction.txt')
df['fam'] = df_seq['pfam_acc']


def count_acc(target, pred):
    acc = 0
    if len(target) != len(pred):
        return -1
    for i in range(len(target)):
        if target[i] == pred[i]:
            acc += 1
    return 1.0 * acc / len(target)


acc = []
for target, pred in zip(df['target'], df['pred']):
    acc.append(count_acc(target, pred))

df['acc'] = acc
ind = (df['acc'] > 0)
df = df[ind]

df_fam_acc = proc_df(df)
plot_df(df_fam_acc)


# for kmer model

df = pd.read_csv('kmer_pred_acc.csv')
df['fam'] = df_seq['pfam_acc']

df_fam_acc = proc_df(df)
plot_df(df_fam_acc)





# seq shift

df = pd.read_csv('seq_unalign_indel_all_cut_dev_sample2.csv', nrows=100000)
seq = df['seq_unalign'].values
seq_shifted = []
for s in seq:
    i = np.random.randint(0, int(len(s)/2))
    seq_shifted.append(s[i:])

df2 = pd.DataFrame({'seq_unalign': seq_shifted, 'fam': df['pfam_acc']})
df2['seq_len'] = df2['seq_unalign'].apply(lambda x: len(x))

df2.to_csv('seq_unalign_indel_all_cut_dev_sample2_shifted.csv', index=False)






