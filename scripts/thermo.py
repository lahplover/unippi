import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


df_fam = pd.read_csv('seq_unalign_indel_all_cut_train_pfam_count.csv')
fam_num_dict = {x: y for x, y in zip(df_fam['fam'], df_fam['num'])}


# df = pd.read_csv('stab_thermo.csv')
df = pd.read_csv('pdb_pfam_mapping_thermo.csv')
fam = df['PFAM_ACC'].apply(lambda x: x.split('.')[0]).values
num = []
for x in fam:
    try:
        num.append(fam_num_dict[x])
    except KeyError:
        num.append(0)
num = np.array(num)
df['num'] = num
df['fam'] = fam

ind = (num > 50000)

pl.hist(num[ind], bins=np.arange(100)*10000)


# load thermo pdb data
df = pd.read_csv('thermo_pdb_seq.csv')

df_thermo = pd.read_csv('thermo_pfam_pdb.csv')
ind = (df_thermo['fam'] == 'PF00072')
pdb_id = df_thermo['PDB_ID'][ind].values
pdb_chain = df_thermo['CHAIN_ID'][ind].values
pdb_id_chain = [x+y for x, y in zip(pdb_id, pdb_chain)]

uniprot_list = []
for i in range(df.shape[0]):
    x = df['PDB ID'][i] + df['Chain ID'][i]
    if x in pdb_id_chain:
        print(x, df['DB ID'][i], df['DB Name'][i])
        p = df['DB ID'][i].split(',')
        uniprot_list.extend(p)

p_uniq = set(x.strip() for x in uniprot_list)
# with open('thermo_pf00072_pdb_uniprot_id.txt', 'wt') as mf:
#     mf.write('uniprot_id\n')
#     for x in p_uniq:
#         mf.write(x + '\n')


df = pd.read_csv('pf00072_unalign_nr90_shuffle_train.csv')
u = df['record_id'].values
uid = np.array([x.split('_')[0] for x in u])

for i in range(uid.shape[0]):
    if uid[i] in p_uniq:
        print(i, uid[i])

u_name = np.array([x.split('/')[0] for x in u])
for i in range(u_name.shape[0]):
    if u_name[i] in ['CHEY_THEMA', 'CHEB_THEMA']:
        print(i, u_name[i])


from sklearn.decomposition import PCA

df_thermo = pd.read_csv('thermo_pf00072_pdb_uniprot_id.txt')
thermo_list = df_thermo['num'].values

df = pd.read_pickle('thermo_vec.pkl')
x = df.values
pca = PCA(n_components=2)
xr = pca.fit(x).transform(x)

pl.figure()
pl.plot(xr[:, 0], xr[:, 1], 'b.', markersize=1)

# pl.figure()
for i in thermo_list:
    pl.plot(xr[i, 0], xr[i, 1], marker='o', color='r', markersize=20)


for i in thermo_list:
    print(np.sum((x[i] - x)**2))

random_list = np.random.randint(0, x.shape[0], 100)
for i in random_list:
    print(np.sum((x[i] - x)**2))




xmean = x.mean(axis=0)
xstd = x.std(axis=0)

xsigma = (x - xmean) / xstd
xsigma2sum = (xsigma**2).sum(axis=1)

a = []

thermo_sigma2sum = xsigma2sum[thermo_list]
