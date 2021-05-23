from time import time
import pandas as pd


fasta_file = open('uniref50.fasta', 'rt')
new_file = open('uniref50.csv', 'wt')

time0 = time()

entry_count = 0
new_seq = ''
for line in fasta_file:
    if line[0] == '>':
        if entry_count > 0:
            entry_use = new_file.write(new_seq + ',' + str(len(new_seq)) + '\n')
        entry_count += 1
        if (entry_count % 10000) == 0:
            print(entry_count, time()-time0)
        new_seq = ''
    else:
        new_seq += line[:-1]
else:
    entry_use = new_file.write(new_seq + ',' + str(len(new_seq)) + '\n')
    entry_count += 1
print(entry_count, time() - time0)

fasta_file.close()
new_file.close()


df = pd.read_csv('uniref50.csv', usecols=['seq_len'])

