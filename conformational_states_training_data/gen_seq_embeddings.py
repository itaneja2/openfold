import argparse
import os
import subprocess 
import pandas as pd
import itertools
import pickle 
import glob
import sys  
import json 
import re 

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL


asterisk_line = '*********************************'

def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


def save_seq_embeddings_data(data, output_fname):
    output_dir = './seq_embeddings_data' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = '%s/%s.pkl' % (output_dir, output_fname)
    print('saving %s' % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


############################

client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device="cuda")

alignment_dir = './alignment_data'
 
conformational_states_df = pd.read_csv('../conformational_states_dataset/dataset/conformational_states_filtered_adjudicated.csv')
conformational_states_df = conformational_states_df[conformational_states_df['use'] == 'y'].reset_index(drop=True)

seq_embeddings_dict = {} #maps uniprot_id-template_pdb_id to embeddings derived esm3

for index,row in conformational_states_df.iterrows():

    print(asterisk_line)
    print('On uniprot_id %d of %d' % (index, len(conformational_states_df)))
    print(row)
 
    uniprot_id = str(row['uniprot_id'])
    pdb_model_name_ref = str(row['pdb_id_ref'])
    pdb_model_name_state_i = str(row['pdb_id_state_i'])
    seg_len = int(row['seg_len'])

    for template_pdb_id in [pdb_model_name_ref, pdb_model_name_state_i]:
  
        curr_alignment_dir = '%s/%s/%s' % (alignment_dir, uniprot_id, template_pdb_id)
        fasta_file = "%s/%s.fasta" % (curr_alignment_dir, template_pdb_id)
        with open(fasta_file, "r") as fp:
            fasta_data = fp.read()
        _, seq = parse_fasta(fasta_data)
        seq = seq[0]
        print(len(seq))

        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        esm_output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        seq_embeddings = esm_output.per_residue_embedding[1:-1,:] #remove BOS,EOS token
        print(seq_embeddings.shape)

        key = '%s-%s' % (uniprot_id, template_pdb_id)
        seq_embeddings_dict[key] = seq_embeddings

    '''if index == 1: 
        print('SAVING CHECKPOINT EMBEDDINGS') 
        print(seq_embeddings_dict)
        save_seq_embeddings_data(seq_embeddings_dict, 'seq_embeddings_dict')'''


print('SAVING ALL EMBEDDINGS') 
print(len(seq_embeddings_dict.keys()))
save_seq_embeddings_data(seq_embeddings_dict, 'seq_embeddings_dict')


