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

from custom_openfold_utils.pdb_utils import num_to_chain, get_pdb_id_seq, get_uniprot_seq

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


def save_seq_embeddings_data(data, output_fname, output_dir):
    if output_dir is None:
        output_dir = './seq_embeddings_data' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = '%s/%s.pkl' % (output_dir, output_fname)
    print('saving %s' % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


############################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fasta_path", type=str, default=None
)
parser.add_argument(
    "--pdb_id", type=str, default=None,
    help="should be of the format XXXX_Y, where XXXX is the pdb_id and Y is the chain_id",
)
parser.add_argument(
    "--uniprot_id", type=str, default=None,
    help="",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="",
)


add_data_args(parser)
args = parser.parse_args()

if args.pdb_id and args.uniprot_id:
    raise ValueError("Only one of pdb_id/uniprot_id should be set")

if args.fasta_path:
    with open(args.fasta_path) as fasta_file:
        seqs, tags = parse_fasta(fasta_file.read())
    seq = seqs[0]
    seq_id = tags[0]
elif args.pdb_id:
    seq = get_pdb_id_seq(args.pdb_id)
    seq_id = args.pdb_id 
elif args.uniprot_id:
    seq = get_uniprot_seq(args.uniprot_id)
    seq_id = args.uniprot_id

client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device="cuda")

protein = ESMProtein(sequence=seq)
protein_tensor = client.encode(protein)
esm_output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
seq_embedding = esm_output.per_residue_embedding[1:-1,:] #remove BOS,EOS token
seq_embedding_dict = {}
seq_embedding_dict[seq_id] = (seq,seq_embedding)

print('SAVING EMBEDDING FOR %s' % seq_id) 
save_seq_embeddings_data(seq_embedding_dict, seq_id, args.output_dir)
