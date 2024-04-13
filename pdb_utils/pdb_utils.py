import logging
import os 
import io 
import subprocess 
import numpy as np
import pandas as pd 
import string
from io import StringIO
from pathlib import Path
import shutil 
import sys

from Bio.PDB import PDBList
from Bio.PDB import PDBParser, MMCIFParser, MMCIF2Dict
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from Bio.SeqUtils import seq1
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from typing import List

sys.path.insert(0, '../')
from openfold.np.relax import cleanup
from openfold.np.relax.amber_minimize import _get_pdb_string
from openfold.np.protein import from_pdb_string, to_modelcif

try:
    import openmm
    from openmm import unit
    from openmm import app as openmm_app
    from openmm.app.internal.pdbstructure import PdbStructure 
    from openmm.app import PDBFile, PDBxFile
    from openmm.app import Modeller
except ImportError:
    import simtk.openmm
    from simtk.openmm import unit
    from simtk.openmm import app as openmm_app
    from simtk.openmm.app.internal.pdbstructure import PdbStructure
    from simtk.openmm.app import PDBFile, PDBxFile
    from simtk.openmm.app import Modeller

import requests
import urllib.request

from pymol import cmd

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

sifts_script = './parse_sifts.py'
pdb_fixinsert_path = './pdb_fixinsert.py' 

def get_uniprot_seq(uniprot_id):
    base_url = "http://www.uniprot.org/uniprot"
    full_url = "%s/%s.fasta" % (base_url, uniprot_id)
    response = requests.post(full_url)
    if response.status_code == 404:
        raise ValueError("uniprot_id %s is invalid, no sequence found"  % uniprot_id)
    response_output = ''.join(response.text)
    seq = StringIO(response_output)
    parsed_seq = list(SeqIO.parse(seq, 'fasta'))[0]
    return str(parsed_seq.seq)

def get_uniprot_id(pdb_id):
    if len(pdb_id.split('_')) > 1:
        pdb_id = pdb_id.split('_')[0]
    base_url = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot"
    full_url = "%s/%s" % (base_url, pdb_id)
    response = requests.get(full_url)
    if response.status_code == 404:
        raise ValueError("pdb_id %s is invalid, no uniprot found"  % pdb_id)
    data = response.json()
    uniprot_id = list(data[pdb_id]['UniProt'].keys())[0]
    return uniprot_id


def get_pdb_id_seq(pdb_id, uniprot_id):
    if len(pdb_id.split('_')) > 1:
        pdb_id_copy = pdb_id
        pdb_id = pdb_id_copy.split('_')[0]
        chain_id = pdb_id_copy.split('_')[1]
    else:
        chain_id = 'A'

    pdb_metadata_df = fetch_pdb_metadata_df(pdb_id)
    curr_uniprot_id = pdb_metadata_df.loc[0,'uniprot_id']
    if curr_uniprot_id != uniprot_id:
        raise ValueError('fetching pdb_id seq, but uniprot_id does not match expected')

    pdb_metadata_df_relchain = pdb_metadata_df[pdb_metadata_df['chain'] == chain_id].reset_index()
    pdb_metadata_df_relchain = pdb_metadata_df_relchain[pdb_metadata_df_relchain['pdb_resnum'] != 'null']
    if len(pdb_metadata_df_relchain) == 0: 
        raise ValueError("fetching pdb_id seq, but can't find pdb_id %s in SIFTS database" % pdb_id)

    seq = ''.join(list(pdb_metadata_df_relchain['pdb_res']))       
    return seq  


def get_pdb_path_seq(pdb_path: str, chain_list: List[str]):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    seq = '' 
    for model in structure:
        for chain in model:
            if chain_list is not None:
                if chain.id in chain_list:
                    for residue in chain.get_residues():
                        seq += seq1(residue.get_resname())
            else:
                for residue in chain.get_residues():
                    seq += seq1(residue.get_resname())

    return seq

def clean_pdb(pdb_path: str, pdb_str: str):
    pdb_file = io.StringIO(pdb_str)
    alterations_info = {}
    fixed_pdb = cleanup.fix_pdb_wo_adding_missing_residues(pdb_file, alterations_info)
    fixed_pdb_file = io.StringIO(fixed_pdb)
    pdb_structure = PdbStructure(fixed_pdb_file)
    cleanup.clean_structure(pdb_structure, alterations_info)
    as_file = openmm_app.PDBFile(pdb_structure)
    pdb_str_cleaned = _get_pdb_string(as_file.getTopology(), as_file.getPositions())
    with open(pdb_path, "w") as f:
        f.write(pdb_str_cleaned)

def renumber_chain_wrt_reference(pdb_modify_path, pdb_ref_path):
    # Parse the PDB files
    parser = PDBParser()
    logger.info('Renumbering chains of %s to match %s' % (pdb_modify_path, pdb_ref_path))
    structure_to_modify = parser.get_structure('renumber', pdb_modify_path)
    reference_structure = parser.get_structure('reference', pdb_ref_path)

    reference_chains_list = [] 
    for model in reference_structure:
        for chain in model:
            reference_chains_list.append(chain.id)

    num_chains = len(reference_chains_list)
    
    chain_num = 0 
    for model in structure_to_modify:
        for chain in model:
            chain.id = reference_chains_list[chain_num]
            chain_num += 1

    # Save the modified structure
    io = PDBIO()
    io.set_structure(structure_to_modify)
    io.save(pdb_modify_path)

    return num_chains


def fetch_pdb_metadata_df(pdb_id):
    curl_cmd = ['curl --silent ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/%s.xml.gz | gunzip | python %s' % (pdb_id.lower(),sifts_script)] 
    output = subprocess.check_output(curl_cmd, shell=True, universal_newlines=True)
    lines = output.strip().split('\n')

    ##SIFTS file provides info on the pdb sequence construct without coordinate info. this is reflected in the column pdb_res. 
    ##if a residue is missing, the pdb_resnum will show up as null 
    ##if a mutation in a residue was made, the residue will differ from the uniprot_res

    lines_arr = [x.split('\t') for x in lines]
    pdb_metadata_df = pd.DataFrame(lines_arr, columns = ['pdb_id', 'chain', 'pdb_res', 'pdb_resnum', 'uniprot_id', 'uniprot_res', 'uniprot_resnum'])
    pdb_metadata_df = pdb_metadata_df[['pdb_id', 'chain', 'uniprot_id', 'pdb_res', 'pdb_resnum', 'uniprot_res', 'uniprot_resnum']]
    return pdb_metadata_df

def get_model_name(pdb_path: str):
    return pdb_path[pdb_path.rfind('/')+1:pdb_path.rfind('.')]

def get_pymol_cmd_superimpose(pdb_model_name, chain=None):
    if chain is not None:
        out = '%s and chain %s' % (pdb_model_name,chain)
    else:
        out = '%s' % (pdb_model_name)
    return out

def get_pymol_cmd_save(pdb_model_name, chain=None):
    if chain is not None:
        out = '%s and chain %s' % (pdb_model_name,chain)
    else:
        out = '%s' % (pdb_model_name)
    return out 

def save_pdb_chain(pdb_input_path, pdb_output_path, chain):
    if os.path.isfile(pdb_output_path) == False:
        cmd.reinitialize()
        cmd.load(pdb_input_path)
        pdb_model_name = get_model_name(pdb_input_path)
        s = get_pymol_cmd_save(pdb_model_name, chain)
        cmd.save(pdb_output_path, s)
        cmd.delete('all')
    return pdb_output_path

def get_rmsd(pdb1_path, pdb2_path, pdb1_chain=None, pdb2_chain=None):

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    cmd.reinitialize()
    cmd.load(pdb1_path)
    cmd.load(pdb2_path)

    s1 = get_pymol_cmd_superimpose(pdb1_model_name, pdb1_chain)
    s2 = get_pymol_cmd_superimpose(pdb2_model_name, pdb2_chain)

    out = cmd.super(s2,s1) #this superimposes s2 onto s1
    rmsd = out[0]*10 #convert to angstrom

    if rmsd < 0:
        logger.info("RMSD < 0")
        rmsd = 0 

    cmd.delete('all')

    return rmsd 


def align_and_get_rmsd(pdb1_path, pdb2_path, pdb1_chain=None, pdb2_chain=None):

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    cmd.reinitialize()
    cmd.load(pdb1_path)
    cmd.load(pdb2_path)

    s1 = get_pymol_cmd_superimpose(pdb1_model_name, pdb1_chain)
    s2 = get_pymol_cmd_superimpose(pdb2_model_name, pdb2_chain)

    logger.info('super %s,%s' % (s2,s1))

    out = cmd.super(s2,s1) #this superimposes s2 onto s1
    rmsd = out[0]*10 #convert to angstrom
    logger.info('RMSD: %.3f' % rmsd)

    if rmsd < 0:
        logger.info("RMSD < 0")
        rmsd = 0 

    s2 = get_pymol_cmd_save(pdb2_model_name)
    cmd.save(pdb2_path, s2)
    cmd.delete('all')

    return rmsd 

def fetch_mmcif(pdb_id: str, save_dir: str):

    """
    Args:
        pdb_id: e.g 1xyz
    """     
    pdb_list = PDBList()
    cif_fname = pdb_list.retrieve_pdb_file(pdb_id, file_format='mmCif', pdir=save_dir)
    return cif_fname


def fetch_pdb(pdb_id: str, save_dir: str, clean=False) -> str:

    """
    Args:
        pdb_id: e.g 1xyz_A or 1xyz
        save_dir: e.g ./pdb_raw_structures_folder
    """ 

    pdb_save_path = "%s/%s.pdb" % (save_dir, pdb_id)

    if os.path.exists(pdb_save_path):
        return pdb_save_path

    cmd.reinitialize()
    if len(pdb_id.split('_')) > 1:
        pdb_id_wo_chain = pdb_id.split('_')[0]
        chain_id = pdb_id.split('_')[1]
        cmd.fetch(pdb_id_wo_chain, async_=0)
        cmd.save(pdb_save_path, "chain %s and %s" % (chain_id, pdb_id_wo_chain))
        chain_id_list = [chain_id]
    else:
        cmd.fetch(pdb_id, async_=0)
        cmd.save(pdb_save_path, pdb_id)
        chain_id_list = None
    cmd.delete('all')

    if len(pdb_id.split('_')) > 1:
        fetch_path = './%s.cif' % pdb_id_wo_chain
    else:
        fetch_path = './%s.cif' % pdb_id

    if os.path.exists(fetch_path):
        os.remove(fetch_path)    

    if clean:
        with open(pdb_save_path, "r") as f:
            pdb_str = f.read()
        clean_pdb(pdb_save_path, pdb_str)

    return pdb_save_path

def fetch_af_pdb(uniprot_id: str, save_dir: str):

    os.makedirs(save_dir, exist_ok=True)

    url = 'https://alphafold.ebi.ac.uk/api/prediction/%s' % uniprot_id
    params = {'key': 'AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'}
    headers = {'accept': 'application/json'}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        logger.info("Failed to retreive AF metadata")
        logger.info(response.raise_for_status())
        return None  

    af_id = data[0]['entryId']
    cif_url = data[0]['cifUrl']
    af_seq = data[0]['uniprotSequence']

    cif_path = '%s/%s.cif' % (save_dir, af_id)
    urllib.request.urlretrieve(cif_url, cif_path)
    pdb_path = cif_path.replace('.cif','.pdb')
    pdb_path = pdb_path.replace('-','')
    convert_mmcif_to_pdb(cif_path, pdb_path)
    os.remove(cif_path)    
 
    return (pdb_path, af_id, af_seq)

def get_pdb_str(pdb_id: str, pdb_path: str) -> str:

    """
    Args:
        pdb_id: e.g 1xf2
        pdb_path: e.g ./pdb_raw_structures_folder/pdb1xf2.cif
    """ 

    logger.info('Getting pdb_str for %s,%s' % (pdb_id,pdb_path))

    if '.cif' in pdb_path:
        parser = MMCIFParser()
        structure = parser.get_structure(pdb_id, pdb_path)
        new_pdb = io.StringIO()
        pdb_io=PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(new_pdb)
        return new_pdb.getvalue()
    else:
        raise ValueError('.cif must be in %s' % pdb_path)


def get_bfactor(cif_path: str):

    structure = MMCIF2Dict.MMCIF2Dict(cif_path)
    b_factor = structure['_atom_site.B_iso_or_equiv']
    res = structure['_atom_site.label_comp_id']
    atom = structure['_atom_site.label_atom_id']

    b_factor_ca = []
    res_ca = []

    for i in range(0,len(atom)):
        if atom[i] == 'CA':
            b_factor_ca.append(float(b_factor[i]))
            res_ca.append(res[i])

    b_factor_ca = np.array(b_factor_ca)
    mean_bfactor_ca = np.mean(b_factor_ca)
    return b_factor_ca, mean_bfactor_ca

def convert_mmcif_to_pdb(cif_path, pdb_path):
    cmd.reinitialize()
    cmd.load(cif_path, 'protein')
    cmd.save(pdb_path, 'protein')
    cmd.delete('all')

def convert_pdb_to_mmcif(pdb_path: str, output_dir: str) -> str:
    """
    Args:
        pdb_path: path to .pdb file 
    """ 

    os.makedirs(output_dir, exist_ok=True)
    model_name = pdb_path[pdb_path.rindex('/')+1:pdb_path.rindex('.pdb')]
    cif_path = os.path.join(output_dir, f'{model_name}.cif')

    with open(pdb_path, "r") as f:
        pdb_str = f.read()
    prot = from_pdb_string(pdb_str)
    cif_string = to_modelcif(prot)
    with open(cif_path, 'w') as f:
        f.write(cif_string)

    return cif_path 

def num_to_chain(number):
    if 0 <= number < 26:
        return string.ascii_uppercase[number]
    else:
        raise ValueError("Number should be between 0 and 25 for A to Z chains")

def assemble_multiple_pdbs_to_single_file(pdb_path_list, output_path_wo_extension):

    pdb_output_path = '%s.pdb' % output_path_wo_extension
    cif_output_path = '%s.cif' % output_path_wo_extension

    cmd.reinitialize()
    pdb_str_all = []
    for i,pdb_path in enumerate(pdb_path_list):
        pdb_str = 'protein_%d' % i
        cmd.load(pdb_path, pdb_str)
        cmd.alter('%s and chain A' % pdb_str, "chain='%s'" % num_to_chain(i))
        pdb_str_all.append(pdb_str)
    cmd.sort()
    pdb_or_str = ' or '.join(pdb_str_all)
    cmd.create("multimer", pdb_or_str)
    cmd.save(pdb_output_path, 'multimer')
    cmd.delete('all')

    with open(pdb_output_path, "r") as f:
        pdb_str = f.read()
    prot = from_pdb_string(pdb_str)
    cif_string = to_modelcif(prot)
    with open(cif_output_path, 'w') as f:
        f.write(cif_string)

    os.remove(pdb_output_path)   
    return cif_output_path 

def superimpose_wrapper_monomer(pdb1_full_id: str, pdb2_full_id: str, pdb1_source: str, pdb2_source: str, pdb1_path: str, pdb2_path: str, save_dir: str):

    """
    Args:
        pdb1_full_id: e.g 1xf2_A
        pdb2_full_id: e.g 1ya3_B
        pdb1_source: either 'pdb' or 'pred_*'
        pdb2_source: either 'pdb' or 'pred_*' 
        pdb1_path: path to first protein structure (if None, then retrieve pdb)
        pdb2_path: path to second protein structure (if None, then retrieve pdb)
    """ 
 
    logger.info('pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))

    if 'pred' in pdb1_source:
        logger.info('ERROR: pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))
        raise ValueError('Because we are superimposing pdb2 on pdb1, pdb1 should be from PDB while pdb2 should be a predicted structure or from PDB')

    if (pdb1_source == 'pdb') and (pdb1_path is None):
        pdb1_id = pdb1_full_id.split('_')[0]
        pdb1_chain = pdb1_full_id.split('_')[1] 
    else:
        pdb1_id = None
        pdb1_chain = None

    if (pdb2_source == 'pdb') and (pdb2_path is None):
        pdb2_id = pdb2_full_id.split('_')[0]
        pdb2_chain = pdb2_full_id.split('_')[1] 
    else:
        pdb2_id = None
        pdb2_chain = None 

    if pdb1_source == 'pred' and pdb2_source == 'pdb':
        raise ValueError('Incompatible combination of pdb1_source and pdb2_source')

    pdb_ref_struc_folder = '%s/pdb_ref_structure' % save_dir
    Path(pdb_ref_struc_folder).mkdir(parents=True, exist_ok=True)
   
    pdb_superimposed_folder = '%s/pdb_superimposed_structures' % save_dir 
    Path(pdb_superimposed_folder).mkdir(parents=True, exist_ok=True)


    if (pdb1_path is None) and (pdb1_source == 'pdb'):
        pdb1_path = fetch_pdb(pdb1_full_id, pdb_ref_struc_folder)
        pdb1_chain_path = pdb1_path.replace('%s.pdb' % pdb1_id, '%s.pdb' % pdb1_full_id)
        pdb1_path = save_pdb_chain(pdb1_path, pdb1_chain_path, pdb1_chain) 
    else:
        if (pdb1_path is not None) and (pdb1_source == 'pdb'):
            shutil.copyfile(pdb1_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb1_full_id))

    if (pdb2_path is None) and (pdb2_source == 'pdb'):
        pdb2_path = fetch_pdb(pdb2_full_id, pdb_ref_struc_folder) 
        pdb2_chain_path = pdb2_path.replace('%s.pdb' % pdb2_id, '%s.pdb' % pdb2_full_id)
        pdb2_path = save_pdb_chain(pdb2_path, pdb2_chain_path, pdb2_chain) 
    else:
        if (pdb2_path is not None) and (pdb2_source == 'pdb'):
            shutil.copyfile(pdb2_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb2_full_id))

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    with open(pdb1_path, "r") as f:
        pdb1_str = f.read()
    with open(pdb2_path, "r") as f:
        pdb2_str = f.read()

    pdb1_input_path = pdb1_path
    pdb2_input_path = pdb2_path 

    pdb1_path_clean = pdb1_path.replace('.pdb', '_clean.pdb') 
    pdb2_path_clean = pdb2_path.replace('.pdb', '_clean.pdb')

    if (pdb1_path is not None) and (pdb1_source == 'pdb'): 
        logger.info('Cleaning pdb file %s' % pdb1_path)
        clean_pdb(pdb1_path_clean, pdb1_str)
        pdb1_input_path = pdb1_path_clean
    if (pdb2_path is not None) and (pdb2_source == 'pdb'):
        logger.info('Cleaning pdb file %s' % pdb2_path)
        clean_pdb(pdb2_path_clean, pdb2_str)
        pdb2_input_path = pdb2_path_clean
    
    pdb2_output_path = '%s/%s.pdb' % (pdb_superimposed_folder, pdb2_model_name)
    shutil.copyfile(pdb2_input_path, pdb2_output_path)

    #pdb1_chain and pdb2_chain are set to A because
    #clean_pdb replaces chain_id with A 
    rmsd = align_and_get_rmsd(pdb1_input_path, pdb2_output_path, 'A', 'A')
    logger.info('SAVING ALIGNED PDB AT %s' % pdb2_output_path)
 
    return rmsd, pdb1_path, pdb2_path


def superimpose_wrapper_multimer(pdb1_id: str, pdb2_id: str, pdb1_source: str, pdb2_source: str, pdb1_path: str, pdb2_path: str, save_dir: str):

    """
    Args:
        pdb1_id: e.g 1ixf2
        pdb2_id: e.g 1ya3
        pdb1_source: either 'pdb' or 'pred_*'
        pdb2_source: either 'pdb' or 'pred_*' 
        pdb1_path: path to first protein structure (if None, then retrieve pdb)
        pdb2_path: path to second protein structure (if None, then retrieve pdb)
    """ 
 
    logger.info('pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))

    if 'pred' in pdb1_source:
        logger.info('ERROR: pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))
        raise ValueError('because we are superimposing pdb2 on pdb1, pdb1 should be from PDB while pdb2 should be a predicted structure or from PDB')

    if pdb1_source == 'pred' and pdb2_source == 'pdb':
        raise ValueError('Incompatible combination of pdb1_source and pdb2_source')

    pdb_ref_struc_folder = '%s/pdb_ref_structure' % save_dir
    Path(pdb_ref_struc_folder).mkdir(parents=True, exist_ok=True)
   
    pdb_superimposed_folder = '%s/pdb_superimposed_structures' % save_dir             
    Path(pdb_superimposed_folder).mkdir(parents=True, exist_ok=True)

    if (pdb1_path is None) and (pdb1_source == 'pdb'):
        pdb1_path = fetch_pdb(pdb1_id, pdb_ref_struc_folder)
    else:
        if (pdb1_path is not None) and (pdb1_source == 'pdb'):
            shutil.copyfile(pdb1_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb1_id))

    if (pdb2_path is None) and (pdb2_source == 'pdb'):
        pdb2_path = fetch_pdb(pdb2_id, pdb_ref_struc_folder)
    else:
        if (pdb2_path is not None) and (pdb2_source == 'pdb'):
            shutil.copyfile(pdb2_path, '%s/%s.pdb' % (pdb_ref_struc_folder,pdb2_id))

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    with open(pdb1_path, "r") as f:
        pdb1_str = f.read()
    with open(pdb2_path, "r") as f:
        pdb2_str = f.read()

    pdb1_input_path = pdb1_path
    pdb2_input_path = pdb2_path 

    pdb1_path_clean = pdb1_path.replace('.pdb', '_clean.pdb') 
    pdb2_path_clean = pdb2_path.replace('.pdb', '_clean.pdb')

    if (pdb1_path is not None) and (pdb1_source == 'pdb'): 
        logger.info('Cleaning pdb file %s' % pdb1_path)
        clean_pdb(pdb1_path_clean, pdb1_str)
        pdb1_input_path = pdb1_path_clean
    if (pdb2_path is not None) and (pdb2_source == 'pdb'):
        logger.info('Cleaning pdb file %s' % pdb2_path)
        clean_pdb(pdb2_path_clean, pdb2_str)
        pdb2_input_path = pdb2_path_clean
 
    pdb2_output_path = '%s/%s.pdb' % (pdb_superimposed_folder, pdb2_model_name)
    shutil.copyfile(pdb2_input_path, pdb2_output_path)

    pdb1_chain = None
    pdb2_chain = None 
    rmsd = align_and_get_rmsd(pdb1_input_path, pdb2_output_path, pdb1_chain, pdb2_chain)
    logger.info('SAVING ALIGNED PDB AT %s' % pdb2_output_path)

    return rmsd, pdb1_path, pdb2_path


def get_flanking_residues_idx(seq1: str, seq2: str):
    """
    seq1: MSDE
    seq2: -SD-
    return: seq1: [], seq2: [0,3]
    """ 

    def get_nterm_flanking(seq_aligned):
        nterm_idx = [] 
        i = 0 
        while i < len(seq_aligned):
            if seq_aligned[i] == '-':
                nterm_idx.append(i)
                i += 1
            else:
                break  
        return nterm_idx
    def get_cterm_flanking(seq1_aligned, seq2_aligned):
        cterm_idx = [] 
        i = len(seq1_aligned) 
        while i > 0:
            if seq1_aligned[i-1] == '-':
                num_gaps = seq2_aligned[0:(i-1)].count('-') #need to use seq2 for number of gaps to get correct idx 
                cterm_idx.append(i-1-num_gaps)
                i -= 1
            else: 
                break
        return cterm_idx

    alignments = pairwise2.align.globalxx(seq1, seq2)

    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    flanking_residues_dict = {} 
    #if seq1 has a gap, then the corresponding residue in seq2 can be 
    #considered flanking (and vice-versa) 
    flanking_residues_dict['seq2_nterm'] = get_nterm_flanking(seq1_aligned)
    flanking_residues_dict['seq2_cterm'] = get_cterm_flanking(seq1_aligned, seq2_aligned)
    flanking_residues_dict['seq1_nterm'] = get_nterm_flanking(seq2_aligned)
    flanking_residues_dict['seq1_cterm'] = get_cterm_flanking(seq2_aligned, seq1_aligned)

    return flanking_residues_dict


def get_starting_idx_aligned(aligned_seq):
    #becase of issues with aligning n/c terminal regions of sequences, we want to start where there are 
    #consecutive hits
    for i in range(0,len(aligned_seq)-1):
        if aligned_seq[i] != '-':
            if aligned_seq[i+1] != '-':
                return i 
    
    return -1


def get_ending_idx_aligned(aligned_seq):
    #becase of issues with aligning n/c terminal regions of sequences, we want to start where there are 
    #consecutive hits
    for i in range(len(aligned_seq)-1,1,-1):
        if aligned_seq[i] != '-':
            if aligned_seq[i-1] != '-':
                return i 
    
    return -1

def get_common_aligned_residues_idx_excluding_flanking_regions(seq1: str, seq2: str):

    seq1_start_idx = get_starting_idx_aligned(seq1)
    seq2_start_idx = get_starting_idx_aligned(seq2)

    seq1_end_idx = get_ending_idx_aligned(seq1)
    seq2_end_idx = get_ending_idx_aligned(seq2)

    start_idx = max(seq1_start_idx, seq2_start_idx)
    end_idx = min(seq1_end_idx, seq2_end_idx)

    return start_idx, end_idx
    

def get_residues_idx_in_seq1_and_seq2(seq1: str, seq2: str):
    """
    seq1: MS-E-
    seq2: MSDER
    return: [0,1,2], [0,1,3]
    """ 

    alignments = pairwise2.align.globalxx(seq1, seq2)
    print('Aligned seq1 with seq2:')
    print(format_alignment(*alignments[0]))

    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    start_aligned_to_original_idx, end_aligned_to_original_idx = get_common_aligned_residues_idx_excluding_flanking_regions(seq1_aligned, seq2_aligned)

    print('start_idx: %s' % start_aligned_to_original_idx)
    print('end_idx: %s' % end_aligned_to_original_idx)

    seq1_residues_idx = [] 
    seq2_residues_idx = [] 

    for i in range(start_aligned_to_original_idx, end_aligned_to_original_idx+1):
        if seq1_aligned[i] != '-' and seq2_aligned[i] != '-':
            num_gaps_seq1 = seq1_aligned[0:i].count('-')
            num_gaps_seq2 = seq2_aligned[0:i].count('-')
            seq1_residues_idx.append(i-num_gaps_seq1)
            seq2_residues_idx.append(i-num_gaps_seq2)

    return seq1_residues_idx, seq2_residues_idx



def get_residues_idx_in_seq2_not_seq1(seq1: str, seq2: str):
    """
    seq1: MS-E-
    seq2: MSDER
    return: [2,4]
    """ 

    alignments = pairwise2.align.globalxx(seq1, seq2)
    #print('Aligned seq1 with seq2:')
    #print(format_alignment(*alignments[0]))

    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    residues_idx = [] 
    for i in range(0,len(seq1_aligned)):
        if seq1_aligned[i] == '-' and seq2_aligned[i] != '-':
            num_gaps = seq2_aligned[0:i].count('-')
            residues_idx.append(i-num_gaps)

    return residues_idx
   

def align_seqs(seq1, seq2):

    alignments = pairwise2.align.globalxx(seq1, seq2)
    print('Aligned PDB1 seq with PDB2 seq:')
    print(format_alignment(*alignments[0]))
    seq1_aligned = alignments[0].seqA
    seq2_aligned = alignments[0].seqB

    seq1_aligned_to_original_idx_mapping = {} #maps each index in pdb_seq_aligned to the corresponding index in pdb_seq (i.e accounting for gaps) 
    for i in range(0,len(seq1_aligned)):
        if seq1_aligned[i] != '-': 
            seq1_aligned_to_original_idx_mapping[i] = i-seq1_aligned[0:i].count('-')

    seq2_aligned_to_original_idx_mapping = {} #maps each index in pdb_seq_aligned to the corresponding index in pdb_seq (i.e accounting for gaps) 
    for i in range(0,len(seq2_aligned)):
        if seq2_aligned[i] != '-': 
            seq2_aligned_to_original_idx_mapping[i] = i-seq2_aligned[0:i].count('-')

    print(seq1_aligned_to_original_idx_mapping)
    print(seq2_aligned_to_original_idx_mapping)

    return seq1_aligned, seq2_aligned, seq1_aligned_to_original_idx_mapping, seq2_aligned_to_original_idx_mapping


def get_af_disordered_domains(pdb_path: str):
    """pdb_path assumes AF structure. for region to be considered disordered domain, must be >=3 residues in length
    """
    #plddt_threshold sourced from https://onlinelibrary.wiley.com/doi/10.1002/pro.4466

    cif_path = convert_pdb_to_mmcif(pdb_path, './cif_temp')
    plddt_scores, mean_plddt = get_bfactor(cif_path)
    os.remove(cif_path)
    pdb_seq = get_pdb_path_seq(pdb_path, None)

    af_disordered = 1-plddt_scores/100
    af_disordered = af_disordered >= .31
     
    af_disordered_domains_idx = []
    start_idx = 0
    while start_idx < len(af_disordered):
        if af_disordered[start_idx]:
            end_idx = start_idx+1
            if end_idx < len(af_disordered):
                while af_disordered[end_idx]:
                    end_idx += 1
                    if end_idx >= len(af_disordered):
                        break 
            #by postcondition: af_disordered[end_idx] = False   
            seg_len = end_idx-start_idx
            if seg_len >= 3:
                af_disordered_domains_idx.append([start_idx, end_idx-1])
            start_idx = end_idx+1
        else:
            start_idx += 1 
               
    af_disordered_domains_seq = [] 
    for i in range(0,len(af_disordered_domains_idx)):
        af_disordered_domains_seq.append([pdb_seq[af_disordered_domains_idx[i][0]:af_disordered_domains_idx[i][1]+1]])

    return af_disordered_domains_idx, af_disordered_domains_seq
    

def get_pdb_disordered_domains(af_seq: str, pdb_seq: str, af_disordered_domains_idx: List[List[int]]):

    alignments = pairwise2.align.globalxx(af_seq, pdb_seq)
    print('Aligned AF seq with PDB seq:')
    print(format_alignment(*alignments[0]))
    af_seq_aligned = alignments[0].seqA
    pdb_seq_aligned = alignments[0].seqB

    af_seq_original_to_aligned_idx_mapping = {} #maps each index in af_seq to the corresponding index in af_seq_aligned (i.e accounting for gaps) 
    for i in range(0,len(af_seq_aligned)):
        if af_seq_aligned[i] != '-':
            af_seq_original_to_aligned_idx_mapping[i-af_seq_aligned[0:i].count('-')] = i 

    pdb_disordered_domains_idx = [] 
    pdb_disordered_domains_seq = [] 

    #if ith residue of af_seq is disordered
    #then, to get the corresponding index in pdb_seq
    #we need to subtract num_gaps(pdb_seq_aligned[0:i]) from i 

    for i in range(0,len(af_disordered_domains_idx)):
        d = af_disordered_domains_idx[i]
        af_start_idx = d[0]
        af_end_idx = d[1]
        af_start_idx_aligned = af_seq_original_to_aligned_idx_mapping[af_start_idx]
        af_end_idx_aligned = af_seq_original_to_aligned_idx_mapping[af_end_idx]
        for j in range(af_start_idx_aligned, af_end_idx_aligned+1): 
            if pdb_seq_aligned[j] != '-':
                num_gaps = pdb_seq_aligned[0:j].count('-')
                pdb_seq_original_idx = j-num_gaps
                pdb_disordered_domains_idx.append(pdb_seq_original_idx)
                pdb_disordered_domains_seq.append(pdb_seq[pdb_seq_original_idx])

    return pdb_disordered_domains_idx, pdb_disordered_domains_seq 
   

def get_ca_coords_dict(pdb_path):

    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model['A']
    
    residue_idx_ca_coords_dict = {} #maps residue index to ca coordinates 

    for residue in chain:
        if 'CA' in residue:
            ca_atom = residue['CA']
            ca_coords = ca_atom.get_coord()
            residue_name = residue.get_resname()
            residue_idx = residue.get_id()[1]-1
            residue_idx_ca_coords_dict[residue_idx] = (ca_coords[0], ca_coords[1], ca_coords[2], residue_name)

    return residue_idx_ca_coords_dict


def get_residue_idx_below_rmsf_threshold(pdb1_seq, pdb2_seq, pdb1_path, pdb2_path, pdb1_include_idx, pdb2_include_idx, pdb1_exclude_idx, pdb2_exclude_idx, rmsf_threshold = 2.0):

    pdb1_seq_aligned, pdb2_seq_aligned, pdb1_seq_aligned_to_original_idx_mapping, pdb2_seq_aligned_to_original_idx_mapping = align_seqs(pdb1_seq, pdb2_seq)

    pdb1_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb1_path)
    pdb2_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb2_path)
 
    pdb1_ca_pos_aligned = []
    pdb2_ca_pos_aligned = [] 

    pdb1_residue_idx_below_threshold = []
    pdb2_residue_idx_below_threshold = [] 

    for i in range(0,len(pdb1_seq_aligned)):
        if (pdb1_seq_aligned[i] == pdb2_seq_aligned[i]) and (pdb1_seq_aligned[i] != '-'):
            pdb1_seq_original_idx = pdb1_seq_aligned_to_original_idx_mapping[i]
            pdb2_seq_original_idx = pdb2_seq_aligned_to_original_idx_mapping[i]
            valid_idx = (pdb1_seq_original_idx in pdb1_include_idx) and (pdb2_seq_original_idx in pdb2_include_idx) and (pdb1_seq_original_idx not in pdb1_exclude_idx) and (pdb2_seq_original_idx not in pdb2_exclude_idx)
            if valid_idx:
                print("aligned idx for pdb1: %d, seq idx: %d:" % (i, pdb1_seq_original_idx))
                print("aligned idx for pdb2: %d, seq idx: %d:" % (i, pdb2_seq_original_idx))
                pdb1_ca_pos = list(pdb1_residue_idx_ca_coords_dict[pdb1_seq_original_idx][0:3])
                pdb2_ca_pos = list(pdb2_residue_idx_ca_coords_dict[pdb2_seq_original_idx][0:3])
                curr_ca_rmsf = np.linalg.norm(np.array(pdb1_ca_pos) - np.array(pdb2_ca_pos))
                print(curr_ca_rmsf)
                if curr_ca_rmsf < rmsf_threshold:
                    pdb1_residue_idx_below_threshold.append(pdb1_seq_original_idx)
                    pdb2_residue_idx_below_threshold.append(pdb2_seq_original_idx)
                pdb1_ca_pos_aligned.append(pdb1_ca_pos)
                pdb2_ca_pos_aligned.append(pdb2_ca_pos)
                
    pdb1_ca_pos_aligned = np.array(pdb1_ca_pos_aligned)
    pdb2_ca_pos_aligned = np.array(pdb2_ca_pos_aligned)

    ca_rmsf = np.linalg.norm(pdb1_ca_pos_aligned-pdb2_ca_pos_aligned, axis=1)

    print(ca_rmsf)
    print(pdb1_residue_idx_below_threshold)
    print(pdb2_residue_idx_below_threshold)

    return pdb1_residue_idx_below_threshold, pdb2_residue_idx_below_threshold

def get_residues_ignore_idx_between_pdb_conformations(state1_pdb_path, state2_pdb_path, state1_af_path, state2_af_path):

    """Ignore residues based on:
        1. disorder
        2. common residues
        3. rmsf threshold
    """  

    state1_pdb_seq = get_pdb_path_seq(state1_pdb_path, None)
    state2_pdb_seq = get_pdb_path_seq(state2_pdb_path, None)
    state1_af_seq = get_pdb_path_seq(state1_af_path, None)
    state2_af_seq = get_pdb_path_seq(state2_af_path, None)

    try: 
        state1_af_disordered_domains_idx, _ = get_af_disordered_domains(state1_af_seq)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION') 
        print(e)
        state1_af_disordered_domains_idx = [] 

    try: 
        state2_af_disordered_domains_idx, _ = get_af_disordered_domains(state2_af_seq)      
    except ValueError as e:
        print('TROUBLE PARSING AF PREDICTION') 
        print(e)
        state2_af_disordered_domains_idx = [] 

    state1_pdb_disordered_domains_idx, _ = get_pdb_disordered_domains(state1_af_seq, state1_pdb_seq, state1_af_disordered_domains_idx) 
    state2_pdb_disordered_domains_idx, _ = get_pdb_disordered_domains(state2_af_seq, state2_pdb_seq, state2_af_disordered_domains_idx) 
    
    state1_pdb_exclusive_residues_idx = get_residues_idx_in_seq2_not_seq1(state1_af_seq, state1_pdb_seq)
    if len(state1_pdb_exclusive_residues_idx) > 0:
        print('Residues in PDB, but not AF:')
        print(pdb_exclusive_residues_idx)
        print('This should be empty')
        print('exiting...')
        sys.exit()
           
    state1_pdb_common_residues_idx, state2_pdb_common_residues_idx = get_residues_idx_in_seq1_and_seq2(state1_pdb_seq, state2_pdb_seq)
    state1_pdb_rmsf_below_threshold_idx, state2_pdb_rmsf_below_threshold_idx = get_residue_idx_below_rmsf_threshold(
                                                                                                state1_pdb_seq,
                                                                                                state2_pdb_seq,
                                                                                                state1_pdb_path,
                                                                                                state2_pdb_path,
                                                                                                state1_pdb_common_residues_idx,
                                                                                                state2_pdb_common_residues_idx,
                                                                                                state1_pdb_disordered_domains_idx,
                                                                                                state2_pdb_disordered_domains_idx)

    state1_pdb_common_residues_idx_complement = sorted(list(set(range(len(state1_pdb_seq))) - set(state1_pdb_common_residues_idx)))
    state2_pdb_common_residues_idx_complement = sorted(list(set(range(len(state2_pdb_seq))) - set(state2_pdb_common_residues_idx)))

    print('STATE 1 PDB disordered:')
    print(state1_pdb_disordered_domains_idx)
    print('STATE 1 PDB intersection complement:')
    print(state1_pdb_common_residues_idx_complement)
    print('STATE 1 PDB below RMSF:')
    print(state1_pdb_rmsf_below_threshold_idx)

    print('STATE 2 PDB disordered:')
    print(state2_pdb_disordered_domains_idx)
    print('STATE 2 PDB intersection complement:')
    print(state2_pdb_common_residues_idx_complement)
    print('STATE 2 PDB below RMSF:')
    print(state2_pdb_rmsf_below_threshold_idx)
    
    state1_pdb_residues_ignore_idx = state1_pdb_disordered_domains_idx+state1_pdb_common_residues_idx_complement+state1_pdb_rmsf_below_threshold_idx
    state2_pdb_residues_ignore_idx = state2_pdb_disordered_domains_idx+state2_pdb_common_residues_idx_complement+state2_pdb_rmsf_below_threshold_idx

    state1_pdb_residues_ignore_idx = list(set(state1_pdb_residues_ignore_idx))
    state2_pdb_residues_ignore_idx = list(set(state2_pdb_residues_ignore_idx))

    return state1_pdb_residues_ignore_idx, state2_pdb_residues_ignore_idx
 

def cartesian_to_spherical(ca_pos_diff):
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""

    #utilizing this convention: https://dynref.engr.illinois.edu/rvs.html
    radius = np.linalg.norm(ca_pos_diff, axis=1)
    xy_norm = np.linalg.norm(ca_pos_diff[:,0:2], axis=1)
    phi = np.arctan2(xy_norm, ca_pos_diff[:,2])
    theta = np.arctan2(ca_pos_diff[:,1], ca_pos_diff[:,0])

    return phi, theta, radius


def get_spherical_coordinate_vector_diff(af_pred_path, pdb_path, pdb_residues_ignore_idx):

    af_pred_seq = get_pdb_path_seq(af_pred_path, None)
    pdb_seq = get_pdb_path_seq(pdb_path, None)
    af_pred_seq_aligned, pdb_seq_aligned, af_pred_seq_aligned_to_original_idx_mapping, pdb_seq_aligned_to_original_idx_mapping = align_seqs(af_pred_seq, pdb_seq)

    af_pred_residue_idx_ca_coords_dict = get_ca_coords_dict(af_pred_path)
    pdb_residue_idx_ca_coords_dict = get_ca_coords_dict(pdb_path)
    
    pdb_ca_pos_aligned = []
    af_pred_ca_pos_aligned = [] 

    for i in range(0,len(pdb_seq_aligned)):
        if (pdb_seq_aligned[i] == af_pred_seq_aligned[i]) and (pdb_seq_aligned[i] != '-'):
            pdb_seq_original_idx = pdb_seq_aligned_to_original_idx_mapping[i]
            af_pred_seq_original_idx = af_pred_seq_aligned_to_original_idx_mapping[i]
            if pdb_seq_original_idx not in pdb_residues_ignore_idx:
                pdb_ca_pos = list(pdb_residue_idx_ca_coords_dict[pdb_seq_original_idx][0:3])
                af_pred_ca_pos = list(af_residue_idx_ca_coords_dict[af_pred_seq_original_idx][0:3])
                pdb_ca_pos_aligned.append(pdb_ca_pos)
                af_pred_ca_pos_aligned.append(af_pred_ca_pos)

    pdb_ca_pos_aligned = np.array(pdb_ca_pos_aligned)
    af_pred_ca_pos_aligned = np.array(af_pred_ca_pos_aligned)

    ca_pos_diff = pdb_ca_pos_aligned - af_pred_ca_pos_aligned
    phi, theta, radius = cartesian_to_spherical(ca_pos_diff)

    return np.array([phi, theta, radius])


def select_rel_chain_and_delete_residues(cif_input_path: str, cif_output_path: str, chain_id: str, residues_delete_idx: List[int]):
    
    pdb = PDBxFile(cif_input_path)
    modeller = Modeller(pdb.topology, pdb.positions)

    chains_to_delete = [] 
    for chain in modeller.getTopology().chains():
        if chain.id != chain_id:
            chains_to_delete.append(chain)
        else:
            print('keeping:')
            print(chain)
    print('deleting chains:')
    print(chains_to_delete)
    modeller.delete(chains_to_delete)    

    residues_to_delete = [] 
    for residue in modeller.topology.residues(): 
        if residue.index in residues_delete_idx:
            residues_to_delete.append(residue)
    print('deleting:')
    print(residues_to_delete)
    modeller.delete(residues_to_delete) 

    with open(cif_output_path, 'w') as f:
        PDBxFile.writeFile(modeller.topology, modeller.positions, f)

