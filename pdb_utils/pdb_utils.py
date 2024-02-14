import os 
import io 
import subprocess 
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

from typing import List

from openfold.np.relax import cleanup
from openfold.np.relax.amber_minimize import _get_pdb_string
from openfold.np.protein import from_pdb_string, to_modelcif

try:
    import openmm
    from openmm import unit
    from openmm import app as openmm_app
    from openmm.app.internal.pdbstructure import PdbStructure 
except ImportError:
    import simtk.openmm
    from simtk.openmm import unit
    from simtk.openmm import app as openmm_app
    from simtk.openmm.app.internal.pdbstructure import PdbStructure

import requests
import urllib.request

from pymol import cmd

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
    return parsed_seq.seq

def get_uniprot_id(pdb_id):
    base_url = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot"
    full_url = "%s/%s" % (base_url, pdb_id)
    response = requests.get(full_url)
    if response.status_code == 404:
        raise ValueError("pdb_id %s is invalid, no uniprot found"  % pdb_id)
    data = response.json()
    uniprot_id = list(data[pdb_id]['UniProt'].keys())[0]
    return uniprot_id


def get_pdb_seq(pdb_path: str, chain_list: List[str]):
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
    print('RENUMBERING CHAINS OF %s TO MATCH %s' % (pdb_modify_path, pdb_ref_path))
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


def align_and_get_rmsd(pdb1_path, pdb2_path, pdb1_chain=None, pdb2_chain=None):

    pdb1_model_name = get_model_name(pdb1_path)
    pdb2_model_name = get_model_name(pdb2_path)
    
    cmd.reinitialize()
    cmd.load(pdb1_path)
    cmd.load(pdb2_path)

    s1 = get_pymol_cmd_superimpose(pdb1_model_name, pdb1_chain)
    s2 = get_pymol_cmd_superimpose(pdb2_model_name, pdb2_chain)

    print('super %s,%s' % (s2,s1))

    out = cmd.super(s2,s1) #this superimposes s2 onto s1
    rmsd = out[0]*10 #convert to angstrom
    print('RMSD: %.3f' % rmsd)

    if rmsd < 0:
        print("RMSD < 0")
        rmsd = 0 

    s2 = get_pymol_cmd_save(pdb2_model_name)
    cmd.save(pdb2_path, s2)
    cmd.delete('all')

    return rmsd 


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

    fetch_path = './%s.cif' % pdb_id
    if os.path.exists(fetch_path):
        os.remove(fetch_path)    

    if clean:
        with open(pdb_save_path, "r") as f:
            pdb_str = f.read()
        clean_pdb(pdb_save_path, pdb_str)

    return pdb_save_path

def fetch_af_pdb(uniprot_id, save_dir):

    url = 'https://alphafold.ebi.ac.uk/api/prediction/%s' % uniprot_id
    params = {'key': 'AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'}
    headers = {'accept': 'application/json'}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Failed to retreive AF metadata")
        print(response.raise_for_status())
        return None  

    af_id = data[0]['entryId']
    cif_url = data[0]['cifUrl']
    af_seq = data[0]['uniprotSequence']

    cif_path = '%s/%s.cif' % (save_dir, af_id)
    urllib.request.urlretrieve(cif_url, cif_path)
    pdb_path = cif_path.replace('.cif','.pdb')
    pdb_path = cif_path.replace('-','')
    convert_mmcif_to_pdb(cif_path, pdb_path)
    os.remove(cif_path)    
 
    return (af_id, pdb_path, af_seq)

def get_pdb_str(pdb_id: str, pdb_path: str) -> str:

    """
    Args:
        pdb_id: e.g 1xf2
        pdb_path: e.g ./pdb_raw_structures_folder/pdb1xf2.cif
    """ 

    print('getting pdb_str for %s,%s' % (pdb_id,pdb_path))

    if '.cif' in pdb_path:
        parser = MMCIFParser()
        structure = parser.get_structure(pdb_id, pdb_path)
        new_pdb = io.StringIO()
        pdb_io=PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(new_pdb)
        return new_pdb.getvalue()
    else:
        print('.cif must be in %s' % pdb_path)


def get_mean_bfactor(pdb_id, pdb_path):

    structure = MMCIF2Dict.MMCIF2Dict(pdb_path)
    b_factor = structure['_atom_site.B_iso_or_equiv']
    res = structure['_atom_site.label_comp_id']
    atom = structure['_atom_site.label_atom_id']

    b_factor_ca = []
    res_ca = []

    for i in range(0,len(atom)):
        if atom[i] == 'CA':
            b_factor_ca.append(float(b_factor[i]))
            res_ca.append(res[i])

    mean_bfactor_ca = np.mean(np.array(b_factor_ca))
    return(mean_bfactor_ca)

def convert_mmcif_to_pdb(cif_path, pdb_path):
    cmd.reinitialize()
    cmd.load(cif_path, 'protein')
    cmd.save(pdb_path, 'protein')
    cmd.delete('all')


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
 
    print('pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))

    if 'pred' in pdb1_source:
        print('ERROR: pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))
        raise ValueError('because we are superimposing pdb2 on pdb1, pdb1 should be from PDB while pdb2 should be a predicted structure or from PDB')

    if pdb1_source == 'pdb':
        pdb1_id = pdb1_full_id.split('_')[0]
        pdb1_chain = pdb1_full_id.split('_')[1] 
    else:
        pdb1_id = None
        pdb1_chain = None

    if pdb2_source == 'pdb':
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
        print('CLEANING PDB FILE %s' % pdb1_path)
        clean_pdb(pdb1_path_clean, pdb1_str)
        pdb1_input_path = pdb1_path_clean
    if (pdb2_path is not None) and (pdb2_source == 'pdb'):
        print('CLEANING PDB FILE %s' % pdb2_path)
        clean_pdb(pdb2_path_clean, pdb2_str)
        pdb2_input_path = pdb2_path_clean
    
    pdb2_output_path = '%s/%s.pdb' % (pdb_superimposed_folder, pdb2_model_name)
    shutil.copyfile(pdb2_input_path, pdb2_output_path)

    rmsd = align_and_get_rmsd(pdb1_input_path, pdb2_output_path, pdb1_chain, pdb2_chain)
    print('SAVING ALIGNED PDB AT %s' % pdb2_output_path)

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
 
    print('pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))

    if 'pred' in pdb1_source:
        print('ERROR: pdb1_source is %s and pdb2_source is %s' % (pdb1_source, pdb2_source))
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

    print(pdb1_path)
    print(pdb2_path)

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
        print('CLEANING PDB FILE %s' % pdb1_path)
        clean_pdb(pdb1_path_clean, pdb1_str)
        pdb1_input_path = pdb1_path_clean
    if (pdb2_path is not None) and (pdb2_source == 'pdb'):
        print('CLEANING PDB FILE %s' % pdb2_path)
        clean_pdb(pdb2_path_clean, pdb2_str)
        pdb2_input_path = pdb2_path_clean
 
    pdb2_output_path = '%s/%s.pdb' % (pdb_superimposed_folder, pdb2_model_name)
    shutil.copyfile(pdb2_input_path, pdb2_output_path)

    pdb1_chain = None
    pdb2_chain = None 
    rmsd = align_and_get_rmsd(pdb1_input_path, pdb2_output_path, pdb1_chain, pdb2_chain)
    print('SAVING ALIGNED PDB AT %s' % pdb2_output_path)

    return rmsd, pdb1_path, pdb2_path



'''
def _save_mmcif_file(
    prot: protein.Protein,
    output_dir: str,
    model_name: str,
    file_id: str,
    model_type: str,
) -> None:
  """Crate mmCIF string and save to a file.
     https://github.com/deepmind/alphafold/blob/6c4d833fbd1c6b8e7c9a21dae5d4ada2ce777e10/run_alphafold.py#L189
  Args:
    prot: Protein object.
    output_dir: Directory to which files are saved.
    model_name: Name of a model.
    file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    model_type: Monomer or multimer.
  """

  mmcif_string = to_mmcif(prot, file_id, model_type)

  # Save the MMCIF.
  mmcif_output_path = os.path.join(output_dir, f'{model_name}.cif')
  with open(mmcif_output_path, 'w') as f:
    f.write(mmcif_string)


def convert_pdb_to_mmcif(pdb_path: str, pdb_id: str, model_type: str, pdb_source: str) -> str:

    """
    Args:
        pdb_path: path to .pdb file 
    """ 

    output_dir = pdb_path[0:pdb_path.rindex('/')]
    model_name = pdb_path[pdb_path.rindex('/')+1:pdb_path.rindex('.pdb')]
    mmcif_output_path = os.path.join(output_dir, f'{model_name}.cif')

    if os.path.isfile(mmcif_output_path):  
        return mmcif_output_path
    else:
        #remove insertion code if file is sourced from pdb
        print('converting %s to %s' % (pdb_path, mmcif_output_path)) 
        if pdb_source == 'pdb':
            pdb_outpath = pdb_path.replace('.','_icrmv.')
            python_cmd = f'python {pdb_fixinsert_path} {pdb_path} > {pdb_outpath}' 
            subprocess.run(python_cmd, shell=True)
        else:
            pdb_outpath = pdb_path 

        with open(pdb_output_path, "r") as f:
            pdb_str = f.read()
        prot = from_pdb_string(pdb_str)
        _save_mmcif_file(prot, output_dir, model_name, pdb_id, model_type)
        return mmcif_output_path 
 
'''
