import argparse
import re 
import sys
import json
import shutil 

import parmed as pmd
from parmed.amber import AmberParm
from openmm import *
from openmm.app import *
from openmm.unit import *
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator

from pymol import cmd

from custom_openfold_utils.seq_utils import align_seqs, get_residues_idx_in_seq1_and_seq2 
from custom_openfold_utils.pdb_utils import get_pdb_path_seq 


def get_ca_position(positions, topology, target_residue_idx):

    try: 
        atom = next(atom for atom in topology.atoms() 
                    if atom.residue.index == target_residue_idx and atom.name == 'CA')
    except StopIteration:
        return None 
        #raise ValueError(f"CA atom for residue {target_residue_idx} not found")

    return positions[atom.index]

def relax_structure_wrt_target(af_pred_path, target_pdb_path, target_pdb_id, output_parent_dir):
    '''Assumes structures are already aligned'''

    input_dir = '%s/target_pdb_id=%s/input' % (output_parent_dir, target_pdb_id)
    os.makedirs(input_dir, exist_ok=True)

    af_fname = af_pred_path.split('/')[-1].split('.')[0]
    af_pred_path_copy = '%s/%s.pdb' % (input_dir, af_fname)
    shutil.copyfile(af_pred_path, af_pred_path_copy)
    target_fname = target_pdb_path.split('/')[-1].split('.')[0]
    target_pdb_path_copy = '%s/%s.pdb' % (input_dir, target_fname)
    shutil.copyfile(target_pdb_path, target_pdb_path_copy)

    af_seq = get_pdb_path_seq(af_pred_path_copy, None)
    target_seq = get_pdb_path_seq(target_pdb_path_copy, None)

    af_common_residues_idx, target_common_residues_idx = get_residues_idx_in_seq1_and_seq2(af_seq, target_seq)
    af_seq_aligned, target_seq_aligned, af_seq_aligned_to_original_idx_mapping, target_seq_aligned_to_original_idx_mapping = align_seqs(af_seq, target_seq)

    #print(af_common_residues_idx)
    #print(target_common_residues_idx)
    #print(af_seq_aligned)
    #print(target_seq_aligned)

    output_dir = '%s/target_pdb_id=%s/output' % (output_parent_dir, target_pdb_id)
    os.makedirs(output_dir, exist_ok=True)

    print('Selecting simulation platform')
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print('Using GPU:CUDA')
    except: 
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
            platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Using GPU:OpenCL')
        except:
            platform = openmm.Platform.getPlatformByName("CPU")
            print("Switching to CPU, no GPU available.")

    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    # Load the PDB file
    print('loading pdb %s' % af_pred_path_copy)
    af_pdb = PDBFile(af_pred_path_copy)
    modeller = Modeller(af_pdb.topology, af_pdb.positions)

    print('loading pdb %s' % target_pdb_path_copy)
    target_pdb = PDBFile(target_pdb_path_copy)
            
    # Create the system
    system = forcefield.createSystem(af_pdb.topology, 
                                     nonbondedMethod=NoCutoff, 
                                     nonbondedCutoff=10*angstroms, 
                                     constraints=HBonds, #SHAKE
                                     )


    AA=['ALA','ASP','CYS','GLU','PHE','GLY','HIS','HID','HIE','HIP','ILE','LYS','LEU','MET','ARG','PRO','GLN','ASN','SER','THR','VAL','TRP','TYR','CYX','LYN','ASH','GLH'] 
    # Minimize the energy with constraints on the protein
    # Apply restraints to protein heavy atoms and counter-ions
    restraint_force = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint_force.addGlobalParameter("k",10.*kilocalories_per_mole/angstroms**2)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')

    for i, res in enumerate(af_pdb.topology.residues()):
        if i in af_common_residues_idx:
            if res.name in AA:       
                print('****************************')
                print('AF residue: %s' % res.name)
                for atom in res.atoms():
                    if atom.name == 'CA': 
                        target_residue_idx = target_common_residues_idx[af_common_residues_idx.index(i)] 
                        print('Target PDB residue: %s' % target_seq[target_residue_idx]) 
                        target_ca_position = get_ca_position(target_pdb.positions, target_pdb.topology, target_residue_idx)
                        if target_ca_position is not None: 
                            print('Target CA position')
                            print(target_ca_position)
                            print('AF CA position')
                            print(af_pdb.positions[atom.index])
                            restraint_force.addParticle(atom.index,target_ca_position.value_in_unit(nanometers))

    system.addForce(restraint_force)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

    print('RELAXING STRUCTURE')
    simulation = Simulation(af_pdb.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=2000)

    af_relaxed_positions = simulation.context.getState(getPositions=True).getPositions()
    af_fname = af_pred_path_copy.split('/')[-1].split('.')[0]
    output_fname = '%s/%s_relaxed_wrt_%s.pdb' % (output_dir, af_fname, target_pdb_id)
    print('SAVING %s' % output_fname)
    PDBFile.writeFile(simulation.topology, af_relaxed_positions, open(output_fname, 'w'))

    return output_fname



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--af_pred_path", type=str, default=None
    )
    parser.add_argument(
        "--target_pdb_path", type=str, default=None
    )
    parser.add_argument(
        "--target_pdb_id", type=str, default=None
    )
    parser.add_argument(
        "--output_parent_dir", type=str, default=None
    )

    args = parser.parse_args()

    relax_structure_wrt_target(args.af_pred_path, args.target_pdb_path, args.target_pdb_id, args.output_parent_dir)
