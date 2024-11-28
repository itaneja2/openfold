#!/usr/bin/env python3
"""Generate Dendrogram from MPP and Lump."""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import os 
import shutil 
import argparse
from itertools import chain 
from functools import lru_cache
import click
import msmhelper as mh
import pandas as pd 
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, Normalize, LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram
from tqdm import tqdm
import colorsys
sys.setrecursionlimit(20000)
np.set_printoptions(threshold=sys.maxsize)

from pymol import cmd 

MIN_EDGE_WEIGHT = .1 


def get_b_factor_percentile(selection, percentile=95):
    b_factors = []
    cmd.iterate(selection, "b_factors.append(b)", space=locals())
    return np.percentile(b_factors, percentile)

#ignore rmsf_bar in selection
def save_frames_by_macrostate(rmsd_df, initial_pred_path, output_dir, frame_stride=10, save_raw_pdb_files=False):

    unique_macrostates = sorted(list(set(list(rmsd_df['macrostate']))))

    #corresponds to microstate that is not assigned a macrostate 
    if -1 in unique_macrostates:
        unique_macrostates.remove(-1)

    #unique_macrostates.remove(1)

    #get mapping between macrostate-cluster_num and md frames  
    md_frames_info = {}
    for macrostate in unique_macrostates:
        rmsd_df_subset = rmsd_df[rmsd_df['macrostate'] == macrostate]
        pdb_files_curr_macrostate = list(rmsd_df_subset['md_frame_path'])
        cluster_nums_curr_macrostate = list(rmsd_df_subset['cluster_num'])
        md_frames_info[macrostate] = {}
        for idx,pdb_file in enumerate(pdb_files_curr_macrostate):
            curr_fname = pdb_file[pdb_file.rindex('/')+1:]
            curr_cluster_num = cluster_nums_curr_macrostate[idx]
            if curr_cluster_num not in md_frames_info[macrostate]:
                md_frames_info[macrostate][curr_cluster_num] = []
            curr_frame_num = int((curr_fname.split('-')[0]).replace('frame',''))
            md_frames_info[macrostate][curr_cluster_num].append((curr_frame_num,curr_cluster_num,pdb_file))
   
    #get mapping between macrostate and relevant subset of md frames  
    md_frames_subset_info = {}  
    for macrostate in md_frames_info:
        md_frames_subset_info[macrostate] = [] 
        for cluster_num in md_frames_info[macrostate]:
            md_frames_info[macrostate][cluster_num] = sorted(md_frames_info[macrostate][cluster_num], key=lambda x:x[0])
            all_md_frames = md_frames_info[macrostate][cluster_num]
            num_md_frames = len(all_md_frames)
            idx_to_keep = [0]
            curr_frame = all_md_frames[0][0]
            if len(all_md_frames) == 1:
                continue
            for i in range(1,len(all_md_frames)):
                if (all_md_frames[i][0] - curr_frame) < frame_stride:
                    pass 
                else:
                    curr_frame = all_md_frames[i][0]
                    idx_to_keep.append(i)
            md_frames_subset_info[macrostate].extend([all_md_frames[i] for i in idx_to_keep])

    print(md_frames_subset_info[2])
                
    for macrostate in unique_macrostates:
        cmd.load(initial_pred_path, 'initial_AF_pred')
        cmd.color("white", 'initial_AF_pred', 0)
        print('on macrostate %d' % macrostate)
        pdb_files_curr_macrostate = md_frames_subset_info[macrostate]
        print('%d files for current macrostate' % len(pdb_files_curr_macrostate))
        for i in range(0,len(pdb_files_curr_macrostate)):
            if (i+1) % 10 == 0:
                print('loading file %d (%s)' % (i+1,pdb_files_curr_macrostate[i][-1]))
            curr_frame_num = pdb_files_curr_macrostate[i][0]
            curr_cluster_num = pdb_files_curr_macrostate[i][1]
            pdb_file = pdb_files_curr_macrostate[i][-1]
            object_name = '%s-%s' % (curr_cluster_num, curr_fname)
            cmd.load(pdb_file, object_name)
        selection = "all and not initial_AF_pred"
        percentile_97 = get_b_factor_percentile(selection, percentile=97)
        print(f"97th percentile of B-factors: {percentile_97}")
        cmd.show_as('cartoon', selection)
        cmd.cartoon('putty', selection)
        cmd.set("cartoon_putty_scale_min", 0, selection)
        cmd.set("cartoon_putty_scale_max", percentile_97, selection)
        cmd.set("cartoon_putty_transform", 0, selection)
        cmd.set("cartoon_putty_radius", 0.1, selection)
        cmd.spectrum("b", "rainbow", selection)
        cmd.ramp_new("rmsf_bar", cmd.get_object_list(selection)[0], [0, percentile_97], "rainbow")
        pymol_session_save_dir = '%s/pymol_sessions' % output_dir
        os.makedirs(pymol_session_save_dir, exist_ok=True)
        session_name = '%s/macrostate%d.pse.gz' % (pymol_session_save_dir, macrostate)
        print('saving %s' % session_name)
        cmd.save(session_name)
        cmd.reinitialize()
        if save_raw_pdb_files:
            pdb_save_dir = '%s/pdb_files/macrostate%d' % (output_dir, macrostate)
            os.makedirs(pdb_save_dir, exist_ok=True)
            for pdb_file in pdb_files_curr_macrostate:
                curr_fname = pdb_file[pdb_file.rindex('/')+1:]
                curr_cluster_num = cluster_nums_curr_macrostate[idx]
                pdb_destination_path = '%s/%s-%s' % (pdb_save_dir, curr_cluster_num, curr_fname)
                shutil.copyfile(pdb_file, pdb_destination_path)
                    

def plot_dendrogram(
    linkage_matrix_path,
    microstate_traj_path,
    rmsd_info_path,
    output_dir, 
    threshold,
    lag_steps,
    cut_params,
    hide_labels,
):
    """Plot MPP result."""
    # parse input and create output basename
    pop_thr, qmin_thr = cut_params
    output_file = (
        f'{linkage_matrix_path}.renamed_by_q.pop{pop_thr:.3f}_qmin{qmin_thr:.2f}'
    )

    # setup matplotlib
    pplt.use_style(figsize=2.6, figratio='golden', true_black=True)
    plt.rcParams['text.usetex'] = False

    # load transitions and sort them
    transitions = np.loadtxt(linkage_matrix_path)
    (
        linkage_mat,
        states_idx_to_microstates,
        states_idx_to_rootstates,
        labels,
    ) = _transitions_to_linkage(transitions, qmin=0)

    # get states
    nstates = len(linkage_mat) + 1
    print(nstates)
    states = np.unique(linkage_mat[:, :2].astype(int))
    print(linkage_mat)
    print(states) #states in linkage matrix 
    print(states_idx_to_microstates) #mapping each state in linkage matrix to all microstates under that node 

    # replace state names by their indices
    transitions_idx, states_idx = mh.rename_by_index(
        transitions[:, :2].astype(int),
        return_permutation=True,
    )
    transitions[:, :2] = transitions_idx

    # estimate population of states
    traj = mh.opentxt(microstate_traj_path)
    microstates, counts = np.unique(traj, return_counts=True)
    print('MICROSTATES')
    print(microstates)
    print(counts)
    print(len(traj))
    pops = counts / len(traj)
    print(pops)
    min_population = min(pops)
    print(states_idx_to_microstates)
    pops = {
        idx_state: np.sum([
            pops[microstates == state]
            for state in states_idx_to_microstates[idx_state]
        ])
        for idx_state in states
    } #cummulative population tree 
    print(pops)
    pops[2 * (nstates - 1)] = 1.0
    print(pops)

    # use population as edge widths
    edge_widths = {
        state: 10 * pops[state] for state in range(2 * nstates - 1)
    }

    '''
    for state in edge_widths:
        num_microstates_in_state = len(states_idx_to_microstates[state])
        print(state)
        print(states_idx_to_microstates[state])
        if num_microstates_in_state > 1:
            edge_widths[state] = min_population/10''' 
     
    #edge_widths[0] = .01
    print(edge_widths)

    # find optimal cut
    microstates, macrostates, macrostates_assignment, microstates_to_delete = mpp_plus_cut(
        states_idx_to_rootstates=states_idx_to_rootstates,
        states_idx_to_microstates=states_idx_to_microstates,
        linkage_mat=linkage_mat,
        microstates=microstates,
        pops=pops,
        pop_thr=pop_thr,
        qmin_thr=qmin_thr,
    )
    n_macrostates = len(macrostates_assignment)

    '''
    # estimate Q(state)
    q_of_t = mh.opentxt(qtraj, dtype=np.float32)
    q_state = {
        idx_state: _mean_val_per_state(
            states_idx_to_microstates[idx_state],
            q_of_t,
            traj,
        )
        for idx_state in tqdm(states)
    }''' 

    rmsd_df = pd.read_csv(rmsd_info_path)
    initial_pred_path = rmsd_df['initial_pred_path'][0]
    rmsd_vals = np.array(rmsd_df['rmsd_wrt_initial_pred'])

    rmsd_state = {
        idx_state: _mean_val_per_state(
            states_idx_to_microstates[idx_state],
            rmsd_vals,
            traj,
        )
        for idx_state in tqdm(states)
    }

    print('rsmd summary')
    print(rmsd_state)
 

    # define colors
    min_rmsd_color = 0
    max_rmsd_color = min(max(rmsd_state.values()),20)
    colors = {
        idx_state: _color_by_observable(rmsd_state[idx_state], min_rmsd_color, max_rmsd_color)
        for idx_state in states
    }
    # add global value
    colors[2 * (nstates - 1)] = _color_by_observable(max_rmsd_color, min_rmsd_color, max_rmsd_color)

    #print(colors)


    # define colors
    #color_list = generate_n_colors(len(states))

    #colors = {
    #    idx_state: color_list[idx_state]
    #    for idx_state in states
    #}
    # add global value
    #colors[2 * (nstates - 1)] = '#1f77b4'


    print('states')
    print(states)
    print('macrostates')
    print(macrostates)



    fig, (ax, ax_mat) = plt.subplots(
        2,
        1,
        gridspec_kw={
            'hspace': 0.05 if hide_labels else 0.3,
            'height_ratios': [9, 1],
        },
    )
    # hide spines of lower mat
    for key, spine in ax_mat.spines.items():
        spine.set_visible(False)

    dendrogram_dict = _dendrogram(
        ax=ax,
        linkage_mat=linkage_mat,
        colors=colors,
        threshold=threshold,
        labels=labels,
        qmin=0,
        edge_widths=edge_widths,
    )

    # plot legend
    cmap, bins = _color_by_observable(None, min_rmsd_color, max_rmsd_color)
    norm = Normalize(bins[0], bins[-1])
    label = 'Mean RMSD per state'

    cmappable = ScalarMappable(norm, cmap)
    plt.sca(ax)
    pplt.colorbar(cmappable, width='5%', label=label, position='top')

    yticks = np.arange(0.5, 1.5 + n_macrostates)
    
    xticks = 10 * np.arange(0, nstates + 1)
    print(nstates)
    print(xticks)
    cmap = LinearSegmentedColormap.from_list(
        'binary', [(0, 0, 0, 0), (0, 0, 0, 1)],
    )

    print('macrostate assignemnt dendrogram pre')
    print(macrostates)
    print('microstates assignment dendrogram pre')
    print(microstates)

    print(dendrogram_dict)

    # permute macrostate assignment and label them
    macrostates_assignment = macrostates_assignment.T[
        dendrogram_dict['leaves']
    ].T
    macrostates = macrostates[dendrogram_dict['leaves']]
    microstates = microstates[dendrogram_dict['leaves']]

    print('macrostates dendrogram post')
    print(macrostates)
    print('microstates assignment dendrogram post')
    print(microstates)


    # apply dynamical correction of minor branches
    dyn_corr_macrostates = mpp_plus_dyn_cor(
        macrostates=macrostates,
        microstates=microstates,
        n_macrostates=n_macrostates,
        pops=pops,
        traj=traj,
        lag_steps=lag_steps,
    )

    print('dyn_corr')
    print(dyn_corr_macrostates)

    # rename macrostates by fraction of native contacts

    #in traj, replaces microstates with dyn_corr_macrostates
    microstates_w_deletion = np.append(microstates, microstates_to_delete)
    dyn_corr_macrostates_w_deletion = np.append(dyn_corr_macrostates,[-1]*len(microstates_to_delete))
    print(microstates_w_deletion)
    print(dyn_corr_macrostates_w_deletion)

    macrotraj = mh.shift_data(traj, microstates_w_deletion, dyn_corr_macrostates_w_deletion)

    print(microstates)
    print(dyn_corr_macrostates)

    print('before')
    print(traj.shape)
    print(traj[0:3300])
    print('***')
    print(macrotraj.shape)
    print(macrotraj[0:3300])

    print('unsorted macrostates')
    print(macrostates)
    print(dyn_corr_macrostates)

    print('mean val per state')
    print(np.unique(dyn_corr_macrostates))
    tmp = [_mean_val_per_state([state], rmsd_vals, macrotraj) for state in np.unique(dyn_corr_macrostates)]
    print(tmp)
    
    macrostates_sorted_by_rmsd = [
        _mean_val_per_state([state], rmsd_vals, macrotraj)
        for state in np.unique(dyn_corr_macrostates)
    ]

    macroperm = np.unique(dyn_corr_macrostates)[np.argsort(macrostates_sorted_by_rmsd)]
    print('sorted macrostates')
    print(dyn_corr_macrostates)
    print(macroperm)
    print(np.unique(dyn_corr_macrostates))
    dyn_corr_macrostates = mh.shift_data(
        dyn_corr_macrostates, macroperm, np.unique(dyn_corr_macrostates),
    )
    print(dyn_corr_macrostates) #just renaming so each entry in macrostate corresponds to its relative ranking of rmsd 
    print('**')



    mh.savetxt(
        f'{output_file}.macrostates',
        np.array([microstates, dyn_corr_macrostates]).T,
        header='microstates macrostates',
        fmt='%.0f',
    )

    microstates_w_deletion = np.append(microstates, microstates_to_delete)
    dyn_corr_macrostates_w_deletion = np.append(dyn_corr_macrostates,[-1]*len(microstates_to_delete))
 
    macrotraj = mh.shift_data(traj, microstates_w_deletion, dyn_corr_macrostates_w_deletion)
    
    print('after sorting')
    print(microstates)
    print(dyn_corr_macrostates)
    print(macrotraj.shape)
    print(macrotraj[0:3300])

    rmsd_df['macrostate'] = macrotraj
    rmsd_df_save_path = rmsd_info_path.replace('rmsd_df.csv', 'rmsd_w_macrostate_info_df.csv') 
    print(rmsd_df_save_path)
    rmsd_df.to_csv(rmsd_df_save_path, index=False)

    print('SAVING FRAMES BY MACROSTATE')
    save_frames_by_macrostate(rmsd_df, initial_pred_path, output_dir) 

    '''# print final sorting order
    macrostates_sorted_by_rmsd = [
        _mean_val_per_state([state], rmsd_vals, macrotraj)
        for state in np.unique(macrostates)
    ]'''

    mh.savetxt(
        f'{output_file}.macrotraj',
        macrotraj,
        header='macrostates',
        fmt='%.0f',
    )
   
    print(dyn_corr_macrostates) 
    print(np.unique(dyn_corr_macrostates))
    print(macrostates_assignment)
    # recalculate macrostates_assignment)
    for idx, mstate in enumerate(np.unique(dyn_corr_macrostates)):
        macrostates_assignment[idx] = dyn_corr_macrostates == mstate

    print('recalc')
    print(macrostates_assignment)

   
    #calculates midpoint of each tick  
    xvals = 0.5 * (xticks[:-1] + xticks[1:])

    for idx, assignment in enumerate(macrostates_assignment):

        macrostates_idx = np.where(assignment == 1)[0]
        print(macrostates_idx)
        xmean = np.median(xvals[assignment == 1])

        print('*')
        print((xmean,idx,f'{idx + 1:.0f}',dyn_corr_macrostates[macrostates_idx][0]))

        pplt.text(
            xmean,
            yticks[idx] - (yticks[1] - yticks[0]),
            #f'{idx + 1:.0f}',
            dyn_corr_macrostates[macrostates_idx][0],
            ax=ax_mat,
            va='top',
            contour=True,
            size='small',
        )

    '''
    for idx,val in enumerate(dyn_corr_macrostates):
        xmean = xvals[idx]
        pplt.text(
            xmean,
            yticks[idx] - (yticks[1] - yticks[0]),
            dyn_corr_macrostates[idx],
            ax=ax_mat,
            va='top',
            contour=True,
            size='small',
        )

    for idx, assignment in enumerate(macrostates_assignment):
        xmean = np.median(xvals[assignment == 1])

        print('*')
        print((xmean,idx,f'{idx + 1:.0f}',dyn_corr_macrostates[idx]))

        pplt.text(
            xmean,
            yticks[idx] - (yticks[1] - yticks[0]),
            f'{idx + 1:.0f}',
            ax=ax_mat,
            va='top',
            contour=True,
            size='small',
        )
    '''
    
    ax_mat.pcolormesh(
        xticks,
        yticks,
        macrostates_assignment,
        snap=True,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # set x-labels
    ax_mat.set_yticks(yticks)
    ax_mat.set_yticklabels([])
    ax_mat.grid(visible=True, axis='y', ls='-', lw=0.5)
    ax_mat.tick_params(axis='y', length=0, width=0)
    ax_mat.set_xlim(ax.get_xlim())
    ax.set_xlabel('')
    ax_mat.set_xlabel('macrostates')
    ax_mat.set_ylabel('')
    fig.align_ylabels([ax, ax_mat])

    ax_mat.set_xticks(np.arange(0.5, 0.5 + len(states)))

    #ax_mat.grid(visible=False, axis='y')
    #ax_mat.grid(visible=False, axis='x')

    if hide_labels:
        for axes in (ax,ax_mat):  # if statemat_file else [ax]:
            axes.set_xticks([])
            axes.set_xticks([], minor=True)
            axes.set_xticklabels([])
            axes.set_xticklabels([], minor=True)

    print(f'saving {output_file}')
    pplt.savefig(f'{output_file}.pdf')


def _color_by_observable(observable, omin, omax, steps=10):
    cmap = plt.get_cmap('plasma_r', steps)
    colors = [cmap(idx) for idx in range(cmap.N)]

    bins = np.linspace(
        omin, omax, steps + 1,
    )

    if observable is None:
        return cmap, bins

    for color, rlower, rhigher in zip(colors, bins[:-1], bins[1:]):
        if rlower <= observable <= rhigher:
            return color
    
    if observable > max(bins):
        return colors[-1]
    if observable < min(bins):
        return colors[0]

    return 'k'


def _mean_val_per_state(states, observable, traj):
    #observable corresponds to some property quantified from MD sims (i.e rmsd, fraction of native contacts, etc.)
    if len(states) >= 1:
        mask = np.full(observable.shape[0], False)
        for state in states:
            mask = np.logical_or(
                mask,
                traj == state,
            )
        observable = observable[mask]

    return np.mean(observable)


def _transitions_to_linkage(trans, *, qmin=0.0):
    """Convert transition matrix to linkage matrix.

    Parameters
    ----------
    transitions: ndarray of shape (nstates - 1, 3)
        Three column: merged state, remaining state, qmin lebel.

    qmin: float [0, 1]
        Qmin cut-off. Returns only sublinkage-matrix.

    """
    transitions = np.copy(trans)
    states = np.unique(transitions[:, :2].astype(int))

    # sort by merging qmin level
    transitions = transitions[
        np.argsort(transitions[:, 2])
    ]

    # create linkage matrix
    mask_qmin = transitions[:, 2] > qmin
    nstates_qmin = np.count_nonzero(mask_qmin) + 1
    linkage_mat = np.zeros((nstates_qmin - 1, 4))

    # replace state names by their indices
    transitions_idx, states_idx = mh.rename_by_index(
        transitions[:, :2][mask_qmin].astype(int),
        return_permutation=True,
    )
    transitions[:, :2][mask_qmin] = transitions_idx
    linkage_mat[:, :3] = transitions[mask_qmin]

    # holds for each state (index) a list corresponding to the microstates
    # it consist of.
    states_idx_to_microstates = {
        idx: [
            state,
            *transitions[~mask_qmin][:, 0][
                transitions[~mask_qmin][:, 1] == state
            ].astype(int),
        ]
        for idx, state in enumerate(states_idx)
    }
    states_idx_to_rootstates = {
        idx: [idx]
        for idx, _ in enumerate(states_idx)
    }

    for idx, nextstate in enumerate(
        range(nstates_qmin, 2 * nstates_qmin - 1),
    ):
        statefrom, stateto = linkage_mat[idx, :2].astype(int)
        states_idx_to_microstates[nextstate] = [
            *states_idx_to_microstates[stateto],
            *states_idx_to_microstates[statefrom],
        ]
        states_idx_to_rootstates[nextstate] = [
            *states_idx_to_rootstates[stateto],
            *states_idx_to_rootstates[statefrom],
        ]

        states = linkage_mat[idx, :2].astype(int)
        for state in states:
            linkage_mat[idx + 1:, :2][
                linkage_mat[idx + 1:, :2] == state
            ] = nextstate

    labels = [
        states_idx_to_microstates[idx][0]
        for idx in range(nstates_qmin)
    ]

    return (
        linkage_mat,
        states_idx_to_microstates,
        states_idx_to_rootstates,
        labels,
    )


def _dendrogram(
    *, ax, linkage_mat, colors, threshold, labels, qmin, edge_widths,
):
    nstates = len(linkage_mat) + 1
    # convert color dictionary to array
    colors_arr = np.array(
        [
            to_hex(colors[state]) for state in range(2 * nstates - 1)
        ],
        dtype='<U7',
    )

    dendrogram_dict = dendrogram(
        linkage_mat,
        leaf_rotation=90,
        get_leaves=True,
        color_threshold=1,
        link_color_func=lambda state_idx: colors_arr[state_idx],
        no_plot=True,
    )
    _plot_dendrogram(
        icoords=dendrogram_dict['icoord'],
        dcoords=dendrogram_dict['dcoord'],
        ivl=dendrogram_dict['ivl'],
        color_list=dendrogram_dict['color_list'],
        threshold=threshold,
        ax=ax,
        colors=colors_arr,
        labels=labels,
        qmin=qmin,
        edge_widths=edge_widths,
    )

    ax.set_ylabel('metastability Qmin')
    ax.set_xlabel('microstates')
    ax.grid(visible=False, axis='x')

    return dendrogram_dict

'''
def _show_xlabels(*, ax, states_perm):
    """Show the xticks together with the corresponding state names."""
    # undo changes of scipy dendrogram
    xticks = ax.get_xticks()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for line in ax.get_xticklines():
        line.set_visible(True)

    for is_major, length_scale in ((True, 4), (False, 1)):
        ax.tick_params(
            axis='x',
            length=length_scale * plt.rcParams['xtick.major.size'],
            labelrotation=90,
            pad=2,
            labelsize='xx-small',
            width=plt.rcParams['xtick.major.width'],
            which='major' if is_major else 'minor',
            top=False,
        )
        offset = 0 if is_major else 1
        ax.set_xticks(xticks[offset::2], minor=not is_major)
        ax.set_xticklabels(states_perm[offset::2], minor=not is_major)
'''

def _plot_dendrogram(
    *,
    icoords,
    dcoords,
    ivl,
    color_list,
    threshold,
    ax,
    colors,
    labels,
    qmin,
    edge_widths,
):
    """Plot dendrogram with colors at merging points."""
    threshold_color = to_hex('pplt:grey')
    # Independent variable plot width
    ivw = len(ivl) * 10
    # Dependent variable plot height
    dvw = 1.05

    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)

    ax.set_ylim([qmin, dvw])
    ax.set_xlim([-0.005 * ivw, 1.005 * ivw])
    ax.set_xticks(iv_ticks)

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(np.asarray(labels)[np.asarray(ivl).astype(int)])

    get_ancestor = _get_ancestor_func(
        icoords, dcoords, ivl,
    )

    # Let's use collections instead. This way there is a separate legend item
    # for each tree grouping, rather than stupidly one for each line segment.
    colors_used = np.unique(colors)
    color_to_lines = {color: [] for color in (*colors_used, threshold_color)}
    width_to_lines = {color: [] for color in (*colors_used, threshold_color)}
    for xline, yline, color in zip(icoords, dcoords, color_list):
        if np.max(yline) <= threshold:
            # split into left and right to color separately
            xline_l = [*xline[:2], np.mean(xline[1:3])]
            xline_r = [np.mean(xline[1:3]), *xline[2:]]

            color_l = _get_ancestor_color(
                icoords,
                dcoords,
                xline[0],
                yline[0],
                color_list,
                ivl,
                colors,
            )
            ancestors_l = get_ancestor(xline[0], yline[0])
            if len(ancestors_l) == 1:
                weight_l = np.sum([
                    edge_widths[ancestor] for ancestor in ancestors_l
                ])
            else:
                weight_l = MIN_EDGE_WEIGHT
            color_r = _get_ancestor_color(
                icoords,
                dcoords,
                xline[3],
                yline[3],
                color_list,
                ivl,
                colors,
            )
            ancestors_r = get_ancestor(xline[3], yline[3])
            if len(ancestors_r) == 1:
                weight_r = np.sum([
                    edge_widths[ancestor] for ancestor in ancestors_r
                ])
            else:
                weight_r = MIN_EDGE_WEIGHT
            color_to_lines[color_l].append(list(zip(xline_l, yline[:3])))
            width_to_lines[color_l].append(
                max(weight_l, MIN_EDGE_WEIGHT),
            )
            color_to_lines[color_r].append(list(zip(xline_r, yline[1:])))
            width_to_lines[color_r].append(
                max(weight_r, MIN_EDGE_WEIGHT),
            )

        elif np.min(yline) >= threshold:
            color_to_lines[threshold_color].append(list(zip(xline, yline)))
        else:
            yline_bl = [yline[0], np.max([threshold, yline[1]])]
            yline_br = [np.max([threshold, yline[2]]), yline[3]]
            color_to_lines[color].append(list(zip(xline[:2], yline_bl)))
            color_to_lines[color].append(list(zip(xline[2:], yline_br)))

            yline_thr = np.where(np.array(yline) < threshold, threshold, yline)
            color_to_lines[threshold_color].append(list(zip(xline, yline_thr)))

    # Construct the collections.
    colors_to_collections = {
        color: LineCollection(
            color_to_lines[color], colors=(color,),
            linewidths=width_to_lines[color],
        )
        for color in (*colors_used, threshold_color)
    }

    # Add all the groupings below the color threshold.
    for color in colors_used:
        ax.add_collection(colors_to_collections[color])
     #If there's a grouping of links above the color threshold, it goes last.
    ax.add_collection(colors_to_collections[threshold_color])


def _get_ancestor_color(
    xlines, ylines, xval, yval, color_list, ivl, colors,
):
    """Get the color of the ancestors."""
    # if ancestor is root
    if not yval:
        ancestor = int(ivl[int((xval - 5) // 10)])
        return colors[ancestor]

    # find ancestor color
    xy_idx = np.argwhere(
        np.logical_and(
            np.array(ylines)[:, 1] == yval,
            np.array(xlines)[:, 1:3].mean(axis=1) == xval,
        ),
    )[0][0]
    return color_list[xy_idx]


def _get_ancestor_func(
    xlines, ylines, ivl,
):
    """Get the color of the ancestors."""
    @lru_cache(maxsize=1024)
    def _get_ancestor_rec(xval, yval):
        # if ancestor is root
        if not yval:
            ancestor = int(ivl[int((xval - 5) // 10)])
            return (ancestor, )

        # find ancestor color
        xy_idx = np.argwhere(
            np.logical_and(
                np.array(ylines)[:, 1] == yval,
                np.array(xlines)[:, 1:3].mean(axis=1) == xval,
            ),
        )[0][0]
        xleft, yleft = xlines[xy_idx][0], ylines[xy_idx][0]
        xright, yright = xlines[xy_idx][3], ylines[xy_idx][3]

        return (
            *_get_ancestor_rec(xleft, yleft),
            *_get_ancestor_rec(xright, yright),
        )

    return _get_ancestor_rec


def state_sequences(macrostates, state):
    """Get continuous index sequences of macrostate in mstate assignment."""
    state_idx = np.where(macrostates == state)[0]
    idx_jump = state_idx[1:] - state_idx[:-1] != 1
    return np.array_split(
        state_idx,
        np.nonzero(idx_jump)[0] + 1,
    )


def mpp_plus_cut(
    *,
    states_idx_to_rootstates,
    states_idx_to_microstates,
    linkage_mat,
    microstates,
    pops,
    pop_thr,
    qmin_thr,
):
    """Apply MPP+ step1: Identify branches."""
    nstates = len(linkage_mat) + 1

    macrostates_set = [
        set(states_idx_to_rootstates[2 * (nstates - 1)]),
    ]
    macrostates_leaf_set = [
        set(states_idx_to_microstates[2 * (nstates - 1)]),
    ]

    print(macrostates_set)
    print(macrostates_leaf_set)
    print(linkage_mat)

    for state_i, state_j, qmin in reversed(linkage_mat[:, :3]):
        if pops[state_i] > pop_thr and pops[state_j] > pop_thr and qmin > qmin_thr:
            mstate_i = set(states_idx_to_rootstates[state_i])
            macrostates_set = [
                mstate - mstate_i
                for mstate in macrostates_set
            ]
            macrostates_set.append(mstate_i)

            mstate_leaf_i = set(states_idx_to_microstates[state_i])
            macrostates_leaf_set = [
                mstate - mstate_leaf_i
                for mstate in macrostates_leaf_set
            ]
            macrostates_leaf_set.append(mstate_leaf_i)
            print('&&&')
            print(macrostates_set)
            print(macrostates_leaf_set)
            print('&&&')


    n_macrostates = len(macrostates_set)
    macrostates_assignment = np.zeros((n_macrostates, nstates))

    for idx, mstate in enumerate(macrostates_set):
        macrostates_assignment[idx][list(mstate)] = 1
    #each row corresponds to a macrostate, column represents which microstates belong to that macrostate
    print(macrostates_set)
    print(macrostates_assignment.shape)
    print(macrostates_assignment)
    print(microstates)
    
    macrostates = np.empty(len(microstates), dtype=np.int64)
    delete_idx = []
    microstates_to_delete = []  
    for idx, microstate in enumerate(microstates):
        for idx_m, macroset in enumerate(macrostates_leaf_set):
            if microstate in macroset:
                macrostates[idx] = idx_m + 1
                print(f'{microstate} in macrostate {macroset}')
                break
        else:
            microstates_to_delete.append(microstate)
            delete_idx.append(idx) 
            print(f'{microstate} not in any macrostate')

    print(macrostates)
    if len(delete_idx) > 0:
        macrostates = np.delete(macrostates, delete_idx)
        microstates = np.delete(microstates, delete_idx)
    print(macrostates)
    print(microstates)

    return microstates, macrostates, macrostates_assignment, microstates_to_delete


def mpp_plus_dyn_cor(
    *,
    macrostates,
    microstates,
    n_macrostates,
    pops,
    traj,
    lag_steps,
):
    """Apply MPP+ step2: Dynamically correct minor branches."""
    # fix dynamically missassigned single-state branches
    # identify them
    dyn_corr_macrostates = macrostates[:]
    for mstate in np.unique(macrostates):
        idx_sequences = state_sequences(macrostates, mstate)
        if len(idx_sequences) > 1:
            highest_pop_sequence = np.argmax([
                np.sum([
                    pops[s] for s in microstates[seq]
                ]) for seq in idx_sequences
            ])
            idx_sequences = [
                seq for idx, seq in enumerate(idx_sequences)
                if idx != highest_pop_sequence
            ]
            for seq in idx_sequences:
                largest_state = np.max(dyn_corr_macrostates)
                for newstate, seq_idx in enumerate(
                    seq,
                    largest_state + 1,
                ):
                    dyn_corr_macrostates[seq_idx] = newstate

    # dynamically reassign all new state to previous macrostates
    mstates = np.unique(dyn_corr_macrostates)
    while len(mstates) > n_macrostates:
        tmat, mstates = mh.msm.estimate_markov_model(
            mh.shift_data(traj, microstates, dyn_corr_macrostates),
            lagtime=lag_steps,
        )

        # sort new states by increasing metastability
        qs = np.diag(tmat)[n_macrostates:]
        idx_sort = np.argsort(qs)
        newstates = mstates[n_macrostates:][idx_sort]

        deletestate = newstates[0]

        # reassign them
        idx = np.where(mstates == deletestate)[0][0]
        idxs_to = np.argsort(tmat[idx])[::-1]
        for idx_to in idxs_to:
            if idx_to == idx:
                continue
            dyn_corr_macrostates[
                dyn_corr_macrostates == deletestate
            ] = mstates[idx_to]
            break

        mstates = np.unique(dyn_corr_macrostates)

    return dyn_corr_macrostates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linkage_matrix_path", type=str, default=None, help='Path to linkage matrix created by MPP, named **_transitions.dat'
    )
    parser.add_argument(
        "--microstate_traj_path", type=str, default=None, help='Path to file where each frame in trajectory is labelled with its microstate'
    )
    parser.add_argument(
        "--rmsd_info_path", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default=None 
    )
    parser.add_argument(
        "--color_threshold", type=float, default=1.0
    )
    parser.add_argument(
        "--lag_steps", type=int, default=None, help='Lagtime in frames'
    )
    parser.add_argument(
        "--cut_params", type=float, nargs=2
    )
    parser.add_argument(
        "--hide_labels", type=bool, default=True
    )
    
    args = parser.parse_args()

    plot_dendrogram(args.linkage_matrix_path, args.microstate_traj_path, args.rmsd_info_path, args.output_dir, args.color_threshold, args.lag_steps, args.cut_params, args.hide_labels)
