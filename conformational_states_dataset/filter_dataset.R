library(dplyr)
library(ggplot2)

home_dir = '/home/ishan/Lasker/openfold/conformational_states_dataset'

conformational_states_df = read.csv(paste(home_dir,'data/conformational_states_df.csv',sep='/'))
conformational_states_df$seg_len = abs(conformational_states_df$seg_end-conformational_states_df$seg_start)+1
conformational_states_df$rmsd_normalized = conformational_states_df$rmsd_wrt_pdb_id_ref/conformational_states_df$seg_len

ggplot(conformational_states_df, aes(x=rmsd_wrt_pdb_id_ref)) + 
  stat_ecdf(color="black")

ggplot(conformational_states_df %>% filter(rmsd_wrt_pdb_id_ref <= 50) %>% as.data.frame, aes(x=rmsd_wrt_pdb_id_ref)) + 
  stat_ecdf(color="black")

ggplot(conformational_states_df, aes(x=rmsd_normalized)) + 
  stat_ecdf(color="black")

ggplot(conformational_states_df %>% filter(rmsd_normalized <= 1) %>% as.data.frame, aes(x=rmsd_normalized)) + 
  stat_ecdf(color="black")



########

conformational_states_df_filtered_rmsd_seg_len = conformational_states_df %>% filter(rmsd_wrt_pdb_id_ref >= 15)  %>% filter(seg_len >= 150) %>% filter(seg_len <= 500) %>% as.data.frame()

#manual filtering criteria:
#significant conformational change NOT confined to a (predicted disordered) linker or tail OR a minor isolated loop change 
#OR variation in the number of visible residues between states A and B


ggplot(conformational_states_df_filtered_rmsd_seg_len, aes(x=seg_len)) + 
  stat_ecdf(color="black")

ggplot(conformational_states_df_filtered_rmsd_seg_len, aes(x=rmsd_wrt_pdb_id_ref)) + 
  stat_ecdf(color="black")

ggplot(conformational_states_df_filtered_rmsd_seg_len, aes(x=rmsd_normalized)) + 
  stat_ecdf(color="black")

write.csv(conformational_states_df_filtered_rmsd_seg_len, paste(home_dir,'data/conformational_states_filtered_df.csv',sep='/'), row.names=FALSE)



