#Usage: python group_gene_expr.py sample_base_name genome_accession extrapolate_tpm
from glob import glob
import subprocess
import pandas as pd
import numpy as np
import os, sys, re
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats.mstats import kruskal
from Bio.Seq import Seq
from Bio import SeqIO
from tqdm import tqdm
from pysam import FastaFile

def get_tpm(file):
    dat = pd.read_table(file)
    dat = dat.drop_duplicates(subset='target_id')
    dat = dat[['target_id','tpm']]
    dat.index = dat['target_id']
    dat = dat.drop(columns = ['target_id'])
    return dat

def scale_func2(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def add_p_val(lft,rgt,y,h,p):
    plt.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    plt.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def plot_line_groups(mat,tpm_g,png_name,fig_title, hue_order):
    tot = mat.shape[1]
    xlab = [''] * tot
    xlab[0] = f'-{tot//2}'
    xlab[tot-1] =  f'+{tot//2}'
    xlab[tot//2] = 'TSS'
    pdFrame = pd.DataFrame(mat)
    pdFrame = pdFrame.assign(group=tpm_g)
    pdFrame_melt = pd.melt(pdFrame, id_vars = 'group',var_name='pos', value_name='read_cv')
    lines = sns.lineplot(data=pdFrame_melt, x="pos", y="read_cv", hue='group', hue_order=hue_order)
    lines.set_xticks(range(len(set(pdFrame_melt.pos))))
    lines.set_xticklabels(xlab)
    lines.tick_params(bottom=False)
    lines.axvline(x=tot//2, color = 'red')
    lines.set(title=fig_title)
    lines.get_figure().savefig(png_name)
    plt.clf()
    
def get_upstream_dist(tss_dat,ref):
    dist_2_gene = []
    for genome in list(set(tss_dat.Genome)):
        dat = tss_dat[tss_dat.Genome==genome]
        genome_len = reference_genome.get_reference_length(genome)
        for i, row in dat.iterrows():
            if row['Strand'] == '-':
                if i == dat.index[-1]:
                    dist_2_gene.append(genome_len - row['Start'])
                    continue
                dist_2_gene.append(dat.loc[i+1,'geneLeft'] - row['Start'])
            else:
                if i == dat.index[0]:
                    dist_2_gene.append(row['Start'])
                    continue
                dist_2_gene.append(row['Start'] - dat.loc[i-1,'geneRight'])
    return(dist_2_gene)

def taxid2ftp(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    ftp = [re.sub(r'^ftp','https',x) for x in ftp]
    return(ftp)

def asm_2_species(asm):
    name = refseq[refseq['ftp_path'].str.contains(asm)]['organism_name'].item()
    return re.match(r'(^[^ ]+ [^ ]+).*',name).group(1)

def prepare_species_folder(ftp):
    asm = ftp.rsplit('/',1)[1]
    tax_path = f'strains/{asm}'
    if not os.path.exists(tax_path):
        os.mkdir(tax_path)
        #Download genome ncbi
        subprocess.call(f'wget {ftp}/{asm}_genomic.fna.gz -O {tax_path}/{asm}.fna.gz', shell= True)
        subprocess.call(f'gzip -df {tax_path}/{asm}.fna.gz', shell= True)

        #Predict genes with prodigal
        subprocess.call(f'prodigal -i {tax_path}/{asm}.fna -o {tax_path}/{asm}_prodigal.gff -d {tax_path}/{asm}_prodigal.fna -f gff', shell= True)

        ###TSS filtering - Remove overlaping TSS (operon start), Allow overlaping with gene bodies
        tss_dat = pd.read_csv(f'{tax_path}/{asm}_prodigal.gff', sep = '\t', header=None, comment = '#')
        tss_dat.drop(columns=[1,2,5,7], inplace = True)
        tss_dat['TSS'] = [x[3] if x[6] == '+' else x[4] for i,x in tss_dat.iterrows()]
        tss_dat = np.array(tss_dat)
        #Remove overlaping TSS -TSS doesnt overlap with TSS
        non_overlap = []
        for chrom in list(set(tss_dat[:,0])):
            subset = tss_dat[np.in1d(tss_dat[:,0],chrom)]
            if(len(subset) == 1):
                non_overlap.append(subset[0])
                continue
            #First case
            if float(subset[0,5])+500 < float(subset[1,5]):
                non_overlap.append(subset[0])
            #Mid case
            for i in range(1,len(subset)-1):
                if float(subset[i,5])-500 > float(subset[i-1,5]) and float(subset[i,5])+500 < float(subset[i+1,5]):
                    non_overlap.append(subset[i])
            #Last case
            if float(subset[len(subset)-1,5])-500 > float(subset[len(subset)-2,5]):
                non_overlap.append(subset[len(subset)-1])
        non_overlap = pd.DataFrame(non_overlap, columns = ["Genome","geneLeft","geneRight","Strand","Desc","Start"])
        non_overlap.to_csv(f'{tax_path}/{asm}_filtered_tss.csv', index = False)

        #Index genome
        genome_fna = f'{tax_path}/{asm}.fna'
        subprocess.call(f'bowtie2-build {genome_fna} {genome_fna}', shell=True)
    return(asm)

with open('matched_samples_8') as f:
    samples = f.read().splitlines()
samp_bn = [x.rsplit('/',1)[1] for x in samples]
RNA_path = '/groups/cgsd/gordonq/TSS_depth/matched_public_data/RNA_primary/'

#Read species abundances
mpa_res = pd.read_table('mpa_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
mpa_res = mpa_res.loc[:,samp_bn]
mpa_res = mpa_res.loc[((mpa_res > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 5 samples
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop('39491') #Eubacterium Rectale no reference in database.

#refseq = pd.read_table("/groups/cgsd/gordonq/database/prokaryotes.txt", header=0)
refseq = pd.read_table('/groups/cgsd/gordonq/database/assembly_summary_refseq.txt', header = 1)

xlab = [''] * 1001
xlab[0] = f'-{1001//2}'
xlab[1001-1] =  f'+{1001//2}'
xlab[1001//2] = 'TSS'
'''
all_df = pd.DataFrame()
for taxid in mpa_res.index:
    ftps = taxid2ftp(taxid,3)
    for ftp in ftps:
        asm = prepare_species_folder(ftp)
        for samp in samp_bn:
            if np.isnan(mpa_res.loc[taxid,samp]):
                continue
            #Quantify RNA
            if not os.path.exists(f'kallisto_res/{asm}_{samp}'):
                cds_fna = f'strains/{asm}/{asm}_prodigal.fna'
                subprocess.call(f'kallisto index {cds_fna} -i strains/{asm}/{asm}_kallisto.db', shell = True)
                subprocess.call(f'kallisto quant -t 64 -i strains/{asm}/{asm}_kallisto.db -o kallisto_res/{asm}_{samp} {RNA_path}{samp}_1.fastq {RNA_path}{samp}_2.fastq', shell = True)
            #Read TPM sample data
            tpm_file = get_tpm(f'kallisto_res/{asm}_{samp}/abundance.tsv')
            chrom_map = pd.read_table(f'strains/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
            chrom_map = chrom_map[~chrom_map.index.duplicated(keep='first')]
            chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom,'Desc']).group(1) for chrom in chrom_map.index}
            tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
            tpm_dict = tpm_file.to_dict()

            mat = pd.read_table(f'matrices/{asm}_{samp}.csv', sep = ',', header=None, index_col=0)
            mat = mat.loc[mat.sum(axis=1)!=0,:]
            tpm_val = [tpm_dict['tpm'].get(x,np.nan) for x in mat.index]
            nan = pd.Series(tpm_val).notnull()
            mat = mat.loc[nan.tolist(),:]
            tpm_val = pd.Series(tpm_val)[nan]
            tpm_val = tpm_val.reset_index(drop=True)
            tpm_color = pd.Series(np.where(tpm_val == 0, "zero", np.where(tpm_val < 100, 'low', np.where(tpm_val < 1000, 'mid', 'high'))))

            ###Get Intergenic distances
            tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
            reference_genome = FastaFile(f'strains/{asm}/{asm}.fna')
            tss_dat['upstream_dist'] = get_upstream_dist(tss_dat,reference_genome)
            tss_dat['operon_state'] = ['short' if x <=50 else 'long' for x in tss_dat['upstream_dist']]
            tss_dat.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat['Desc']])

            ### Box/Line plot, group by TPM expression ###
            scaled_mat = scale_func2(mat)
            bias_vec = list(map(calc_bias, np.array(scaled_mat)))
            df = pd.DataFrame({'log_tpm':np.log(tpm_val + 0.001).tolist(),'tpm_group':tpm_color.tolist(),'bias':bias_vec, 'samp':samp, 'asm':asm, 'upstream_dist':tss_dat.loc[scaled_mat.index,'operon_state']}, index = scaled_mat.index)
            #plot_line_groups(scaled_mat,tpm_color.tolist(),f'line_plots/{asm}_{samp}_mean.pdf',"Read Coverage Mean", ['zero','low','mid','high'])

            fna = f'strains/{asm}/{asm}.fna'
            fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
            names = list(fasta_sequences.keys())
            tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
            tss_dat.index = [re.match(r'ID=([^;]*);.*',x).group(1) for x in tss_dat['Desc']]
            tss_dat = tss_dat.loc[scaled_mat.index,:]
            tss_dat_np = tss_dat.to_numpy()
            gc_bias = []
            for i in tss_dat_np:
                genome_len = len(fasta_sequences[i[0]].seq)
                start = i[5]
                strand = i[3]
                if start <= 501 or start >= genome_len-500:
                    continue
                left_perc = 100*len(re.findall("G|C",str(fasta_sequences[i[0]].seq[start-500:start])))/500
                right_perc = 100*len(re.findall("G|C",str(fasta_sequences[i[0]].seq[start:start+500])))/500
                if strand == '+': 
                    gc_bias.append(right_perc-left_perc)
                else:
                    gc_bias.append(left_perc-right_perc)
            df['gc_bias'] = gc_bias
            all_df = pd.concat([all_df, df])
all_df.to_csv('all_sample_species_tpm_vs_bias.csv')
'''
all_df = pd.read_csv('all_sample_species_tpm_vs_bias.csv', index_col = 0)
asm_2_species_dict = {x:asm_2_species(x) for x in set(all_df['asm'].tolist())}

all_df['species'] = all_df['asm'].map(asm_2_species_dict)
all_df.loc[all_df['species'] == '[Eubacterium] eligens','species'] = 'Lachnospira eligens' 
grouped_df = all_df.groupby(['species','samp','tpm_group']).mean().reset_index()
g = sns.boxplot(x="tpm_group", y="bias", data=grouped_df, order = ['zero','low','mid','high'])
plt.ylim(-250,550)
add_p_val(0,3,450,10,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='high','bias'],grouped_df.loc[grouped_df['tpm_group']=='zero','bias']).pvalue)
add_p_val(2,3,400,10,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='mid','bias'],grouped_df.loc[grouped_df['tpm_group']=='high','bias']).pvalue)
add_p_val(1,2,350,10,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='low','bias'],grouped_df.loc[grouped_df['tpm_group']=='mid','bias']).pvalue)
add_p_val(0,1,300,10,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='zero','bias'],grouped_df.loc[grouped_df['tpm_group']=='low','bias']).pvalue) 
g.figure.savefig(f"tpm_vs_bias_plots/All_bias_vs_tpm_boxplot.pdf")
plt.clf()
kruskal(grouped_df.loc[grouped_df['tpm_group']=='high','bias'].tolist(),grouped_df.loc[grouped_df['tpm_group']=='mid','bias'].tolist(),
      grouped_df.loc[grouped_df['tpm_group']=='low','bias'].tolist(),grouped_df.loc[grouped_df['tpm_group']=='zero','bias'].tolist())
# KruskalResult(statistic=139.40525411915178, pvalue=5.078036479843691e-30)
#GC content
g = sns.boxplot(x="tpm_group", y="gc_bias", data=grouped_df, order = ['zero','low','mid','high'], showfliers=False)
plt.ylim(-0.5,5.5)
add_p_val(0,3,5,0.05,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='high','gc_bias'],grouped_df.loc[grouped_df['tpm_group']=='zero','gc_bias']).pvalue)
add_p_val(2,3,4.5,0.05,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='mid','gc_bias'],grouped_df.loc[grouped_df['tpm_group']=='high','gc_bias']).pvalue)
add_p_val(1,2,4,0.05,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='low','gc_bias'],grouped_df.loc[grouped_df['tpm_group']=='mid','gc_bias']).pvalue)
add_p_val(0,1,3.5,0.05,ttest_rel(grouped_df.loc[grouped_df['tpm_group']=='zero','gc_bias'],grouped_df.loc[grouped_df['tpm_group']=='low','gc_bias']).pvalue) 
g.figure.savefig(f"tpm_vs_bias_plots/All_gc_vs_tpm_boxplot.pdf")
plt.clf()

###Species specific
for spec in list(set(grouped_df.species)):
    df = grouped_df[grouped_df['species']==spec]
    g = sns.boxplot(x="tpm_group", y="bias", data=df, order = ['zero','low','mid','high'])
    plt.ylim(-200,540)
    add_p_val(0,3,450,10,ttest_rel(df.loc[df['tpm_group']=='high','bias'],df.loc[df['tpm_group']=='zero','bias']).pvalue)
    add_p_val(2,3,400,10,ttest_rel(df.loc[df['tpm_group']=='mid','bias'],df.loc[df['tpm_group']=='high','bias']).pvalue)
    add_p_val(1,2,350,10,ttest_rel(df.loc[df['tpm_group']=='low','bias'],df.loc[df['tpm_group']=='mid','bias']).pvalue)
    add_p_val(0,1,300,10,ttest_rel(df.loc[df['tpm_group']=='zero','bias'],df.loc[df['tpm_group']=='low','bias']).pvalue) 
    g.figure.savefig(f"tpm_vs_bias_plots/{spec}_bias_vs_tpm_boxplot.pdf")
    plt.clf()

    g = sns.boxplot(x="tpm_group", y="gc_bias", data=df, order = ['zero','low','mid','high'])
    plt.ylim(-1,5.5)
    add_p_val(0,3,4,0.05,ttest_rel(df.loc[df['tpm_group']=='high','gc_bias'],df.loc[df['tpm_group']=='zero','gc_bias']).pvalue)
    add_p_val(2,3,3.6,0.05,ttest_rel(df.loc[df['tpm_group']=='mid','gc_bias'],df.loc[df['tpm_group']=='high','gc_bias']).pvalue)
    add_p_val(1,2,3.3,0.05,ttest_rel(df.loc[df['tpm_group']=='low','gc_bias'],df.loc[df['tpm_group']=='mid','gc_bias']).pvalue)
    add_p_val(0,1,3,0.05,ttest_rel(df.loc[df['tpm_group']=='zero','gc_bias'],df.loc[df['tpm_group']=='low','gc_bias']).pvalue) 
    g.figure.savefig(f"tpm_vs_bias_plots/{spec}_gc_vs_tpm_boxplot.pdf")
    plt.clf()

###Line plots
#TPM
all_df = pd.DataFrame()
for samp in tqdm(samp_bn):
    all_samp = pd.DataFrame()
    for asm in asm_2_species_dict.keys():
        if not os.path.exists(f'matrices/{asm}_{samp}.csv'):
            continue
        tpm_file = get_tpm(f'kallisto_res/{asm}_{samp}/abundance.tsv')
        chrom_map = pd.read_table(f'strains/{asm}/{asm}_filtered_tss.csv', sep = ',', header=0, index_col=0)
        chrom_map = chrom_map[~chrom_map.index.duplicated(keep='first')]
        chrom_map = {chrom:re.match(r'ID=(\d*)_',chrom_map.loc[chrom,'Desc']).group(1) for chrom in chrom_map.index}
        tpm_file.index = [x.replace(x.rsplit('_',1)[0],chrom_map.get(x.rsplit('_',1)[0],'na')) for x in tpm_file.index]
        tpm_dict = tpm_file.to_dict()

        mat = pd.read_table(f'matrices/{asm}_{samp}.csv', sep = ',', header=None, index_col=0)
        mat = mat.loc[mat.sum(axis=1)!=0,:]
        tpm_val = [tpm_dict['tpm'].get(x,np.nan) for x in mat.index]
        nan = pd.Series(tpm_val).notnull()
        mat = mat.loc[nan.tolist(),:]
        tpm_val = pd.Series(tpm_val)[nan]
        tpm_val = tpm_val.reset_index(drop=True)
        tpm_color = pd.Series(np.where(tpm_val == 0, "zero", np.where(tpm_val < 100, 'low', np.where(tpm_val < 1000, 'mid', 'high'))))
        scaled_mat = scale_func2(mat)
        scaled_mat = scaled_mat.assign(group=tpm_color.tolist())
        pdFrame_melt = pd.melt(scaled_mat, id_vars = 'group',var_name='pos', value_name='read_cv')
        pdFrame_melt['species'] = asm_2_species_dict[asm]
        all_samp = pd.concat([all_samp,pdFrame_melt])
    all_samp.loc[all_samp['species'] == '[Eubacterium] eligens','species'] = 'Lachnospira eligens' 
    all_samp = all_samp.groupby(['species','group','pos']).mean().reset_index()
    all_samp['sample'] = samp
    all_df = pd.concat([all_df,all_samp])           
all_df.to_csv('tpm_vs_bias_lineplot.csv')

lines = sns.lineplot(data=all_df, x="pos", y="read_cv", hue='group', hue_order=['zero','low','mid','high'])
lines.set_xticks(range(len(set(pdFrame_melt.pos))))
lines.set_xticklabels(xlab)
lines.tick_params(bottom=False)
lines.axvline(x=1001//2, color = 'red')
lines.set(title="Species Average")
lines.get_figure().savefig(f'tpm_vs_bias_plots/all_samp_species_lineplot.pdf')
plt.clf()

#Species-specific
for spe in list(set(all_df.species)):
    df = all_df[all_df['species']==spe]
    lines = sns.lineplot(data=df, x="pos", y="read_cv", hue='group', hue_order=['zero','low','mid','high'])
    lines.set_xticks(range(len(set(pdFrame_melt.pos))))
    lines.set_xticklabels(xlab)
    lines.tick_params(bottom=False)
    lines.axvline(x=1001//2, color = 'red')
    lines.set(title=spe)
    lines.get_figure().savefig(f'tpm_vs_bias_plots/{spe}_all_samp_lineplot.pdf')
    plt.clf()