import subprocess
from glob import glob
import re, os
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import pyBigWig
from tqdm import tqdm

def scale_func2(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points
    
def get_bias_from_mat(asm,sample):
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        return(np.array([0]))
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    bias = np.array(list(map(calc_bias, np.array(scale_func2(mat)))))
    #bias[np.isnan(bias)] = 0
    return bias

def get_null_bias_from_mat(asm,samp,annot):
    if not os.path.exists(f'beds/{asm}_{samp}.bw'):
        return(np.array([0]))
    depth = pyBigWig.open(f'beds/{asm}_{samp}.bw')
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(annot.Genome)):
        if genome_dict[chrom] < 1000:
            continue
        random_tss = randomize(np.array(annot[annot.Genome==chrom]['Start']))
        strand_perc = sum((annot.Genome==chrom)&(annot.Strand=='+'))/sum(annot.Genome==chrom)
        for i in random_tss:
            if i <= 500:
                i = 501
            if i >= genome_dict[chrom]-500:
                i = genome_dict[chrom]-501
            window = depth.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_mat.append(window)
    null_mat = np.array(null_mat)
    bias = np.array(list(map(calc_bias, np.array(scale_func2(null_mat)))))
    return bias

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

def get_avg_r(a,b):
    r_vals = []
    for r in itertools.product(a, b):
        nan = np.isnan(r[0]) | np.isnan(r[1])
        if np.array_equal(r[0][~nan], r[1][~nan]):
            continue
        r_vals.append(np.corrcoef(r[0][~nan],r[1][~nan])[0][1])
    return np.mean(r_vals)

def taxid2ftp(taxid,n=1):
    strain = refseq[(refseq['species_taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
    if len(strain) == 0:
        strain = refseq[(refseq['species_taxid']==int(taxid))]
    ftp = strain['ftp_path'][:n].tolist()
    ftp = [re.sub(r'^ftp','https',x) for x in ftp]
    return(ftp)

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

###Initialise Data
with open('matched_samples_8') as f:
    matched_samples = f.read().splitlines()
matched_bn = [x.rsplit('/',1)[1] for x in matched_samples]

with open('arg_samples') as f:
    arg_samples = f.read().splitlines()
arg_bn = [x.rsplit('/',1)[1] for x in arg_samples]

with open('hyp_samples') as f:
    hyp_samples = f.read().splitlines()
hyp_bn = [x.rsplit('/',1)[1] for x in hyp_samples]

with open('crc_samples') as f:
    crc_samples = f.read().splitlines()
crc_bn = [x.rsplit('/',1)[1] for x in crc_samples]

all_bn = matched_bn+arg_bn+hyp_bn+crc_bn
all_location = matched_samples+arg_samples+hyp_samples+crc_samples
basename2location = {all_bn[i]: all_location[i] for i in range(len(all_bn))}

#Read species abundances
mpa_res = pd.read_table('mpa_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
#mpa_res = mpa_res.loc[((mpa_res[matched_bn] > 1).sum(axis=1) >= 5) & ((mpa_res[arg_bn] > 1).sum(axis=1) >= 5) & ((mpa_res[hyp_bn] > 1).sum(axis=1) >= 5) & ((mpa_res[crc_bn] > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 3 samples in each dataset
mpa_res = mpa_res.loc[((mpa_res[matched_bn] > 1).sum(axis=1) >= 5) & ((mpa_res[arg_bn] > 1).sum(axis=1) >= 5) & ((mpa_res[hyp_bn] > 1).sum(axis=1) >= 5),:]
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop('39491') #Remove Eubacterium Rectale because no reference genome in refseq

#refseq = pd.read_table("/groups/cgsd/gordonq/database/prokaryotes.txt", header=0)
refseq = pd.read_table('/groups/cgsd/gordonq/database/assembly_summary_refseq.txt', header = 1)

for taxid in mpa_res.index:
    #Prepare species annotation folders
    ftps = taxid2ftp(taxid,3)
    for ftp in ftps:
        asm = prepare_species_folder(ftp)
        for sample in all_bn:
            if np.isnan(mpa_res.loc[taxid,sample]):
                continue
            DNA_loc = basename2location[sample]   
            
            ### Align reads ##
            if not os.path.exists(f'bams/{asm}_{sample}.sorted.bam'):
                subprocess.call(f'bowtie2 -p 64 -x strains/{asm}/{asm}.fna -1 {DNA_loc}_1.fastq.gz -2 {DNA_loc}_2.fastq.gz |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
                subprocess.call(f'samtools index bams/{asm}_{sample}.sorted.bam -@ 64', shell = True)

            #Get read depth of genome
            if not os.path.exists(f'beds/{asm}_{sample}.bw'):
                subprocess.call(f'bamCoverage --bam bams/{asm}_{sample}.sorted.bam -p 64 -o beds/{asm}_{sample}.bw -of bigwig', shell = True)

            ### Generate matrix ###
            if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
                bp_depth = pyBigWig.open(f'beds/{asm}_{sample}.bw')
                genome_dict = bp_depth.chroms()
                genomes = list(genome_dict.keys())
                tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv', header = 0)
                tss_dat = tss_dat.to_numpy()
                locus_tag = np.array([re.match(r'^ID=([^;]*)',x).group(1) for x in tss_dat[:,4]])
                mat = []
                for row in tss_dat:
                    chrom = row[0]
                    if (row[5] <= 550) or (row[5] + 550 >= genome_dict[chrom]):
                        window = [0]*1001
                    else:
                        window = bp_depth.values(chrom,int(row[5])-501,int(row[5])+500)
                        if row[3]=="-": window.reverse()
                    mat.append(window)
                mat = np.array(mat)
                x = np.hstack((locus_tag[np.newaxis].T,mat))
                np.savetxt(f'matrices/{asm}_{sample}.csv',x,fmt='%s',delimiter=',')

#Remove CRC
def get_cor_mat(asm):
    matched_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in matched_bn])
    arg_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in arg_bn])
    hyp_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in hyp_bn])
    #crc_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in crc_bn])
    matched_samples_bias = matched_samples_bias[[False if len(x)==1 else True for x in matched_samples_bias]]
    arg_samples_bias = arg_samples_bias[[False if len(x)==1 else True for x in arg_samples_bias]]
    hyp_samples_bias = hyp_samples_bias[[False if len(x)==1 else True for x in hyp_samples_bias]]
    #crc_samples_bias = crc_samples_bias[[False if len(x)==1 else True for x in crc_samples_bias]]
    #all_bias_groups = [matched_samples_bias,arg_samples_bias,hyp_samples_bias,crc_samples_bias]
    all_bias_groups = [matched_samples_bias,arg_samples_bias,hyp_samples_bias]
    heatmap_vals = []
    for x in all_bias_groups:
        for y in all_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)


all_spe_mat = np.array([[0]*3]*3)
axis_labels = ['matched','ARG','HYP']
diag = []
triangle = []
for taxid in tqdm(mpa_res.index):
    print(taxid)
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_cor_mat(asm)
    cor_mat = cor_mat/len(asms)
    all_spe_mat = all_spe_mat + cor_mat
    diag.append(cor_mat.diagonal())
    triangle.append(cor_mat[np.triu(cor_mat, k =1)!=0])
    #g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0)
    #g.set_title(taxid)
    #g.figure.savefig(f"heatmaps/data_comparison_{taxid}.png")
    #plt.clf()
all_spe_mat = all_spe_mat/len(mpa_res)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.4, vmin=0, cmap="Oranges")
g.set_title("4 species average")
g.figure.savefig(f"heatmaps/dataset_comparison_no_crc.pdf")
plt.clf() 

#diag = np.hstack(diag)
diag = all_spe_mat.diagonal()
diag.mean() #0.3446823078855062
diag.std() #0.033765901446573154
#triangle = np.hstack(triangle)
triangle = all_spe_mat[np.triu(all_spe_mat, k =1)!=0]
triangle.mean() # 0.3000244748928326
triangle.std() # 0.024080439078531284
stats.ttest_ind(diag, triangle) #Ttest_indResult(statistic=1.5577495486399298, pvalue=0.19428968639433827)

### Get Null ###
def get_null_mat(asm):
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    matched_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in matched_bn])
    arg_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in arg_bn])
    hyp_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in hyp_bn])
    #crc_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in crc_bn])
    matched_samples_bias = matched_samples_bias[[False if len(x)==1 else True for x in matched_samples_bias]]
    arg_samples_bias = arg_samples_bias[[False if len(x)==1 else True for x in arg_samples_bias]]
    hyp_samples_bias = hyp_samples_bias[[False if len(x)==1 else True for x in hyp_samples_bias]]
    #crc_samples_bias = crc_samples_bias[[False if len(x)==1 else True for x in crc_samples_bias]]
    null_bias_groups = [matched_samples_bias,arg_samples_bias,hyp_samples_bias]
    heatmap_vals = []
    for x in null_bias_groups:
        for y in null_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)

all_spe_mat = np.array([[0]*3]*3)
axis_labels = ['matched','ARG','HYP']
for taxid in tqdm(mpa_res.index):
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_null_mat(asm)
    cor_mat = cor_mat/len(asms)
    all_spe_mat = all_spe_mat + cor_mat
    #g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels)
    #g.set_title(taxid)
    #g.figure.savefig(f"heatmaps/data_comparison_{taxid}.png")
    #plt.clf()
all_spe_mat = all_spe_mat/len(mpa_res)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmax = 0.4, vmin=0, cmap="Oranges")
g.set_title("Null species average")
g.figure.savefig(f"heatmaps/dataset_comparison_avg_null_no_crc.pdf")
plt.clf() 

all_spe_mat.mean() # 0.00028834790396005985
all_spe_mat.std() # 0.001413551330624313
 