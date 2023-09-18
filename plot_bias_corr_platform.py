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

def scale_func2(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(left_sum-right_sum)

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points
    
def get_bias_from_mat_filtshort(asm,sample):
    if not os.path.exists(f'matrices/{asm}_{sample}.csv'):
        return(np.array([0]))
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    tss_dat.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat['Desc']])
    tss_dat['len'] = tss_dat['geneRight']-tss_dat['geneLeft']
    tss_dat = tss_dat[tss_dat['len'] > 1000]
    mat = pd.read_table(f'matrices/{asm}_{sample}.csv', sep = ',', header=None, index_col=0)
    mat = mat.loc[tss_dat.index,:]
    bias = np.array(list(map(calc_bias, np.array(scale_func2(mat)))))
    #bias[np.isnan(bias)] = 0
    return bias
    
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

def get_avg_r(a,b):
    r_vals = []
    for r in itertools.product(a, b):
        nan = np.isnan(r[0]) | np.isnan(r[1])
        if np.array_equal(r[0][~nan], r[1][~nan]):
            continue
        r_vals.append(np.corrcoef(r[0][~nan],r[1][~nan])[0][1])
    return np.mean(r_vals)

def taxid2ftp(taxid,n=1):
    strain = refseq[(refseq['taxid']==int(taxid)) & (refseq['assembly_level']=='Complete Genome')]
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
        subprocess.call(f'gzip -d {tax_path}/{asm}.fna.gz', shell= True)
        
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
        subprocess.call(f'bwa index {genome_fna}', shell=True)
    return(asm)

###Initialise Data
with open('matched_samples_8') as f:
    illumina_samples = f.read().splitlines()
illumina_bn = [x.rsplit('/',1)[1] for x in illumina_samples]

with open('pacbio_idlist') as f:
    pacbio_samples = f.read().splitlines()
pacbio_bn = [x.rsplit('/',1)[1] for x in pacbio_samples]

with open('nanopore_idlist') as f:
    nanopore_samples = f.read().splitlines()
nanopore_bn = [x.rsplit('/',1)[1] for x in nanopore_samples]

all_bn = illumina_bn+pacbio_bn+nanopore_bn
all_location = illumina_samples+pacbio_samples+nanopore_samples
basename2location = {all_bn[i]: all_location[i] for i in range(len(all_bn))}

#Read species abundances
illumina = pd.read_table('mpa_res/merged_mpa.txt', comment= '#')
illumina = illumina[illumina['clade_name'].str.contains("s__")]
illumina.index = [int(re.match('.*\|([^\|]*)',x).group(1)) for x in illumina['NCBI_tax_id']]
illumina = illumina[illumina_bn]
pacbio = pd.read_csv('/groups/cgsd/gordonq/TSS_depth/pacbio_hifi_dataset/merged_k2_raw.csv',header = 0, index_col = 0)
pacbio = 100*pacbio/pacbio.sum(axis=0)
pacbio.index = pacbio.index.values
nanopore = pd.read_csv('/groups/cgsd/gordonq/TSS_depth/intergenic_hypothesis/nanopore_dataset/merged_k2_raw.csv',header = 0, index_col = 0)
nanopore = 100*nanopore/nanopore.sum(axis=0)

merge = pd.concat([illumina,pacbio,nanopore], axis=1)
merge = merge.loc[((merge[illumina_bn] > 0.5).sum(axis=1) >= 4) & ((merge[pacbio_bn] > 0.5).sum(axis=1) >= 4) & ((merge[nanopore_bn] > 0.5).sum(axis=1) >= 4),:] # Filter species with atleast 0.5% abundance in atleast 4 samples in each dataset
merge[merge < 0.5] = np.nan
merge = merge.drop(39491) #Remove Eubacterium Rectale because no reference genome in refseq

refseq = pd.read_table('/groups/cgsd/gordonq/database/assembly_summary_refseq.txt', header = 1)

#Align sample reads to genomes
for taxid in merge.index:
    #Prepare species annotation folders
    ftps = taxid2ftp(taxid,3)
    print(taxid)
    for ftp in ftps:
        asm = prepare_species_folder(ftp)
        print(asm)
        subprocess.call(f'bwa index strains/{asm}/{asm}.fna', shell=True)
        #Align samples
        for sample in all_bn:
            if np.isnan(merge.loc[taxid,sample]):
                continue
            DNA_loc = basename2location[sample]
            ### Align reads ##
            if not os.path.exists(f'bams/{asm}_{sample}.sorted.bam'):
                if sample in pacbio_bn: #pacbio
                    subprocess.call(f'bwa mem -t 64 -x pacbio strains/{asm}/{asm}.fna {DNA_loc}.fastq |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
                elif sample in nanopore_bn: #Nanopore
                    subprocess.call(f'bwa mem -t 64 -x ont2d strains/{asm}/{asm}.fna {DNA_loc}.fastq |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_{sample}.sorted.bam --threads 64', shell=True)
                else: #Illumina
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

def get_cor_mat(asm):
    illumina_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in illumina_bn])
    pacbio_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in pacbio_bn])
    nanopore_samples_bias = np.array([get_bias_from_mat(asm,samp) for samp in nanopore_bn])
    #illumina_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in illumina_bn])
    #pacbio_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in pacbio_bn])
    #nanopore_samples_bias = np.array([get_bias_from_mat_filtshort(asm,samp) for samp in nanopore_bn])
    illumina_samples_bias = illumina_samples_bias[[False if len(x)==1 else True for x in illumina_samples_bias]]
    pacbio_samples_bias = pacbio_samples_bias[[False if len(x)==1 else True for x in pacbio_samples_bias]]
    nanopore_samples_bias = nanopore_samples_bias[[False if len(x)==1 else True for x in nanopore_samples_bias]]
    all_bias_groups = [illumina_samples_bias,pacbio_samples_bias,nanopore_samples_bias]
    heatmap_vals = []
    for x in all_bias_groups:
        for y in all_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)


#Generate heatmaps
axis_labels = ['illumina','pacbio','nanopore']
all_spe_mat = np.array([[0]*3]*3)
pacbio_vals = []
nanopore_vals = []
pac_vs_nano_vals = []
ill_vs_pac_vals = []
ill_vs_nano_vals = []
for taxid in merge.index:
    print(taxid)
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    if len(asms) == 0:
        continue
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        print(asm)
        cor_mat = cor_mat + get_cor_mat(asm)
    cor_mat = cor_mat/len(asms)
    pacbio_vals.append(cor_mat[1][1])
    nanopore_vals.append(cor_mat[2][2])
    pac_vs_nano_vals.append(cor_mat[1][2])
    ill_vs_pac_vals.append(cor_mat[0][1])
    ill_vs_nano_vals.append(cor_mat[0][2])
    all_spe_mat = all_spe_mat + cor_mat
    #g = sns.heatmap(cor_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0)
    #g.set_title(taxid)
    #g.figure.savefig(f"heatmaps/platform_comparison_{taxid}.png")
    #plt.clf()
all_spe_mat = all_spe_mat/len(merge.index)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels,vmin=0,vmax=0.8,cmap='Oranges')
g.set_title("Species average")
g.figure.savefig(f"heatmaps/platform_comparison_avg_species_newthresh.pdf")
plt.clf() 

np.mean(pacbio_vals) #0.6542966199689032
np.std(pacbio_vals) #0.05232718634859169
np.mean(nanopore_vals) #0.7790258368235096
np.std(nanopore_vals) #0.020070147919363512
np.mean(pac_vs_nano_vals) #0.5738139767588447
np.std(pac_vs_nano_vals) #0.037122332649173896
np.mean(ill_vs_pac_vals) #0.21593135952677756
np.std(ill_vs_pac_vals) #0.09433580306845772
np.mean(ill_vs_nano_vals) #0.21969912702411754
np.std(ill_vs_nano_vals) #0.11907457565451368

### Get Null ###
def get_null_mat(asm):
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    illumina_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in illumina_bn])
    pacbio_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in pacbio_bn])
    nanopore_samples_bias = np.array([get_null_bias_from_mat(asm,samp,tss_dat) for samp in nanopore_bn])
    illumina_samples_bias = illumina_samples_bias[[False if len(x)==1 else True for x in illumina_samples_bias]]
    pacbio_samples_bias = pacbio_samples_bias[[False if len(x)==1 else True for x in pacbio_samples_bias]]
    nanopore_samples_bias = nanopore_samples_bias[[False if len(x)==1 else True for x in nanopore_samples_bias]]
    null_bias_groups = [illumina_samples_bias,pacbio_samples_bias,nanopore_samples_bias]
    heatmap_vals = []
    for x in null_bias_groups:
        for y in null_bias_groups:
            heatmap_vals.append(get_avg_r(x,y))
    cor_mat = np.array(heatmap_vals).reshape(3,3)
    return(cor_mat)

all_spe_mat = np.array([[0]*3]*3)
null_ill_vs_pac_vals = []
null_ill_vs_nano_vals = []
for taxid in merge.index:
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    if len(asms) == 0:
        continue
    cor_mat = np.array([[0]*3]*3)
    for asm in asms:
        cor_mat = cor_mat + get_null_mat(asm)
    cor_mat = cor_mat/len(asms)
    null_ill_vs_pac_vals.append(cor_mat[0][1])
    null_ill_vs_nano_vals.append(cor_mat[0][2])
    all_spe_mat = all_spe_mat + cor_mat

all_spe_mat = all_spe_mat/len(merge.index)
all_spe_mat = abs(all_spe_mat)
g = sns.heatmap(all_spe_mat, xticklabels=axis_labels, yticklabels=axis_labels, vmin =0, vmax = 0.8, cmap = 'Oranges')
g.set_title("Null species average")
g.figure.savefig(f"heatmaps/platform_comparison_avg_null_newthresh.pdf")
plt.clf()

np.mean(null_ill_vs_pac_vals)
np.std(null_ill_vs_pac_vals)
np.mean(null_ill_vs_nano_vals)
np.std(null_ill_vs_nano_vals)

stats.ttest_ind(ill_vs_pac_vals, null_ill_vs_pac_vals) #Ttest_indResult(statistic=3.9948274852191927, pvalue=0.007161061087509554)
stats.ttest_ind(ill_vs_nano_vals, null_ill_vs_nano_vals) #Ttest_indResult(statistic=3.238483504159996, pvalue=0.01772060085178355)
