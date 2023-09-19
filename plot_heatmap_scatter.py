from glob import glob
import re, os
import pandas as pd
import seaborn as sns
import pyBigWig
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio import SeqIO
from scipy.stats import pearsonr
from pysam import FastaFile
import subprocess

def scale_func2(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def plot_heatmap(mat,title,vmax,dpi=300):
    g = sns.clustermap(mat, col_cluster = False, row_cluster = False, xticklabels = False, yticklabels=False, figsize=(10, 25), vmax=vmax, cmap = 'Oranges')
    g.savefig(title, dpi = dpi)

def get_null(mat,depth):
    genome_dict = depth.chroms()
    random_pos = []
    for i in range(mat.shape[0]):
        chrom = np.random.choice(list(genome_dict.keys()))
        position = np.random.randint(500,genome_dict[chrom]-500)
        direction = np.random.choice([-1,1])
        random_pos.append([chrom,position,direction])
    null_mat = []
    for i in random_pos:
        if i[1] <= 500 or i[1] >= genome_dict[i[0]]-500:
            continue
        window = depth.values(i[0],i[1]-501,i[1]+500)
        if i[2] == -1: window.reverse()
        null_mat.append(window)
    return(pd.DataFrame(null_mat))

def get_fixed_null(depth,annot):
    genome_dict = depth.chroms()
    null_mat = []
    for chrom in list(set(annot.Genome)):
        random_tss = randomize(np.array(annot[annot.Genome==chrom]['Start']))
        strand_perc = sum((annot.Genome==chrom)&(annot.Strand=='+'))/sum(annot.Genome==chrom)
        for i in random_tss:
            if i <= 500 or i >= genome_dict[chrom]-500:
                continue
            window = depth.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_mat.append(window)
    return(pd.DataFrame(null_mat))

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

def sort_tss_bias(x):
    #input pandas matrix
    x['bias'] = [calc_bias(i) for i in np.array(x)]
    index = x.sort_values(['bias'], ascending = False).index
    return(index)

def randomize(intervals):
    intervals = intervals.astype(int)
    point_diffs = intervals[1:] - intervals[:-1]
    max_len = intervals[-1]
    new_points = np.concatenate((np.array(point_diffs).cumsum(),[max_len]))
    new_points = (new_points + np.random.randint(0,max_len))%max_len
    return new_points

def get_simu_bias_mat(asm):
    fna = f'strains/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    tss_dat_np = tss_dat.to_numpy()
    simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
    simu_mat = []
    for i in tss_dat_np:
        genome_len = len(fasta_sequences[i[0]].seq)
        start = i[5]
        strand = i[3]
        if start <= 501 or start >= genome_len-500:
            window = [0]*1001
        else:
            window = simu_bw.values(i[0],start-501,start+500)
        if i[3]=="-": window.reverse()
        simu_mat.append(window)
    simu_mat = pd.DataFrame(simu_mat)
    simu_mat.index = [re.match(r'ID=([^;]*);.*',x[4]).group(1) for x in tss_dat_np]
    scaled_mat = scale_func2(simu_mat)
    scaled_mat = scaled_mat.fillna(0)
    return(scaled_mat)

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

###Initialise Data
with open('matched_samples_8') as f:
    samples = f.read().splitlines()
samp_bn = [x.rsplit('/',1)[1] for x in samples]
basename2location = {samp_bn[i]: samples[i] for i in range(len(samples))}

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

###Prepare files alignment###
for taxid in mpa_res.index:
    ftps = taxid2ftp(taxid,3)
    for ftp in ftps:
        asm = prepare_species_folder(ftp)
        for sample in samp_bn:
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

###################
####### GC ########
###################
###GC content Null distribution
def get_gc_null(tss_dat,fasta_sequences):
    null_gc = []
    for chrom in list(set(tss_dat.Genome)):
        random_tss = randomize(np.array(tss_dat[tss_dat.Genome==chrom]['Start']))
        strand_perc = sum((tss_dat.Genome==chrom)&(tss_dat.Strand=='+'))/sum(tss_dat.Genome==chrom)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len-500:
                continue
            left_perc = 100*len(re.findall("G|C",str(fasta_sequences[chrom].seq[i-500:i])))/500
            right_perc = 100*len(re.findall("G|C",str(fasta_sequences[chrom].seq[i:i+500])))/500
            if np.random.binomial(1,strand_perc)==1: 
                null_gc.append(right_perc-left_perc)
            else:
                null_gc.append(left_perc-right_perc)
    return(np.mean(null_gc))

all_strains = []
for taxid in mpa_res.index:
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    print(taxid)
    print(asms)
    all_strains = all_strains + asms
dataest_asm_2_species = {x:asm_2_species(x) for x in all_strains}

all_pvals = []
for asm in all_strains:
    fna = f'strains/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
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
    real_bias = np.mean(gc_bias)
    null_bias = []
    for i in tqdm(range(500)):
        null_bias.append(get_gc_null(tss_dat,fasta_sequences))
    pval = (sum(null_bias > real_bias)+1)/len(null_bias)
    all_pvals.append([asm,pval])
    #plt.suptitle('GC null distribution')
    #plt.hist(null_bias, bins=50)
    #plt.axvline(x=real_bias, color='r')
    #plt.text(1, 10, f'p={pval}', size=12, color='black',weight='bold')
    #plt.savefig(f"null_bias_distributions/{asm}_gc.pdf")
    #plt.clf()
    
all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('null_bias_distributions/all_pvals_gc_newthresh.csv')

all_pvals = pd.read_csv('null_bias_distributions/all_pvals_gc_newthresh.csv', index_col=0)
g = sns.boxplot(data = all_pvals, y = '1', color = 'tab:blue')
plt.axvline(0.05, color='r')
g.figure.savefig(f"null_bias_distributions/All_samp_species_pval_boxplot_newthresh.pdf", bbox_inches="tight")
plt.clf()
'''
###GC content heatmap
#Real TSS GC
sorted_mat = tss_gc_mat.loc[sort_tss_bias(gc_mat_scaled),:]
vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
plot_heatmap(sorted_mat, f'heatmaps/{asm}_gc.png', vmax = vmax)
#Null GC
null_mat = get_gc_null(tss_dat)
scaled_null = scale_func2(null_mat)
scaled_null = scaled_null.fillna(0)
sorted_null = null_mat.loc[sort_tss_bias(scaled_null),:]
plot_heatmap(sorted_null, f'heatmaps/{asm}_gc_null.png',vmax=vmax)
'''

##########################
####### SIMULATED ########
##########################
###Simulated Reads null distribution
def get_simu_null(simu_bw,tss_dat,fasta_sequences):
    null_simu_mat = []
    for chrom in list(set(tss_dat.Genome)):
        random_tss = randomize(np.array(tss_dat[tss_dat.Genome==chrom]['Start']))
        strand_perc = sum((tss_dat.Genome==chrom)&(tss_dat.Strand=='+'))/sum(tss_dat.Genome==chrom)
        genome_len = len(fasta_sequences[chrom].seq)
        for i in random_tss:
            if i <= 501 or i >= genome_len-500:
                continue
            window = simu_bw.values(chrom,i-501,i+500)
            if np.random.binomial(1, strand_perc) == 0: window.reverse()
            null_simu_mat.append(window)
    return(pd.DataFrame(null_simu_mat))
    
for asm in all_strains:
    if os.path.exists(f'strains/{asm}/pe_simu1.fq'):
        continue
    fna_file = f'strains/{asm}/{asm}.fna'
    subprocess.call(f'ml ART && art_illumina -na -ss HS25 -i {fna_file} -p -l 150 -c 100000 -m 200 -s 10 -o strains/{asm}/pe_simu', shell= True)
    subprocess.call(f'bowtie2 -x {fna_file} -1 strains/{asm}/pe_simu1.fq -2 strains/{asm}/pe_simu2.fq |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_simu.sorted.bam --threads 64', shell=True)
    subprocess.call(f'samtools index bams/{asm}_simu.sorted.bam -@ 64',shell=True)
    subprocess.call(f'bamCoverage --bam bams/{asm}_simu.sorted.bam -p 64 -o beds/{asm}_simu.bw -of bigwig', shell = True)
    
all_pvals = []
for asm in all_strains:
    scaled_mat = get_simu_bias_mat(asm)
    simu_bias = list(map(calc_bias, np.array(scaled_mat)))
    simu_bw = pyBigWig.open(f'beds/{asm}_simu.bw')
    tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
    fna = f'strains/{asm}/{asm}.fna'
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    null_bias = []
    for i in tqdm(range(500)):
        null_mat = get_simu_null(simu_bw,tss_dat,fasta_sequences)
        scaled_null = scale_func2(null_mat)
        scaled_null = scaled_null.fillna(0)
        null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
    pval = (sum(null_bias > np.mean(simu_bias))+1)/len(null_bias)
    all_pvals.append([asm,pval])
    #plt.suptitle('Simulated reads null distribution')
    #plt.hist(null_bias, bins=50)
    #plt.axvline(x=np.mean(simu_bias), color='r')
    #plt.text(-7, 3, f'p={pval}', size=12, color='white',weight='bold')
    #plt.savefig(f"null_bias_distributions/{asm}_simu.pdf")
    #plt.clf()
    
all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('null_bias_distributions/all_pvals_simu_newthresh.csv')

all_pvals = pd.read_csv('null_bias_distributions/all_pvals_simu_newthresh.csv', index_col=0)
all_pvals['species'] = all_pvals['0'].map(dataest_asm_2_species)
all_pvals.loc[all_pvals['species'] == '[Eubacterium] eligens','species'] = 'Lachnospira eligens' 
all_pvals = all_pvals.groupby(['species']).mean().reset_index()
g = sns.boxplot(data = all_pvals, x = '1', color = 'tab:blue', width = 0.1)
g.figure.savefig(f"null_bias_distributions/All_species_simu_pval_boxplot_newthresh.pdf", bbox_inches="tight")
plt.clf()

'''
#Heatmaps
scaled_mat = scale_func2(simu_mat)
sorted_mat = simu_mat.loc[sort_tss_bias(scaled_mat),:]
vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
plot_heatmap(sorted_mat, f'heatmaps/{asm}_simu.png',vmax=vmax)

simu_null_mat = get_simu_null()
scaled_mat = scale_func2(simu_null_mat)
scaled_mat = scaled_mat.fillna(0)
sorted_mat = simu_null_mat.loc[sort_tss_bias(scaled_mat),:]
plot_heatmap(sorted_mat, f'heatmaps/{asm}_simu_null.png',vmax=vmax)
'''
###################################
####### REAL READ COVERAGE ########
###################################
###Fixed intervals null permutations
all_pvals = []
for asm in all_strains:
    for samp in samp_bn:
        if not os.path.exists(f'matrices/{asm}_{samp}.csv'):
            continue
        tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        tss_dat_np = tss_dat.to_numpy()
        mat = pd.read_table(f'matrices/{asm}_{samp}.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_func2(mat)
        scaled_mat = scaled_mat.fillna(0)
        real_bias = list(map(calc_bias, np.array(scaled_mat)))

        null_bias = []
        bw_depth = pyBigWig.open(f'beds/{asm}_{samp}.bw')
        for i in tqdm(range(500)):
            null_mat = get_fixed_null(bw_depth,tss_dat)
            scaled_null = scale_func2(null_mat)
            scaled_null = scaled_null.fillna(0)
            null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
        pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)
        #plt.suptitle(f'{samp} {asm}')
        #plt.hist(null_bias, bins=50)
        #plt.axvline(x=real_bias, color='r')
        #plt.text(-5, 3, f'p={pval}', size=12, color='white',weight='bold')
        #plt.savefig(f"null_bias_distributions/{asm}_{samp}_fixed.pdf")
        #plt.clf()

        simu_scaled_mat = get_simu_bias_mat(asm)
        simu_bias = list(map(calc_bias, np.array(simu_scaled_mat)))
        simu_corr = pearsonr(simu_bias,real_bias)
        all_pvals.append([asm,samp,pval,simu_corr])
        
all_pvals = pd.DataFrame(all_pvals)
all_pvals.to_csv('null_bias_distributions/all_pvals_newthresh.csv')

all_pvals = pd.read_csv('null_bias_distributions/all_pvals_newthresh.csv', index_col=0, header = 0, names=['asm','sample','pval','pearson'])
all_pvals['corr'] = [float(re.match(r'\(([^,]*),.*',x).group(1)) for x in all_pvals['pearson']]
all_pvals['species'] = all_pvals['asm'].map(dataest_asm_2_species)
all_pvals.loc[all_pvals['species'] == '[Eubacterium] eligens','species'] = 'Lachnospira eligens' 
all_pvals = all_pvals.groupby(['sample','species']).mean().reset_index()

g = sns.boxplot(data = all_pvals, y = 'species', x = 'pval', orient = 'h', color = 'tab:blue', showfliers=False)
plt.axvline(0.05, color='r')
plt.xscale("log")
g.figure.savefig(f"null_bias_distributions/All_samp_species_pval_boxplot_newthresh.pdf", bbox_inches="tight")
plt.clf()

g = sns.boxplot(data = all_pvals, y = 'species', x = 'corr', orient = 'h', color = 'tab:blue')
g.figure.savefig(f"null_bias_distributions/All_samp_species_simu_corr_boxplot_newthresh.pdf", bbox_inches="tight")
plt.clf()

###Plot heatmaps
for samp in tqdm(samples):
    strains = glob(f'matrices/*_{samp}.csv')
    strains = [re.match(f'matrices/(.*)_{samp}\.csv$',x).group(1) for x in strains]
    for asm in tqdm(strains):
        #Real TSS
        mat = pd.read_table(f'matrices/{asm}_{samp}.csv', sep = ',', header=None, index_col=0)
        scaled_mat = scale_func2(mat)
        scaled_mat = scaled_mat.fillna(0)
        sorted_mat = mat.loc[sort_tss_bias(scaled_mat),:]
        vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
        plot_heatmap(sorted_mat, f'heatmaps/{asm}_{samp}.png', vmax = vmax, dpi = 300)

        #bias = list(map(calc_bias, np.array(scaled_mat)))
        #perc = sum(i >0 for i in bias)/len(bias)
        #all_perc.append(perc)
        #Null
        bw_depth = pyBigWig.open(f'beds/{asm}_{samp}.bw')
        tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        null_mat = get_fixed_null(bw_depth,tss_dat)
        scaled_null = scale_func2(null_mat)
        scaled_null = scaled_null.fillna(0)
        sorted_null = null_mat.loc[sort_tss_bias(scaled_null),:]
        #bias = list(map(calc_bias, np.array(scaled_null)))
        #perc = sum(i >0 for i in bias)/len(bias)
        #null_perc.append(perc)
        plot_heatmap(sorted_null, f'heatmaps/{asm}_{samp}_null_fixed.png',vmax=vmax, dpi = 300)



#########################################
####### Validation in Long reads ########
#########################################

###Long read validation #Read alignment to species was performed in plot_bias_corr_platform.py
with open('pacbio_idlist') as f:
    samples = f.read().splitlines()
pacbio_samples = [x.rsplit('/',1)[1] for x in samples]

with open('nanopore_idlist') as f:
    samples = f.read().splitlines()
nanopore_samples = [x.rsplit('/',1)[1] for x in samples]

all_LR_pvals = []
for samp in (pacbio_samples+nanopore_samples):
    strains = glob(f'matrices/*_{samp}.csv')
    strains = [re.match(f'matrices/(.*)_{samp}\.csv$',x).group(1) for x in strains]
    for asm in strains:
        #Get upstream intergenic dist
        #gene_annot = pd.read_table(f'strains/{asm}/{asm}_prodigal.gff', header=None, comment='#')
        #gene_annot.drop(columns=[1,2,5,7], inplace = True)
        #gene_annot['TSS'] = [x[3] if x[6] == '+' else x[4] for i,x in gene_annot.iterrows()]
        #gene_annot.columns = ["Genome","geneLeft","geneRight","Strand","Desc","Start"]
        #reference_genome = FastaFile(f'strains/{asm}/{asm}.fna')
        #gene_annot['upstream_dist'] = get_upstream_dist(gene_annot,reference_genome)
        #gene_annot['operon_state'] = ['short' if x <=50 else 'long' for x in gene_annot['upstream_dist']]
        #gene_annot.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in gene_annot['Desc']])
        tss_dat = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv')
        tss_dat.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in tss_dat['Desc']])
        #tss_dat['operon_state'] = gene_annot.loc[tss_dat.index,'operon_state']
        #tss_dat = tss_dat[tss_dat.operon_state == 'long']
        tss_dat['len'] = tss_dat['geneRight']-tss_dat['geneLeft']
        tss_dat = tss_dat[tss_dat['len'] > 1000] #Filter short genes as long reads will cover past it
    
        mat = pd.read_table(f'matrices/{asm}_{samp}.csv', sep = ',', header=None, index_col=0)
        mat = mat.loc[tss_dat.index,]
        scaled_mat = scale_func2(mat)
        scaled_mat = scaled_mat.fillna(0)
        
        #Plot heatmap
        #sorted_mat = mat.loc[sort_tss_bias(scaled_mat),:]
        #sorted_mat = sorted_mat.loc[sorted_mat.sum(axis=1)!=0,:]
        #vmax = np.percentile(sorted_mat.to_numpy().flatten(),95)
        #plot_heatmap(sorted_mat, f'heatmaps/{asm}_{samp}.png', vmax = vmax)
        
        real_bias = list(map(calc_bias, np.array(scaled_mat)))
        null_bias = []
        bw_depth = pyBigWig.open(f'beds/{asm}_{samp}.bw')
        for i in tqdm(range(500)):
            null_mat = get_fixed_null(bw_depth,tss_dat)
            scaled_null = scale_func2(null_mat)
            scaled_null = scaled_null.fillna(0)
            null_bias.append(np.mean(list(map(calc_bias, np.array(scaled_null)))))
        pval = (sum(null_bias > np.mean(real_bias))+1)/len(null_bias)
        all_LR_pvals.append([asm,samp,pval])
        #plt.hist(null_bias, bins=50)
        #plt.axvline(x=np.mean(real_bias), color='r')
        #plt.text(-7, 3, f'p={pval}', size=12, color='white',weight='bold')
all_LR_pvals = pd.DataFrame(all_LR_pvals)
all_LR_pvals.to_csv('null_bias_distributions/all_longread_pvals_filtshort.csv')


#all_LR_pvals_long = pd.read_csv('null_bias_distributions/all_longread_pvals_long.csv', index_col = 0, header= 0, names = ['Strain','Sample','pval'])
all_LR_pvals = pd.read_csv('null_bias_distributions/all_longread_pvals.csv', index_col = 0, header= 0, names = ['Strain','Sample','pval'])
all_LR_pvals_filtshort = pd.read_csv('null_bias_distributions/all_longread_pvals_filtshort.csv', index_col = 0, header= 0, names = ['Strain','Sample','pval'])


#all_LR_pvals_long['upstream_dist'] = 'long'
all_LR_pvals['upstream_dist'] = 'all'
all_LR_pvals_filtshort['upstream_dist'] = 'filt_short'
merge = pd.concat([all_LR_pvals,all_LR_pvals_filtshort], axis=0)

all_strains = list(set(merge['Strain']))
platform_asm_2_species = {x:asm_2_species(x) for x in all_strains}
merge['species'] = merge['Strain'].map(platform_asm_2_species)
merge = merge.groupby(['species','Sample','upstream_dist']).mean().reset_index()
sum(merge[merge.upstream_dist=='all']['pval'] < 0.05)
sum(merge[merge.upstream_dist=='filt_short']['pval'] < 0.05)

g = sns.boxplot(data = merge, y = 'species', x = 'pval', orient = 'h', hue = 'upstream_dist', showfliers=False)
plt.xscale("log")
plt.axvline(0.05, color='r')
g.figure.savefig(f"null_bias_distributions/longread_pval_boxplot.pdf", bbox_inches="tight")
plt.clf()

for asm in set(all_LR_pvals_filtshort.Strain):
    print(f'{asm}: {np.mean(all_LR_pvals_filtshort[all_LR_pvals_filtshort.Strain==asm]["pval"])}')
