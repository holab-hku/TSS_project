import pysam
from pysam import FastaFile
import pyBigWig
import os, sys, re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import pingouin as pg
from align.calign import aligner
from align.matrix import DNAFULL
from Bio.Seq import Seq
import subprocess

SMALL_BUFFER = 5

def calc_upstream_diversity(gene,aln_file,read_index,window=50, read_len=151):
    align_score = []
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-300,int(gene[5])+200)
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped:
                if not read.mate_is_unmapped:
                    read_seq = aln_file.mate(read).seq
                else:
                    for pair in read_index.find(read.query_name):
                        if pair.is_unmapped:
                            read_seq = pair.seq #Because mapped read is in reverse direction, unmapped read should be forward, thus get original direction sequence
                align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-200,int(gene[5])+300)
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if not read.is_reverse and not read.is_unmapped:
                if not read.mate_is_unmapped:
                    read_seq = aln_file.mate(read).seq
                else:
                    for pair in read_index.find(read.query_name):
                        if pair.is_unmapped:
                            read_seq = str(Seq(pair.seq).reverse_complement()) #Because mapped read is in forward direction, the unmapped read is reverse and needs to be flipped
                align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
    return np.median(align_score)

def calc_positive_diversity(gene,aln_file,read_index, frag_len=200, read_len=151):
    if gene[3] == '+':
        gene[5] = int(gene[5]) + frag_len
    else:
        gene[5] = int(gene[5]) - frag_len
    return calc_upstream_diversity(gene,aln_file,read_index, read_len=read_len)

def count_unmapped(gene,aln_file, window=50, read_len = 151):
    count = 0
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped and read.mate_is_unmapped:
                count+=1
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if not read.is_reverse and not read.is_unmapped and read.mate_is_unmapped:
                count+=1
    return count

def add_p_val(lft,rgt,y,h,p):
    plt.plot([lft, lft, rgt, rgt], [y, y+h, y+h, y], lw=1.5, c='k')
    plt.text((lft + rgt) * .5, y+h, ('n.s.' if p > 0.15 else 'p < %.2g' if p > 0.001 else 'p < %.1g') % max(p+1e-20, 1e-20), ha='center', va='bottom', color='k')

def calc_negative_diversity(gene,aln_file,read_set,window=50, read_len=151):
    align_score = []
    num_reads = 0
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-300,int(gene[5])+200)
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped:
                num_reads += 1
        while num_reads > 1:
            rand_ind = np.random.randint(1,len(read_set))
            read_seq = read_set[rand_ind]
            align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
            num_reads = num_reads - 1
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        genome_seq = reference_genome.fetch(gene[0],int(gene[5])-200,int(gene[5])+300)
        for read in aln_file.fetch(gene[0],start,stop):
            if not read.is_reverse and not read.is_unmapped:
                num_reads += 1
        while num_reads > 1:
            rand_ind = np.random.randint(1,len(read_set))
            read_seq = read_set[rand_ind]
            align_score.append(aligner(read_seq, genome_seq, method='glocal', max_hits=1, matrix=DNAFULL)[0][-1])
            num_reads = num_reads - 1
    return np.median(align_score)

def calc_negative_read_count(gene,aln_file,window=50, read_len=151):
    align_score = []
    num_reads = 0
    if gene[3] == '+':
        start = int(gene[5]) - SMALL_BUFFER
        stop = int(gene[5]) + window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if read.is_reverse and not read.is_unmapped:
                num_reads += 1
    else:
        stop = int(gene[5]) - read_len + SMALL_BUFFER
        start = int(gene[5]) - read_len - window
        for read in aln_file.fetch(gene[0],start,stop):
            if read.reference_start < start or read.reference_start > stop:
                continue
            if not read.is_reverse and not read.is_unmapped:
                num_reads += 1
    return num_reads

def scale_func2(d):
    return (d-d.mean(axis=1)[:,None]) / d.std(axis=1)[:,None]

def calc_bias(x):
    left_sum = np.sum(x[:500])
    right_sum = np.sum(x[500:])
    return(right_sum-left_sum)

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

refseq = pd.read_table('/groups/cgsd/gordonq/database/assembly_summary_refseq.txt', header = 1)
with open('arg_samples') as f:
    samples = f.read().splitlines()
samp_bn = [x.rsplit('/',1)[1] for x in samples]
basename2location = {samp_bn[i]: samples[i] for i in range(len(samples))}

mpa_res = pd.read_table('mpa_res/merged_mpa.txt', comment= '#')
mpa_res = mpa_res[mpa_res['clade_name'].str.contains("s__")]
mpa_res.index = [re.match('.*\|([^\|]*)',x).group(1) for x in mpa_res['NCBI_tax_id']]
mpa_res = mpa_res.drop(columns=['clade_name', 'NCBI_tax_id'])
mpa_res = mpa_res.loc[:,samp_bn]
mpa_res = mpa_res.loc[((mpa_res > 1).sum(axis=1) >= 5),:] # Filter species with atleast 1% abundance in atleast 50% samples
mpa_res[mpa_res < 1] = np.nan
mpa_res = mpa_res.drop('39491') #Eubacterium Rectale no reference in database.

###################
###Prepare Files###
###################
'''
for taxid in mpa_res.index:
    #Prepare species annotation folders
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

#############################################################################################################

all_dat = pd.DataFrame()
for taxid in mpa_res.index:
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    for asm in asms:
        # Load gene regions
        gene_annot = pd.read_table(f'strains/{asm}/{asm}_prodigal.gff', header=None, comment='#')
        gene_annot.drop(columns=[1,2,5,7], inplace = True)
        gene_annot['TSS'] = [x[3] if x[6] == '+' else x[4] for i,x in gene_annot.iterrows()]
        gene_annot.columns = ["Genome","geneLeft","geneRight","Strand","Desc","Start"]
        reference_genome = FastaFile(f'strains/{asm}/{asm}.fna')
        gene_annot['upstream_dist'] = get_upstream_dist(gene_annot,reference_genome)
        gene_annot['operon_state'] = ['short' if x <=50 else 'long' for x in gene_annot['upstream_dist']]
        gene_annot.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in gene_annot['Desc']])
        #Filter short genes
        gene_annot['len'] = gene_annot['geneRight']-gene_annot['geneLeft']
        gene_annot = gene_annot[gene_annot['len'] > 500]
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome

        for sample in samp_bn:
            # Load alignments
            if not os.path.exists(f'bams/{asm}_{sample}.sorted.bam'):
                continue
            aln_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
            name_indexed = pysam.IndexedReads(aln_file)
            name_indexed.build()
            read_set = np.array([rec.seq for rec in aln_file])
            #Get read length
            for read in aln_file.fetch():
                read_len = len(read.get_forward_sequence())
                break
            #Calculate fragment length distribution
            fragment_len_dist =[]
            for read in aln_file.fetch():
                if read.mate_is_unmapped or read.is_unmapped or not read.is_proper_pair:
                    continue
                if not read.is_reverse:
                    fragment_len_dist.append(read.template_length)
            frag_len = np.median(np.abs(fragment_len_dist))

            #Calc upstream read diversity metrics
            gene_annot['upstream_unmapped_count'] = [count_unmapped(x,aln_file,window=50, read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['upstream_diversity'] = [calc_upstream_diversity(x,aln_file,name_indexed,read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['positive_diversity'] = [calc_positive_diversity(x,aln_file,name_indexed, frag_len, read_len) for x in np.array(gene_annot)]
            gene_annot['negative_diversity'] = [calc_negative_diversity(x,aln_file, read_set, read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['negative_read_count'] = [calc_negative_read_count(x,aln_file, read_len=read_len) for x in np.array(gene_annot)]

            sample_summary = gene_annot.groupby('operon_state')['upstream_unmapped_count','upstream_diversity','positive_diversity','negative_diversity','negative_read_count'].mean()
            sample_summary = sample_summary.melt(ignore_index=False)
            sample_summary['sample'] = sample
            sample_summary['asm'] = asm
            all_dat = all_dat.append(sample_summary)
all_dat.to_csv("all_samp_upstream_read_diversity.csv")
'''
all_strains = []
for taxid in mpa_res.index:
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    print(taxid)
    print(asms)
    all_strains = all_strains + asms
asm_2_species_dict = {x:asm_2_species(x) for x in all_strains}

'''
all_dat = pd.read_csv("all_samp_upstream_read_diversity.csv")
all_dat['species'] = all_dat.asm.map(asm_2_species_dict)
subset = all_dat[all_dat['variable'].isin(['negative_diversity','upstream_diversity','positive_diversity'])]
parsed_dat = subset.groupby(['sample','species','variable']).mean().reset_index()
order = ['negative_diversity','upstream_diversity','positive_diversity']
g = sns.boxplot(x=parsed_dat["variable"],y=parsed_dat["value"], order=order, color = 'tab:blue')
add_p_val(0,1,800,10,stats.ttest_rel(parsed_dat.loc[parsed_dat.variable=='negative_diversity','value'],parsed_dat.loc[parsed_dat.variable=='upstream_diversity','value']).pvalue)
#Ttest_relResult(statistic=-137.42636364251544, pvalue=1.173190878870822e-75)
add_p_val(1,2,920,10,stats.ttest_rel(parsed_dat.loc[parsed_dat.variable=='upstream_diversity','value'],parsed_dat.loc[parsed_dat.variable=='positive_diversity','value']).pvalue)
#Ttest_relResult(statistic=-11.416979800729763, pvalue=1.4241875653132817e-16)
g.figure.savefig(f"hetero_plots/all_samp_species_paired.pdf")
plt.clf()

order = ['upstream_diversity','positive_diversity']
g = sns.boxplot(x=parsed_dat["variable"],y=parsed_dat["value"], order=order, showfliers=False, color='tab:blue')
add_p_val(0,1,780,5,stats.ttest_rel(parsed_dat.loc[parsed_dat.variable=='upstream_diversity','value'],parsed_dat.loc[parsed_dat.variable=='positive_diversity','value']).pvalue)
g.figure.savefig(f"hetero_plots/all_samp_species_paired_zoomin.pdf")
plt.clf()

### Operon state ###
subset =  all_dat[all_dat['variable'] == 'negative_diversity']
subset = subset.groupby(['operon_state','species']).mean().reset_index()
g = sns.boxplot(x=subset["operon_state"],y=subset["value"], showfliers=False)
g.set_title('Negative diversity')
add_p_val(0,1,77,0.2,stats.ttest_ind(subset.loc[subset.operon_state=='long','value'],subset.loc[subset.operon_state=='short','value']).pvalue)
g.figure.savefig(f"hetero_plots/all_samp_species_negative_diversity.pdf")
plt.clf()

subset =  all_dat[all_dat['variable'] == 'positive_diversity']
subset = subset.groupby(['operon_state','asm']).mean().reset_index()
g = sns.boxplot(x=subset["operon_state"],y=subset["value"])
g.set_title('Positive diversity')
add_p_val(0,1,750,5,stats.ttest_ind(subset.loc[subset.operon_state=='long','value'],subset.loc[subset.operon_state=='short','value']).pvalue)
g.figure.savefig(f"hetero_plots/all_samp_species_positive_diversity.pdf")
plt.clf()

#Long vs short
subset =  all_dat[all_dat['variable'] == 'upstream_diversity']
subset = subset.groupby(['operon_state','sample','species']).mean().reset_index()
g = sns.boxplot(x=subset["operon_state"],y=subset["value"], showfliers=False)
g.set_title('Upstream diversity')
add_p_val(0,1,760,5,stats.ttest_ind(subset.loc[subset.operon_state=='long','value'],subset.loc[subset.operon_state=='short','value']).pvalue)
g.figure.savefig(f"hetero_plots/all_samp_species_upstream_diversity.pdf")
plt.clf()

subset =  all_dat[all_dat['variable'] == 'upstream_unmapped_count']
g = sns.boxplot(x=subset["operon_state"],y=subset["value"])
g.set_title('Upstream unmapped read count')
add_p_val(0,1,11,0.5,stats.ttest_ind(subset.loc[subset.operon_state=='outside','value'],subset.loc[subset.operon_state=='within','value']).pvalue)
g.figure.savefig(f"hetero_plots/{asm}_10samp_upstream_unmapped_count.pdf")
plt.clf()

subset =  all_dat[all_dat['variable'] == 'negative_read_count']
g = sns.boxplot(x=subset["operon_state"],y=subset["value"])
g.set_title('Negative control read count')
add_p_val(0,1,36,1,stats.ttest_ind(subset.loc[subset.operon_state=='outside','value'],subset.loc[subset.operon_state=='within','value']).pvalue)
g.figure.savefig(f"hetero_plots/{asm}_10samp_negative_control_read_count.pdf")
plt.clf()
'''

'''
### Bias vs operon state ###
all_dat = pd.DataFrame()
for sample in tqdm(samples):
    mat = pd.read_csv(f'matrices/{asm}_{sample}.csv', header = None, index_col = 0)
    scaled_mat = scale_func2(mat)
    gene_annot['bias'] = list(map(calc_bias, np.array(scaled_mat.loc[gene_annot.index,:])))
    gene_annot = gene_annot.dropna()
    avg = gene_annot.groupby('operon_state')['bias'].mean()
    all_dat = all_dat.append(avg)
melt_dat = all_dat.melt()
g = sns.boxplot(data = melt_dat, x = 'variable', y='value')
add_p_val(0,1,190,5,stats.ttest_ind(all_dat['outside'],all_dat['within']).pvalue)
g.figure.savefig(f"box_plots/{asm}_10samp_bias_vs_operonstate.pdf")
plt.clf()
'''

### Heterogenity vs bias ###
all_dat = pd.DataFrame()
for taxid in mpa_res.index:
    ftps = taxid2ftp(taxid,3)
    asms = [x.rsplit('/',1)[1] for x in ftps]
    for asm in asms:
        # Load gene regions
        gene_annot = pd.read_table(f'strains/{asm}/{asm}_prodigal.gff', header=None, comment='#')
        gene_annot.drop(columns=[1,2,5,7], inplace = True)
        gene_annot['TSS'] = [x[3] if x[6] == '+' else x[4] for i,x in gene_annot.iterrows()]
        gene_annot.columns = ["Genome","geneLeft","geneRight","Strand","Desc","Start"]
        reference_genome = FastaFile(f'strains/{asm}/{asm}.fna')
        gene_annot['upstream_dist'] = get_upstream_dist(gene_annot,reference_genome)
        gene_annot['operon_state'] = ['short' if x <=50 else 'long' for x in gene_annot['upstream_dist']]
        gene_annot.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in gene_annot['Desc']])
        #Filter short genes
        gene_annot['len'] = gene_annot['geneRight']-gene_annot['geneLeft']
        gene_annot = gene_annot[gene_annot['len'] > 500]
        gene_annot = gene_annot[(gene_annot.Strand == '-') & (gene_annot.Start > 750)] #Incase of fetching beyond start of genome
        filt_genes = pd.read_csv(f'strains/{asm}/{asm}_filtered_tss.csv', header=0)
        filt_genes.index = np.array([re.match(r'ID=([^;]*)',x).group(1) for x in filt_genes['Desc']])
        gene_annot = gene_annot.loc[gene_annot.index.isin(filt_genes.index),:]
        
        for sample in samp_bn:
            if not os.path.exists(f'bams/{asm}_{sample}.sorted.bam'):
                continue
            mat = pd.read_csv(f'matrices/{asm}_{sample}.csv', header = None, index_col = 0)
            scaled_mat = scale_func2(mat)
            # Load alignments
            aln_file = pysam.AlignmentFile(f'bams/{asm}_{sample}.sorted.bam', "rb")
            name_indexed = pysam.IndexedReads(aln_file)
            name_indexed.build()
            #Get read length
            for read in aln_file.fetch():
                read_len = len(read.get_forward_sequence())
                break
            gene_annot['upstream_diversity'] = [calc_upstream_diversity(x,aln_file,name_indexed,read_len=read_len) for x in np.array(gene_annot)]
            gene_annot['bias'] = list(map(calc_bias, np.array(scaled_mat.loc[gene_annot.index,:])))
            gene_annot = gene_annot.dropna()
            gene_annot['sample'] = sample
            gene_annot['asm'] = asm
            all_dat = all_dat.append(gene_annot[['sample','asm','upstream_diversity','bias',]])
all_dat.to_csv("all_samp_bias_vs_align_score.csv")
'''
all_dat = pd.read_csv("all_samp_bias_vs_align_score.csv", index_col=0)
corr_res = []
for samp in set(all_dat['sample']):
    for asm in set(all_dat['asm']):
        dat = all_dat[(all_dat['sample']==samp)&(all_dat['asm']==asm)]
        if len(dat) == 0:
            continue
        res = stats.pearsonr(dat['bias'], dat['upstream_diversity'])
        corr_res.append([samp,asm,res[0],res[1]])
corr_res = pd.DataFrame(corr_res, columns = ['sample','asm','corr','pvalue'])
corr_res['species'] = corr_res['asm'].map(asm_2_species_dict)
corr_res = corr_res.groupby(['sample','species']).mean().reset_index()

corr_pivot = corr_res.pivot("sample", "species", "corr")
corr_pivot = corr_pivot.transpose()
g = sns.heatmap(corr_pivot, vmin=-0.3, vmax=0.3, cmap="coolwarm")
g.figure.savefig(f"hetero_plots/hetero_vs_bias_corr_heatmap.pdf", bbox_inches="tight")
plt.clf()

g = sns.boxplot(y=corr_res["species"],x=corr_res["corr"], color = 'tab:blue')
g.figure.savefig(f"hetero_plots/hetero_vs_bias_corr.pdf", bbox_inches="tight")
plt.clf()

g = sns.boxplot(y=corr_res["species"],x=corr_res["pvalue"], color = 'tab:blue', showfliers=False)
plt.xscale('log')
plt.axvline(0.05, color='r')
g.figure.savefig(f"hetero_plots/hetero_vs_bias_pvalue.pdf", bbox_inches="tight")
plt.clf()

g = sns.regplot(data = all_dat, x = 'bias', y = 'upstream_diversity')
stats.pearsonr(all_dat['bias'], all_dat['upstream_diversity'])
#(-0.1531051871194317, 0.0)

fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(
    data=all_dat,
    x="bias",
    y="upstream_diversity",
    color="k",
    ax=ax,
)
sns.kdeplot(
    data=all_dat,
    x="bias",
    y="upstream_diversity",
    levels=5,
    fill=True,
    alpha=0.6,
    cut=2,
    ax=ax,
)
'''