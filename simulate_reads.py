import sys, re
import subprocess
from glob import glob

all_strains = glob(f'matrices/*_X31*.csv')
all_strains = list(set([re.match(f'matrices/(.*)_X31.*\.csv$',x).group(1) for x in all_strains]))

#asm = sys.argv[1]
#asm = 'GCF_002734145.1_ASM273414v1'  #853 Faecalibacterium prausnitzii A2165

for asm in all_strains:
    fna_file = f'strains/{asm}/{asm}.fna'
    subprocess.call(f'ml ART && art_illumina -na -ss HS25 -i {fna_file} -p -l 150 -c 100000 -m 200 -s 10 -o strains/{asm}/pe_simu', shell= True)
    subprocess.call(f'bowtie2 -x {fna_file} -1 strains/{asm}/pe_simu1.fq -2 strains/{asm}/pe_simu2.fq |samtools view --threads 64 -b - |samtools sort - -o bams/{asm}_simu.sorted.bam --threads 64', shell=True)
    subprocess.call(f'samtools index bams/{asm}_simu.sorted.bam -@ 64',shell=True)
    #subprocess.call(f'bamCoverage --bam bams/{asm}_simu.sorted.bam -p 64 -o beds/{asm}_simu.bed -of bedgraph --binSize 1', shell = True)
    subprocess.call(f'bamCoverage --bam bams/{asm}_simu.sorted.bam -p 64 -o beds/{asm}_simu.bw -of bigwig', shell = True)