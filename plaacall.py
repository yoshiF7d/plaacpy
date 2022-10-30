import plaacmod
import re
import os
import argparse
from Colors import Colors
from plaacmod import Plaac

order = ['YBR289W', 'YIL130W', 'YMR164C', 'YDL035C', 'YLR206W', 'YPR042C', 'YBR112C', 'YDL167C', 'YMR043W', 'YBR016W', 'YEL007W', 'YKL054C', 'YNL161W', 'YGL122C', 'YDR172W', 'YKL032C', 'YOL051W', 'YCL028W', 'YIR006C', 'YPL016W', 'YMR263W', 'YLR278C', 'YPL089C', 'YMR047C', 'YPL226W', 'YNL016W', 'YMR070W', 'YER112W', 'YGL025C', 'YGL181W', 'YBL081W', 'YDL012C', 'YPL190C', 'YML017W', 'YBR108W', 'YOR197W', 'YBR238C', 'YHR161C', 'YOR048C', 'YDL005C', 'YPL184C', 'YDR145W', 'YIL105C', 'YGL014W', 'YGL066W', 'YGL013C', 'YKL068W', 'YDR213W', 'YAL021C', 'YOL123W', 'YOR329C', 'YFR019W', 'YDR409W', 'YPR022C', 'YNL298W', 'YPR154W', 'YBR059C', 'YMR124W', 'YFL010C', 'YLR187W', 'YHR135C', 'YHR149C', 'YNL229C', 'YHR082C', 'YBL007C', 'YOR290C', 'YLL013C', 'YMR016C', 'YDL088C', 'YDR505C', 'YER040W', 'YJL141C', 'YNL288W', 'YDL161W', 'YDR081C', 'YDR293C', 'YGL049C', 'YDL189W', 'YIR023W', 'YNL251C', 'YPL204W', 'YBL051C', 'YCR077C', 'YMR173W', 'YCL037C', 'YER158C', 'YJR091C', 'YDL160C', 'YNL208W', 'YBR212W', 'YJR093C', 'YGL172W', 'YNL124W', 'YJL041W', 'YKL038W', 'YER111C', 'YPR088C', 'YLR191W', 'YER151C', 'YIR001C', 'YGR162W', 'YNR052C', 'YOR113W', 'YGR241C', 'YGL178W', 'YGR136W', 'YOR372C', 'YMR136W', 'YOL004W', 'YLR177W', 'YML117W', 'YGR178C', 'YDR515W', 'YNL154C', 'YHL024W', 'YNL278W', 'YGR119C', 'YDR251W', 'YOR098C', 'YDR206W', 'YMR216C', 'YGL215W', 'YER165W', 'YOR359W', 'YDL233W', 'YNL197C', 'YDR096W', 'YLR403W', 'YOR181W', 'YOL019W', 'YPL229W', 'YGR009C', 'YMR172W', 'YHR119W', 'YLR095C', 'YBL047C', 'YIL122W', 'YBR057C', 'YCR093W', 'YGR097W', 'YKL204W', 'YGR032W', 'YCR084C', 'YDR192C', 'YIL109C', 'YNL263C', 'YOR363C', 'YHR178W', 'YGL092W', 'YDR207C', 'YDR171W', 'YPL085W', 'YOR104W', 'YNL068C', 'YGR249W', 'YPL055C', 'YDR143C', 'YGL036W', 'YHR206W', 'YPL247C', 'YLR342W', 'YKL139W', 'YLR228C', 'YHL002W', 'YBL084C', 'YDR122W', 'YOR188W', 'YIR010W', 'YIL055C', 'YPL032C', 'YGR268C', 'YGR156W', 'YMR037C', 'YHR086W', 'YOL116W', 'YKL112W', 'YER109C', 'YHR056C', 'YPL049C', 'YJR090C', 'YHR084W', 'YDR432W', 'YPR161C', 'YIL135C', 'YNL167C', 'YGR250C', 'YDL140C', 'YMR276W', 'YPR124W', 'YDR436W', 'YJL048C', 'YIL101C', 'YOR007C', 'YMR002W', 'YDR388W', 'YNL118C', 'YPR019W', 'YLL032C', 'YDL195W', 'YJL056C', 'YNL103W', 'YKL025C', 'YER118C']

parser = argparse.ArgumentParser()
parser.add_argument('dirin')
parser.add_argument('fileout')
args = parser.parse_args()

files = os.listdir(args.dirin)
files = [file for file in files if re.match('(.+)_protein.fsa',file)]
files.sort(key=lambda file:order.index(re.match('(.+)_protein.fsa',file)[1]))

variants = {}

plaac = Plaac()

for file in files:
    fasta = plaacmod.readFasta(os.path.join(args.dirin,file))
    #print(Colors.CYAN + re.match('(.+)_protein.fsa',file)[1] + Colors.RESET)

    for f in fasta:
        match = re.match('(.+?)_(.+)',f[0])
        seqid,name = match[1],match[2]
        #print(name)
        if not name in variants:
            variants[name] = {}
        
        score = plaac.score(f[1])
        print(
            Colors.CYAN + seqid + Colors.RESET + ' ' + 
            Colors.YELLOW + name + Colors.RESET + ' ' + 
            str(score)
        )
        variants[name][seqid] = score

with open(args.fileout,'w') as f:
    table = []
    header = ['seqid']
    for name in variants:
        header.append(name)
    
    table.append(','.join(header))

    for seqid in order:
        scores = [seqid]
        for name in variants:
            val = variants[name].get(seqid)
            ref = variants['S288C'].get(seqid)
            if val:
                scores.append(ref - val)
            else:
                scores.append(None)
        
        table.append(','.join(list(map(str,scores))))
    
    f.write('\n'.join(table))
        
        
    