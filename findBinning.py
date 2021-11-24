import os
import sys
import glob
import ROOT

assert len(sys.argv) == 2
path = sys.argv[1]

for filepath in glob.glob(os.path.join(path,'**'),recursive=True):
    if filepath.endswith('.root'):
        F = ROOT.TFile(filepath)
        h = None
        for key in F.GetListOfKeys():
            if key.GetClassName().startswith('TH'):
                h = F.Get(key.GetName())
                break
        if h is None:
            continue
        bins = [round(h.GetXaxis().GetBinLowEdge(i),6) for i in range(1,h.GetNbinsX()+2)]
        F.Close()
        print (f'Root file {filepath} : ')
        print ('\t',bins, f'({len(bins)-1} bins)')
        

