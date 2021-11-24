import sys
import ROOT
import argparse
import numpy as np
from root_numpy import hist2array, array2hist

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = 2001

parser = argparse.ArgumentParser(description='Plotter of systematic contributions')
parser.add_argument('--file', action='store', required=True, type=str, 
                    help='Root file containing the nominal and systematics')
parser.add_argument('--nominal', action='store', required=True, type=str, default=None,
                    help='Name of nominal histogram')
parser.add_argument('--output', action='store', required=False, type=str, default='SystHist.pdf',
                    help='Name of output PDF [default = SystHist.pdf]')
parser.add_argument('--xlabel', action='store', required=False, type=str, default='',
                    help='Title of xlabel [optionnal]')
parser.add_argument('--rebin', action='store', required=False, type=int, default=None,
                    help='Rebin factor [optionnal]')
args = parser.parse_args()


f = ROOT.TFile(args.file)
if not f.Get(args.nominal):
    raise RuntimeError("Could not find histogram {} in file {}".format(args.nominal,f))
hnom = f.Get(args.nominal)
hnom.GetXaxis().SetTitle(args.xlabel)

if args.rebin is not None:
    hnom = hnom.Rebin(args.rebin)

hups = {}
hdowns = {}
for key in f.GetListOfKeys():
    if args.nominal in key.GetName():
        if key.GetName().endswith("down"):
            systName = key.GetName().split("__")[1][:-4]
            hdowns[systName] = f.Get(key.GetName())
            if args.rebin is not None:
                hdowns[systName] = hdowns[systName].Rebin(args.rebin)
        elif key.GetName().endswith("up"):
            systName = key.GetName().split("__")[1][:-2]
            hups[systName] = f.Get(key.GetName())
            if args.rebin is not None:
                hups[systName] = hups[systName].Rebin(args.rebin)

hmax = max([hup.GetMaximum() for hup in hups.values()])
hnom.SetMaximum(hmax*1.1)

min_colors = 51
max_colors = 100

for i,systName in enumerate(hups.keys()):
    col = round(min_colors+i*(max_colors-min_colors)/len(hups.keys()))
    hups[systName].SetLineColor(col)
    hdowns[systName].SetLineColor(col)
    hups[systName].SetLineStyle(2)
    hdowns[systName].SetLineStyle(3)
    i += 2

hnom.SetLineColor(1)
hnom.SetLineWidth(2)

    
c = ROOT.TCanvas()
c.Print(args.output+'[')

c.Clear()
hnom.Draw("hist")
all_leg = ROOT.TLegend(0.5,0.4,0.98,0.98)
for systName in hups.keys():
    hups[systName].Draw("hist same")
    hdowns[systName].Draw("hist same")
    all_leg.AddEntry(hups[systName],"{} - Up".format(systName))
    all_leg.AddEntry(hdowns[systName],"{} - Down".format(systName))
all_leg.SetNColumns(2)
all_leg.SetFillStyle(0)
all_leg.Draw()
c.Print(args.output,'Title:All systematics')

# One stat per plot #
for systName in hups.keys():
    print (systName)
    c.Clear()
    one_leg = ROOT.TLegend(0.65,0.7,0.98,0.98)
    one_leg.SetTextSize(0.033)
    hnom.SetTitle(systName)
    hnom.Draw("hist")
    hups[systName].Draw("hist same")
    hdowns[systName].Draw("hist same")
    one_leg.AddEntry(hups[systName],"#splitline{{{} - Up}}{{Integral = {:.6f}}}".format(systName,hups[systName].Integral()))
    one_leg.AddEntry(hnom,"#splitline{{{} - Nominal}}{{Integral = {:.6f}}}".format(systName,hnom.Integral()))
    one_leg.AddEntry(hdowns[systName],"#splitline{{{} - Down}}{{Integral = {:.6f}}}".format(systName,hdowns[systName].Integral()))
    one_leg.Draw()
    c.Print(args.output,'Title:'+systName)

c.Print(args.output+']')

