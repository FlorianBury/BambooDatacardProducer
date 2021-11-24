import numpy as np
from copy import deepcopy
from IPython import embed
import ROOT

def getBinning(h):
    assert isinstance(h,ROOT.TH2)
    x = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1,h.GetNbinsX()+2)])
    y = np.array([h.GetYaxis().GetBinLowEdge(i) for i in range(1,h.GetNbinsY()+2)])
    return x,y

def MomentMorphing(name,title,h1,h2,par1,par2,par3,norm=1.):
    assert par3 <= par2 and par1 <= par3

    x1,y1 = getBinning(h1)
    x2,y2 = getBinning(h2)

    if not np.all(x1 == x2):
        raise RuntimeError('X binning of histograms is not equal')
    if not np.all(y1 == y2):
        raise RuntimeError('Y binning of histograms is not equal')

    if h1.Integral() <= 0. or h2.Integral() <= 0.:
        h3 = ROOT.TH2F(name,title,len(x1)-1,x1,len(y1)-1,y1)
        x3,y3 = getBinning(h3)
        assert np.all(x1 == x3)
        assert np.all(y1 == y3)
        return h3

    xVar = ROOT.RooRealVar("xVar","xVar",x1.min(),x1.max())
    yVar = ROOT.RooRealVar("yVar","yVar",y1.min(),y1.max())
    #xVar = ROOT.RooRealVar("xVar","xVar",(x1[0]+x1[1])/2,(x1[-1]+x1[-2])/2)
    #yVar = ROOT.RooRealVar("yVar","yVar",(y1[0]+y1[1])/2,(y1[-1]+y1[-2])/2)

    parVar = ROOT.RooRealVar("parVar","parVar",par1,par2)

    listOfMorphs = ROOT.RooArgList("listOfMorphs")
    paramVec = ROOT.TVectorD(2) 
    paramVec[0] = par1
    paramVec[1] = par2
    listPdfs = []
    listHist = []

    h1D = ROOT.RooDataHist(h1.GetName()+"DataHist",h1.GetName()+"DataHist",ROOT.RooArgList(xVar,yVar),h1)
    h2D = ROOT.RooDataHist(h2.GetName()+"DataHist",h2.GetName()+"DataHist",ROOT.RooArgList(xVar,yVar),h2)
    listHist.append(h1D)
    listHist.append(h2D)
    h1Pdf = ROOT.RooHistPdf(h1.GetName()+"Pdf",h1.GetName()+"Pdf",ROOT.RooArgSet(xVar,yVar),h1D)
    h2Pdf = ROOT.RooHistPdf(h2.GetName()+"Pdf",h2.GetName()+"Pdf",ROOT.RooArgSet(xVar,yVar),h2D)
    
    listPdfs.append(deepcopy(h1Pdf))
    listPdfs.append(deepcopy(h2Pdf))


    for pdf in listPdfs:
        listOfMorphs.add(pdf)

    morph = ROOT.RooMomentMorph('morph','morph',
                                parVar,
                                ROOT.RooArgList(xVar,yVar),
                                listOfMorphs,
                                paramVec,
                                ROOT.RooMomentMorph.Linear)
                                #ROOT.RooMomentMorph.NonLinear)
                                #ROOT.RooMomentMorph.SineLinear)

    parVar.setVal(par3)
    h3 = morph.createHistogram(name,
                               xVar,
                               ROOT.RooFit.Binning(x1.shape[0]-1),
                               ROOT.RooFit.YVar(yVar,ROOT.RooFit.Binning(y1.shape[0]-1)))

    h3.SetTitle(title)
    h3.Scale(norm/h3.Integral())

    try:
        x3,y3 = getBinning(h3)
        assert np.all(x1 == x3)
        assert np.all(y1 == y3)
    except:
        print ('weird thing in binning')
        embed()
            
    return h3


