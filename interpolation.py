import os
import math
import ctypes
import logging
import ROOT
from time import perf_counter

path = os.path.abspath(os.path.dirname(__file__))
ROOT.gInterpreter.ProcessLine(f'#include "{os.path.join(path,"th1fmorph.cc")}"')
ROOT.gInterpreter.ProcessLine(f'#include "{os.path.join(path,"th2fmorph.cc")}"')
ROOT.gInterpreter.ProcessLine(f'#include "{os.path.join(path,"morphing.h")}"')

from IPython import embed

class Interpolation:
    def __init__(self,p1,p2,p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        if self.p3 > self.p2 or self.p3 < self.p1:
            raise RuntimeError(f'Extrapolation dangerous : p1 = {self.p1}, p2 = {self.p2}, p3 = {self.p3}')
        self.w1 = 1-(self.p3-self.p1)/(self.p2-self.p1)
        self.w2 = 1-(self.p2-self.p3)/(self.p2-self.p1)

    def scaleHistogram(self,h,norm1,norm2):
        if norm1 == norm2:
            norm = norm1
        else:
            norm = self.w1*norm1 + self.w2*norm2
        if h.Integral() != 0.:
            h.Scale(norm/h.Integral())

    def __call__(self,h1,h2,name):
        if h1.__class__.__name__.startswith('TH1') and h2.__class__.__name__.startswith('TH1'):
            return self.interpolate1D(h1,h2,name)
        elif h1.__class__.__name__.startswith('TH2') and h2.__class__.__name__.startswith('TH2'):
            return self.interpolate2D(h1,h2,name)
        else:
            raise RuntimeError(f"Could not find interpolation method for h1 of class {h1.__class__.__name__} and h2 of class {h2.__class__.__name__}")
        # TH2 inherits from TH1 so cannot use isinstance

    def interpolate1D(self,h1,h2,name):
        h3 = ROOT.th1fmorph(name,name, h1, h2, self.p1, self.p2, self.p3, 1.)
        err1 = ctypes.c_double(0.)
        err2 = ctypes.c_double(0.)
        err3 = ctypes.c_double(0.)
        norm1 = h1.IntegralAndError(1,h1.GetNbinsX(),err1)
        norm2 = h2.IntegralAndError(1,h2.GetNbinsX(),err2)
        self.scaleHistogram(h3,norm1,norm2)
        norm3 = h3.IntegralAndError(1,h3.GetNbinsX(),err3)
        if norm3 != 0.:
            intErr3 = math.sqrt(self.w1 * err1.value**2 + self.w2 * err2.value**2)
            factor = intErr3/err3.value
            for i in range(1,h3.GetNbinsX()+1): 
                h3.SetBinError(i,h3.GetBinError(i)*factor)
        return h3
    
    def interpolate2D(self,h1,h2,name):
        # Find shifts of HME #
        h1y = h1.ProjectionY(h1.GetName()+'_py')
        h2y = h2.ProjectionY(h2.GetName()+'_py')
        h3y = self.__call__(h1y,h2y,h1y.GetName()+'_1dinterp_py')
        binmax1 = self.findMax(h1y) 
        binmax2 = self.findMax(h2y)
        binmax3 = self.findMax(h3y)
        shift1 = binmax3-binmax1
        shift2 = binmax3-binmax2
        if shift1 < 0:
            logging.warning(f'Negative shift 1 : {shift1}')
        if shift2 > 0:
            logging.warning(f'Positive shift 2 : {shift2}')
        
        # Correct h1 and h2 in terms of HME #
        self.h1s = self.shiftY(h1,shift1)
        self.h2s = self.shiftY(h2,shift2)

        # Interpolate using the shifted histograms #
        h3 = ROOT.th2fmorph(name,name,self.h1s,self.h2s,self.p1,self.p2,self.p3,True) # x axis projections

        #h3 = ROOT.MomentMorphing(name,name, h1, h2, float(self.p1), float(self.p2), float(self.p3), 1.)
        #embed()

        err1 = ctypes.c_double(0.)
        err2 = ctypes.c_double(0.)
        err3 = ctypes.c_double(0.)
        norm1 = h1.IntegralAndError(1,h1.GetNbinsX(),1,h1.GetNbinsY(),err1)
        norm2 = h2.IntegralAndError(1,h2.GetNbinsX(),1,h2.GetNbinsY(),err2)
        self.scaleHistogram(h3,norm1,norm2)
        norm3 = h3.IntegralAndError(1,h3.GetNbinsX(),1,h3.GetNbinsY(),err3)
        if norm3 != 0.:
            intErr3 = math.sqrt(self.w1 * err1.value**2 + self.w2 * err2.value**2)
            factor = intErr3/err3.value
            for i in range(1,h3.GetNbinsX()+1): 
                for j in range(1,h3.GetNbinsY()+1): 
                    h3.SetBinError(i,j,h3.GetBinError(i,j)*factor)
        return h3

    @staticmethod
    def shiftY(h,shift):
        assert h.__class__.__name__.startswith('TH2')
        h_tmp = getattr(ROOT,h.__class__.__name__)(h.GetName()+'_tmp',h.GetName()+'_tmp',
                                                   h.GetNbinsX(),
                                                   h.GetXaxis().GetBinLowEdge(1),
                                                   h.GetXaxis().GetBinUpEdge(h.GetNbinsX()),
                                                   h.GetNbinsY(),
                                                   h.GetYaxis().GetBinLowEdge(1),
                                                   h.GetYaxis().GetBinUpEdge(h.GetNbinsY()))
                                                   
        for x in range(1,h.GetNbinsX()+1):
            for y in range(1,h.GetNbinsY()+1):
                if y == 1:
                    h_tmp.SetBinContent(x,y,h.GetBinContent(x,y))
                elif shift > 0:
                    if (h.GetNbinsY()-y)>shift:
                        h_tmp.SetBinContent(x,y,h.GetBinContent(x,y-shift))
                    else:
                        h_tmp.SetBinContent(x,y,h.GetBinContent(x,y))
                elif shift < 0:
                    if y > shift:
                        h_tmp.SetBinContent(x,y,h.GetBinContent(x,y-shift))
                    else:
                        h_tmp.SetBinContent(x,y,h.GetBinContent(x,y))
                else:
                    h_tmp.SetBinContent(x,y,0.)

        return h_tmp

    @staticmethod
    def findMax(h):
        """ Find maximum of the histogram except the first bin (failed HME) """
        maxh = 0.
        imax = 0
        for i in range(2,h.GetNbinsX()+1):
            y = h.GetBinContent(i)
            if y > maxh:
                maxh = y
                imax = i
        return imax




def InterpolateContent(content1,content2,param1,param2,paramInt,matchingHist,matchingGroup,era):
    interpolate = Interpolation(float(param1),float(param2),float(paramInt)) 
    contInt = {}    
    matchHist1 = [f'{val[param1]}_{era}' for val in matchingHist.values()]
    matchHist2 = [f'{val[param2]}_{era}' for val in matchingHist.values()]
    if len(set(content1.keys())- set(matchHist1)) > 0:
        raise RuntimeError('Categories in content 1 not present in matching hist dict : '+', '.join([hist for hist in content1.keys() if hist not in matchHist1]))
    if len(set(content2.keys())- set(matchHist2)) > 0:
        raise RuntimeError('Categories in content 2 not present in matching hist dict : '+', '.join([hist for hist in content2.keys() if hist not in matchHist2]))
    # Checks on content #
    for histNameInt in matchingHist.keys():
        histName1 = matchingHist[histNameInt][param1] + f'_{era}'
        histName2 = matchingHist[histNameInt][param2] + f'_{era}'
        if histName1 not in content1.keys() or histName2 not in content2.keys():
            continue
        histNameInt += f'_{era}'
        logging.info(f'Interpolating between histograms {histName1} [param = {param1}] and {histName2} [param = {param2}] -> {histNameInt} [param = {paramInt}]')
        contInt[histNameInt] = {}
        matchGroup1 = [val[param1] for val in matchingGroup.values()]
        matchGroup2 = [val[param2] for val in matchingGroup.values()]
        if len(set(content1[histName1].keys()) - set(matchGroup1)) > 0:
            raise RuntimeError(f'Groups present in content 1 with category {histName1} not present in matching group dict : '+','.join([group for group in content1[histName1].keys() if group not in matchGroup1]))
        if len(set(content2[histName2].keys()) - set(matchGroup2)) > 0:
            raise RuntimeError(f'Groups present in content 2 with category {histName2} not present in matching group dict : '+','.join([group for group in content2[histName2].keys() if group not in matchGroup2]))
        for groupInt in matchingGroup.keys():
            contInt[histNameInt][groupInt] = {}
            group1 = matchingGroup[groupInt][param1]
            group2 = matchingGroup[groupInt][param2]
            if set(content1[histName1][group1].keys()) != set(content2[histName2][group2].keys()):
                raise RuntimeError(f'Different systematics between content1 [hist {histName1} of group {group1}] and content2 [hist {histName2} of group {group2}]')
            start = perf_counter()
            logging.info(f'... Processing {group1} and {group2} -> {groupInt} : {len(content1[histName1][group1].keys()):3d} histograms')
            for systName in content1[histName1][group1].keys():
                h1 = content1[histName1][group1][systName]
                h2 = content2[histName2][group2][systName]
                h3 = interpolate(h1,h2,h1.GetName()+"interp")
                contInt[histNameInt][groupInt][systName] = h3
            stop = perf_counter()
            logging.info(f'... Time = {stop-start:7.2f} s -> {(stop-start)/len(content1[histName1][group1].keys()):5.3f} s/histogram')
    return contInt


