import os
import sys
import copy
import json
import functools
import numpy as np
from IPython import embed
import logging
import ROOT

class NonClosureDY:
    def __init__(self,**kwargs):
        self.path_json = os.path.abspath(os.path.join(os.path.dirname(__file__),kwargs['path_json']))
        logging.info(f'\tLoading {self.path_json}')
        with open(self.path_json,'r') as handle:
            self.content = json.load(handle)

    @property
    def group(self):
        return 'DY'

    @property
    def categories(self):
        return self.content.keys()

    @staticmethod
    def _correct_nom(x,coefficients):
        return coefficients[0] + coefficients[1] * x
        #return lambda x : coefficients[0] + coefficients[1] * x

    @staticmethod
    def _shift1_up(x,coeffs,eigenVals,eigenVecs):
        return (coeffs[0] + np.sqrt(eigenVals[0]) * eigenVecs[0][0]) \
             + (coeffs[1] + np.sqrt(eigenVals[0]) * eigenVecs[1][0]) * x
        #return lambda x : (coeffs[0] + np.sqrt(eigenVals[0]) * eigenVecs[0][0]) \
        #                + (coeffs[1] + np.sqrt(eigenVals[0]) * eigenVecs[1][0]) * x
    @staticmethod
    def _shift1_down(x,coeffs,eigenVals,eigenVecs):
        return (coeffs[0] - np.sqrt(eigenVals[0]) * eigenVecs[0][0]) \
             + (coeffs[1] - np.sqrt(eigenVals[0]) * eigenVecs[1][0]) * x
        #return lambda x : (coeffs[0] - np.sqrt(eigenVals[0]) * eigenVecs[0][0]) \
        #                + (coeffs[1] - np.sqrt(eigenVals[0]) * eigenVecs[1][0]) * x
    @staticmethod
    def _shift2_up(x,coeffs,eigenVals,eigenVecs):
        return  (coeffs[0] + np.sqrt(eigenVals[1]) * eigenVecs[0][1]) \
              + (coeffs[1] + np.sqrt(eigenVals[1]) * eigenVecs[1][1]) * x
        #return lambda x : (coeffs[0] + np.sqrt(eigenVals[1]) * eigenVecs[0][1]) \
        #                + (coeffs[1] + np.sqrt(eigenVals[1]) * eigenVecs[1][1]) * x
    @staticmethod
    def _shift2_down(x,coeffs,eigenVals,eigenVecs):
        return (coeffs[0] - np.sqrt(eigenVals[1]) * eigenVecs[0][1]) \
             + (coeffs[1] - np.sqrt(eigenVals[1]) * eigenVecs[1][1]) * x
        #return lambda x : (coeffs[0] - np.sqrt(eigenVals[1]) * eigenVecs[0][1]) \
        #                + (coeffs[1] - np.sqrt(eigenVals[1]) * eigenVecs[1][1]) * x

    def _getParameters(self,key,x):
        entry = self.content[key]
        if isinstance(entry,dict):
            return np.array(entry['coefficients']),np.array(entry['covariance'])
        elif isinstance(entry,list):
            for e in entry:
                assert 'range' in e.keys()
                if x >= e['range'][0] and x <= e['range'][1]:
                    return np.array(e['coefficients']),np.array(e['covariance'])
            raise RuntimeError
        else:
            raise NotImplementedError


    def modify(self,h,cat,group,**kwargs):
        assert isinstance(h,ROOT.TH1)
        key = kwargs['key']
        if key not in self.categories:
            raise RuntimeError(f'Could not find key `{key}` in {self.path_json}')
        #coefficients = self.content[key]['coefficients']
        #f = self._fit(coefficients)
        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)
            coefficients,_ = self._getParameters(key,x)
            correction = self._correct_nom(x,coefficients)
            y = h.GetBinContent(i)
            h.SetBinContent(i,y*correction)

    def additional(self,h,cat,group,systName,**kwargs):
        assert isinstance(h,ROOT.TH1)
        key = kwargs['key']
        #coefficients = self.content[key]['coefficients']
        #covariance   = np.array(self.content[key]['covariance'])
        #assert covariance.shape == (2,2)
        #eigenValues , eigenVectors = np.linalg.eigh(covariance)
        h_shape1_up   = h.Clone(f"{h.GetName()}_{systName}_shape1_up")
        h_shape1_down = h.Clone(f"{h.GetName()}_{systName}_shape1_down")
        h_shape2_up   = h.Clone(f"{h.GetName()}_{systName}_shape2_up")
        h_shape2_down = h.Clone(f"{h.GetName()}_{systName}_shape2_down")

        #lambda_shape1_up    = self._shift1_up(coefficients,eigenValues,eigenVectors)
        #lambda_shape1_down  = self._shift1_down(coefficients,eigenValues,eigenVectors)
        #lambda_shape2_up    = self._shift2_up(coefficients,eigenValues,eigenVectors)
        #lambda_shape2_down  = self._shift2_down(coefficients,eigenValues,eigenVectors)

        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)
            y = h.GetBinContent(i)
            coefficients,covariance = self._getParameters(key,x)
            assert covariance.shape == (2,2)
            eigenValues , eigenVectors = np.linalg.eigh(covariance)
            h_shape1_up.SetBinContent(i,   y * self._shift1_up(x,coefficients,eigenValues,eigenVectors))
            h_shape1_down.SetBinContent(i, y * self._shift1_down(x,coefficients,eigenValues,eigenVectors))
            h_shape2_up.SetBinContent(i,   y * self._shift2_up(x,coefficients,eigenValues,eigenVectors))
            h_shape2_down.SetBinContent(i, y * self._shift2_down(x,coefficients,eigenValues,eigenVectors))

        return {f'{systName}_shape1Up'   : h_shape1_up,
                f'{systName}_shape1Down' : h_shape1_down,
                f'{systName}_shape2Up'   : h_shape2_up,
                f'{systName}_shape2Down' : h_shape2_down}

class NonClosureFake:
    def __init__(self,**kwargs):
        self.path_json = os.path.abspath(os.path.join(os.path.dirname(__file__),kwargs['path_json']))
        logging.info(f'\tLoading {self.path_json}')
        with open(self.path_json,'r') as handle:
            self.content = json.load(handle)

    @property
    def group(self):
        return 'Fakes'

    @property
    def categories(self):
        return self.content.keys()

    @staticmethod
    def _nom_up(nom):
        return lambda x : 1 + abs(1-nom)

    @staticmethod
    def _nom_down(nom):
        return lambda x : 1 - abs(1-nom)

    @staticmethod
    def _slope_up(slope,cog):
        return lambda x : 1 + min(max(slope*(x-cog),-1.),1.)

    @staticmethod
    def _slope_down(slope,cog):
        return lambda x : 1 - min(max(slope*(x-cog),-1.),1.)

    def modify(self,h,cat,group,**kwargs):
        # Decision to only use the norm effect and not the slope 
        assert isinstance(h,ROOT.TH1)
        if cat not in self.categories:
            raise RuntimeError(f'Could not find cat `{cat}` in {self.path_json}')
        if len(set(['cog','nom','slope']).intersection((self.content[kwargs['key']].keys()))) < 3:
            return
        cog = self.content[kwargs['key']]['cog']
        nom   = self.content[kwargs['key']]['nom']
        slope = self.content[kwargs['key']]['slope']
        #print (f"cog = {cog:5.3f},{nom:5.3f},{slope:5.3f}")

        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)-cog
            y = h.GetBinContent(i)
            ny = y*nom*(1+slope*x)
            h.SetBinContent(i,ny)
            #print (f"x = {x+cog:10.5f}, x-cog = {x:10.5f}, y = {y:10.5f},y*nom = {y*nom:10.5f},(1+slope*x) = {(1+slope*x):10.5f},ny = {ny:10.5f} -> diff = {2*(ny-y)/(ny+y+1e-8)*100:5.2f}%")

    def additional(self,h,cat,group,systName,**kwargs):
        assert isinstance(h,ROOT.TH1)
        h = copy.deepcopy(h)
        cog   = self.content[kwargs['key']]['cog']      if 'cog' in self.content[kwargs['key']].keys() else None
        nom   = self.content[kwargs['key']]['nom']      if 'nom' in self.content[kwargs['key']].keys() else None
        slope = self.content[kwargs['key']]['slope']    if 'slope' in self.content[kwargs['key']].keys() else None 
        
        self.modify(h,cat,group,**kwargs)

        if nom is not None:
            h_nom_up     = h.Clone(f"{h.GetName()}_{systName}_nom_up")
            h_nom_down   = h.Clone(f"{h.GetName()}_{systName}_nom_down")
            lambda_nom_up     = self._nom_up(nom)
            lambda_nom_down   = self._nom_down(nom)

        if cog is not None and slope is not None:
            h_slope_up   = h.Clone(f"{h.GetName()}_{systName}_slope_up")
            h_slope_down = h.Clone(f"{h.GetName()}_{systName}_slope_down")
            lambda_slope_up   = self._slope_up(slope,cog)
            lambda_slope_down = self._slope_down(slope,cog)

        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)
            y = h.GetBinContent(i)
            if nom is not None:
                h_nom_up.SetBinContent(i,   y * lambda_nom_up(x))
                h_nom_down.SetBinContent(i, y * lambda_nom_down(x))
            if cog is not None and slope is not None:
                h_slope_up.SetBinContent(i,   y * lambda_slope_up(x))
                h_slope_down.SetBinContent(i, y * lambda_slope_down(x))

        dRet = {}
        if nom is not None:
            dRet[f'{systName}_nomUp']   = h_nom_up
            dRet[f'{systName}_nomDown'] = h_nom_down
        if cog is not None and slope is not None:
            dRet[f'{systName}_slopeUp']   = h_slope_up
            dRet[f'{systName}_slopeDown'] = h_slope_down

        return dRet
        
        #return {f'{systName}_nomUp'     : h_nom_up,
        #        f'{systName}_nomDown'   : h_nom_down,
        #        f'{systName}_slopeUp'   : h_slope_up,
        #        f'{systName}_slopeDown' : h_slope_down}

class Theory:
    def __init__(self,**kwargs):
        self.path_json = os.path.abspath(os.path.join(os.path.dirname(__file__),kwargs['path_json']))
        logging.info(f'\tLoading {self.path_json}')
        with open(self.path_json,'r') as handle:
            self.content = json.load(handle)

    @functools.cached_property
    def group(self):
        groups = []
        for cat in self.content.keys():
            for group in self.content[cat].keys():
                if group not in groups:
                    groups.append(group)
        return groups

    def _getFactors1D(self,cat,group,x):
        up, down  = 1.,1.
        idx = -1
        for i,binCfg in enumerate(self.content[cat][group]):
            if x > binCfg['bin'][0] and x < binCfg['bin'][1]:
                up = binCfg['up']
                down = binCfg['down']
                idx = i
                break
        if idx != -1:
            del self.content[cat][group][idx]
        return up, down

    def _getFactors2D(self,cat,group,x,y):
        up, down  = 1.,1.
        idx = -1
        for i,binCfg in enumerate(self.content[cat][group]):
            if x > binCfg['binx'][0] and x < binCfg['binx'][1] and y > binCfg['biny'][0] and y < binCfg['biny'][1]:
                up = binCfg['up']
                down = binCfg['down']
                idx = i
                break
        if idx != -1:
            del self.content[cat][group][idx]
        return up, down

    def additional(self,h,cat,group,systName,**kwargs):
        if group in self.content[cat].keys():
            h_up = h.Clone(f"{h.GetName()}_{systName}_up")
            h_down = h.Clone(f"{h.GetName()}_{systName}_down")
            if h.__class__.__name__.startswith('TH1'):
                for i in range(1,h.GetNbinsX()+1):
                    x = h.GetXaxis().GetBinCenter(i)
                    val = h.GetBinContent(i)
                    up,down = self._getFactors1D(cat,group,x)
                    h_up.SetBinContent(i,val*up)
                    h_down.SetBinContent(i,val*down)
            elif h.__class__.__name__.startswith('TH2'):
                for i in range(1,h.GetNbinsX()+1):
                    for j in range(1,h.GetNbinsY()+1):
                        x = h.GetXaxis().GetBinCenter(i)
                        y = h.GetYaxis().GetBinCenter(j)
                        val = h.GetBinContent(i)
                        up,down = self._getFactors2D(cat,group,x,y)
                        h_up.SetBinContent(i,val*up)
                        h_down.SetBinContent(i,val*down)
            else:
                raise ValueError(f'Histogram type {h.__class__.__name__} from systematic {systName} not understood')

            return {f'{systName}Up'    : h_up,
                    f'{systName}Down'  : h_down}
        else:
            return {}

class WjetsAdHoc:
    def __init__(self,**kwargs):
        self._type = kwargs['type']                                                                                                                                                                        
        if self._type not in ['linear','quadratic']:
            raise RuntimeError


    @property
    def group(self):
        return 'WJets'

    def _up(self,x,xb,xm,xp):
        if self._type == 'linear':
            if x < xb: 
                a = 0.2/(xb-xm)
                b = (0.8*xb-xm)/(xb-xm)
            else:
                a = 0.2/(xp-xb)
                b = (xp-1.2*xb)/(xp-xb)
            return a * x + b 
        if self._type == 'quadratic':
            if abs(xb-xp) < abs(xb-xm):
                a = 0.4/(xb-xp)**2
                b = -0.8*xb/(xb-xp)**2
                c = 0.8+0.4*xb**2/(xb-xp)**2
            else:
                a = 0.4/(xb-xm)**2
                b = -0.8*xb/(xb-xm)**2
                c = 0.8+0.4*xb**2/(xb-xm)**2
            return a*x**2 + b*x + c 
        #if self._type == 'quadratic':
        #if self._type == 'linear':
        #    return lambda x : 0.8 + 0.4*x
        #if self._type == 'quadratic':
        #    return lambda x : 1.6*x**2-1.6*x+1.2

    def _down(self,x,xb,xm,xp):
        if self._type == 'linear':
            if x < xb: 
                a = 0.2/(xm-xb)
                b = (xm-1.2*xb)/(xm-xb)
            else:
                a = 0.2/(xb-xp)
                b = (0.8*xb-xp)/(xb-xp)
            return a * x + b 
        if self._type == 'quadratic':
            if abs(xb-xp) < abs(xb-xm):
                a = -0.4/(xb-xm)**2
                b = +0.8*xb/(xb-xm)**2
                c = 1.2-0.4*xb**2/(xb-xm)**2
            else:
                a = -0.4/(xb-xp)**2
                b = +0.8*xb/(xb-xp)**2
                c = 1.2-0.4*xb**2/(xb-xp)**2
            return a*x**2 + b*x + c

    def additional(self,h,cat,group,systName,**kwargs):
        assert isinstance(h,ROOT.TH1)
        h_up   = h.Clone(f"{h.GetName()}_{systName}_{self._type}_up")
        h_down = h.Clone(f"{h.GetName()}_{systName}_{self._type}_down")

        xb = h.GetMean()
        xm = h.GetXaxis().GetBinCenter(1)
        xp = h.GetXaxis().GetBinCenter(h.GetNbinsX())

        #embed()
        for i in range(1,h.GetNbinsX()+1):
            #x = h.GetXaxis().GetBinCenter(i)
            x = xm + (i-1)*(xp-xm)/(h.GetNbinsX()-1)
            y = h.GetBinContent(i)
            h_up.SetBinContent(i,y*self._up(x,xb,xm,xp))
            h_down.SetBinContent(i,y*self._down(x,xb,xm,xp))
            #print (i,x,self._up(x,xb,xm,xp),self._down(x,xb,xm,xp))
        h_up.Scale(h.Integral()/h_up.Integral())
        h_down.Scale(h.Integral()/h_down.Integral())

        return {f'{systName}_{self._type}_Up'   : h_up,
                f'{systName}_{self._type}_Down' : h_down}




