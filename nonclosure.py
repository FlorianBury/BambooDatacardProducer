import os
import sys
import json
import numpy as np
import ROOT

class NonClosureDY:
    def __init__(self,**kwargs):
        self.path_json = kwargs['path_json'] 
        with open(self.path_json,'r') as handle:
            self.content = json.load(handle)

    @property
    def group(self):
        return 'DY'

    @property
    def sample(self):
        return None

    @property
    def categories(self):
        return self.content.keys()

    @staticmethod
    def _fit(coefficients):
        return lambda x : coefficients[0] + coefficients[1] * x

    @staticmethod
    def _shift_nom(coefficients):
        return lambda x : coefficients[0] + coefficients[1] * x

    @staticmethod
    def _shift1_up(coeffs,eigenVals,eigenVecs):
        return lambda x : (coeffs[0] + np.sqrt(eigenVals[0]) * eigenVecs[0][0]) \
                        + (coeffs[1] + np.sqrt(eigenVals[0]) * eigenVecs[1][0]) * x
    @staticmethod
    def _shift1_down(coeffs,eigenVals,eigenVecs):
        return lambda x : (coeffs[0] - np.sqrt(eigenVals[0]) * eigenVecs[0][0]) \
                        + (coeffs[1] - np.sqrt(eigenVals[0]) * eigenVecs[1][0]) * x
    @staticmethod
    def _shift2_up(coeffs,eigenVals,eigenVecs):
        return lambda x : (coeffs[0] + np.sqrt(eigenVals[1]) * eigenVecs[0][1]) \
                        + (coeffs[1] + np.sqrt(eigenVals[1]) * eigenVecs[1][1]) * x
    @staticmethod
    def _shift2_down(coeffs,eigenVals,eigenVecs):
        return lambda x : (coeffs[0] - np.sqrt(eigenVals[1]) * eigenVecs[0][1]) \
                        + (coeffs[1] - np.sqrt(eigenVals[1]) * eigenVecs[1][1]) * x

    def modify(self,h,key,**kwargs):
        assert isinstance(h,ROOT.TH1)
        if key not in self.categories:
            raise RuntimeError(f'Could not find key `{key}` in {self.path_json}')
        coefficients = self.content[key]['coefficients']
        f = self._fit(coefficients)
        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)
            y = h.GetBinContent(i)
            h.SetBinContent(i,y*f(x))

    def additional(self,h,key,systName,**kwargs):
        assert isinstance(h,ROOT.TH1)
        coefficients = self.content[key]['coefficients']
        covariance   = np.array(self.content[key]['covariance'])
        assert covariance.shape == (2,2)
        eigenValues , eigenVectors = np.linalg.eigh(covariance)
        h_shape1_up   = h.Clone(f"{h.GetName()}_{systName}_shape1_up")
        h_shape1_down = h.Clone(f"{h.GetName()}_{systName}_shape1_down")
        h_shape2_up   = h.Clone(f"{h.GetName()}_{systName}_shape2_up")
        h_shape2_down = h.Clone(f"{h.GetName()}_{systName}_shape2_down")

        lambda_shape1_up    = self._shift1_up(coefficients,eigenValues,eigenVectors)
        lambda_shape1_down  = self._shift1_down(coefficients,eigenValues,eigenVectors)
        lambda_shape2_up    = self._shift2_up(coefficients,eigenValues,eigenVectors)
        lambda_shape2_down  = self._shift2_down(coefficients,eigenValues,eigenVectors)

        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)
            y = h.GetBinContent(i)
            h_shape1_up.SetBinContent(i,   y * lambda_shape1_up(x))
            h_shape1_down.SetBinContent(i, y * lambda_shape1_down(x))
            h_shape2_up.SetBinContent(i,   y * lambda_shape2_up(x))
            h_shape2_down.SetBinContent(i, y * lambda_shape2_down(x))

        return {f'{systName}_shape1Up'   : h_shape1_up,
                f'{systName}_shape1Down' : h_shape1_down,
                f'{systName}_shape2Up'   : h_shape2_up,
                f'{systName}_shape2Down' : h_shape2_down}

class NonClosureFake:
    def __init__(self,**kwargs):
        self.path_json = kwargs['path_json'] 
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
    def _slope_up(slope):
        return lambda x : 1 + max(min(slope*x,1.),-1.)

    @staticmethod
    def _slope_down(slope):
        return lambda x : 1 - max(min(slope*x,1.),-1.)

    def modify(self,h,key,**kwargs):
        # Decision to only use the norm effect and not the slope 
        pass
        #assert isinstance(h,ROOT.TH1)
        #if key not in self.categories:
        #    raise RuntimeError(f'Could not find key `{key}` in {self.path_json}')
        #cog = self.content[key]['cog']

        #for i in range(1,h.GetNbinsX()+1):
        #    x = h.GetXaxis().GetBinCenter(i)-cog
        #    y = h.GetBinContent(i)
        #    h.SetBinContent(i,y*f(x))

    def additional(self,h,key,systName,**kwargs):
        assert isinstance(h,ROOT.TH1)
        cog   = self.content[key]['cog']
        nom   = self.content[key]['nom']
        slope = self.content[key]['slope']
        
        h_nom_up     = h.Clone(f"{h.GetName()}_{systName}_nom_up")
        h_nom_down   = h.Clone(f"{h.GetName()}_{systName}_nom_down")
        h_slope_up   = h.Clone(f"{h.GetName()}_{systName}_slope_up")
        h_slope_down = h.Clone(f"{h.GetName()}_{systName}_slope_down")

        lambda_nom_up     = self._nom_up(nom)
        lambda_nom_down   = self._nom_down(nom)
        lambda_slope_up   = self._slope_up(nom)
        lambda_slope_down = self._slope_down(nom)

        for i in range(1,h.GetNbinsX()+1):
            x = h.GetXaxis().GetBinCenter(i)-cog
            y = h.GetBinContent(i)
            h_nom_up.SetBinContent(i,   y * lambda_nom_up(x))
            h_nom_down.SetBinContent(i, y * lambda_nom_down(x))
            h_slope_up.SetBinContent(i,   y * lambda_slope_up(x))
            h_slope_down.SetBinContent(i, y * lambda_slope_down(x))

        
        #return {f'{systName}_nomUp'     : h_nom_up,
        #        f'{systName}_nomDown'   : h_nom_down,
        #        f'{systName}_slopeUp'   : h_slope_up,
        #        f'{systName}_slopeDown' : h_slope_down}
        # Decision to only use the norm effect and not the slope 

        return {f'{systName}_nomUp'     : h_nom_up,
                f'{systName}_nomDown'   : h_nom_down}
