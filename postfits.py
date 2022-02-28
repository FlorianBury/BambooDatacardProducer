import os
import sys
import copy
import math
import ROOT
import logging
import numpy as np
import ctypes
from IPython import embed

from context import TFileOpen

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

class PostfitPlots:
    def __init__(self,bin_name,output_path,fit_diagnostics_path,processes,fit_type,header,analysis,eras,categories,bin_edges=None,labels=None,label_positions=None,plot_options={},unblind=False,show_ratio=False,sort_by_yield=True,energy='13 TeV',verbose=False):
        """
            Class that performs the postfit plots

            # List of arguments #
            - bin_name [str]                : name given to the plot 
            - output_path [str]             : output path for the pdfs (if does not exist, will be created)
            - fit_diagnostics_path [str]    : path to the fitdiagnostic file [warning : see note below]
            - processes [dict]              : dict of process name and plotting options [see below]
            - fit_type [str]                : prefit | postfit_b | postfit_s [not other option allowed], used for the plotting and for finding correct subdir in fitdiag file
            - header [str]                  : name that will be printed on the plot (eg, the category)
            - analysis [str]                : name of the analysis (eg, HH, ttH, ...) to be printed for the case of postfit shapes as "#mu(HH) = "
            - eras [str/list(str)]          : eras to be looked into (can be a single one, or list of eras to be aggregated), lumi is taken from there, /!\ Only 2016, 2017 and 2018 coded so far
            - categories [list(str)]        : names of the categories (= combine bins) to be taken from the fitdiag file (without the era in the name)
            - bin_edges [list(list/float)]  : list of bin edges to rebin the histogram from combine (if multiple categories, need to provide one set of bin edges per category)
            - labels [list(str)]            : list of categories labels to be put on top of each category (if several) [OPTIONAL]
            - label_positions [list(float)] : x positions of the labels [OPTIONAL], if not provided and labels are, will find some "smart" positions
            - plot_options [dict]           : plotting options to override default [see below]
            - unblind [bool]                : whether to show data [DEFAULT=False]
            - show_ratio [bool]             : whether to show the bottom plot [DEFAULT=False]
            - sort_by_yield [bool]          : whether to sort the MC backgrounds per yields [DEFAULT=True], if False will plot the in the other given in the processes
            - verbose [bool]                : show verbose log [DEFAULT=False]

            # Notes #
            - Names of categories in the fitdiag (and datacard) need to be of type {category}_{era}
            - Do not put data in the processes dict

            # Processes #
            Dictionary needs to be like the following :
            {'<processName>' : {'label':'<label>','color':'<color>','type':'<type>','group':'<group>'}, [...]}
            where  : 
                - <processName> : name of the process to be taken from the fitfiag file
                - <label>       : what to put in the legend
                - <type>        : mc | signal
                - <color>       : color of the process, can be number (see TColor) or a hex value
                - <group>       : group name (just used internally), can be the same as processName, useful when merging processes (eg, VH, WH, ttH, etc into single H)
            optional args : 
                - 'scale'       : float to scale the process, useful for small signals (no effect on MC background)
                - 'fill_style'  : fill style (see TAttLine), can be used for both signal and mc background)

            # Plot options #
            Some of the plotting options (eg canvas and pad sizes) are hardcoded but many others can be overriden from their defaults values.
            These are passed as a dictionary to plot_options, full example is below 
            plot_options = {
                'legend': {
                    'columns': <n_columns>,
                    'textsize': <textsize>,
                    'textfont': <textfont>,
                    'headersize': <headersize>,
                    'headerfont': <headerfont>,
                 },
                'x-axis': {'textsize': <textsize>,
                           'textfont': <textfont>, 
                           'labelsize': <labelsize>,
                           'offset': <offset>,
                           'label': <x-axis label>,
                           'divisions': <number of divisions>
                           'min': <min x-axis>,
                           'max': <max x-axis>,
                },
                'y-axis': {'textsize': <textsize>,
                           'textfont': <textfont>, 
                           'labelsize': <labelsize>,
                           'offset': <offset>,
                           'label': <y-axis label>,
                           'divisions': <number of divisions>
                           'min': <min y-axis>,
                           'max': <max y-axis>
                },
                'logy': True/False,
                'logx': True/False,
                'ratio':{'min': <min y-ratio-axis>,   # only applied if show_ratio is True
                         'min': <max y-ratio-axis>,
                         'offset': <offset>,
                         'textsize': <textsize>,
                         'textfont': <textfont>, 
                         'labelsize': <labelsize>,
                },
                'labels':{'textsize': <textsize>, # only if several categories and labels is provided
                          'textfont': <textfont>, 
                }
            }

            Please note that most of those options have reasonnable default values, maybe try without first and then improve on that
        """
        # Init #
        self._bin_name              = bin_name
        self._output_path           = output_path
        self._fit_diagnostics_path  = fit_diagnostics_path
        self._processes             = processes
        self._fit_type              = fit_type
        self._analysis              = analysis
        self._eras                  = eras
        self._categories            = categories
        self._bin_edges             = bin_edges
        self._labels                = labels
        self._label_positions       = label_positions
        self._plot_options          = plot_options
        self._unblind               = unblind
        self._show_ratio            = show_ratio
        self._sort_by_yield         = sort_by_yield

        # Verbosity #
        logging.basicConfig(level   = logging.DEBUG,
                            format  = '%(asctime)s - %(levelname)s - %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S')

        if not verbose:
            logging.getLogger().setLevel(logging.INFO)

        # Determine the folder name #
        if self._fit_type == 'prefit':
            folderName = 'shapes_prefit'
            self._header_legend = f'{header}, prefit'
        elif self._fit_type == 'postfit_b':
            folderName = 'shapes_fit_b'
            self._header_legend = f'{header}, #mu({self._analysis})=#hat{{#mu}}'
        elif self._fit_type == 'postfit_s':
            folderName = 'shapes_fit_s'
            self._header_legend = f'{header}, #mu({self._analysis})=#hat{{#mu}}'
        else:
            raise RuntimeError(f'Unknown fit_type {self._fit_type}')

        # Check processes #
        must_have = {'label','color','type','group'}
        self._processes['data'] = {'group':'data','type':'data','color':1,'label':'Data'}
        self._processes['total_background'] = {'group':'total','type':'mc','color':12,'label':'Uncertainty'}
        self._options = {}
        if not isinstance(self._processes,dict):
            raise RuntimeError('`processes` entry needs to be a dict')
        for pName,pCfg in self._processes.items():
            if not isinstance(pCfg,dict):
                raise RuntimeError(f'Process {pName} needs to be a dict') 
            if len(must_have - set(pCfg.keys())) > 0:
                raise RuntimeError(f'Process {pName} is missing {[key for key in pCfg.keys() if key not in must_have]}')
            self._options[pCfg['group']] = {k:v for k,v in pCfg.items() if k != 'group'}

        # Check categories #
        if isinstance(self._categories,list):
            pass
        elif isinstance(self._categories,str):
            self._categories = [self._categories]
        else:
            raise RuntimeError(f'`categories` entry type {type(self._categories)} not understood')

        # Check eras #
        if isinstance(self._eras,list):
            pass
        elif isinstance(self._eras,str):
            self._eras = [self._eras]
        else:
            raise RuntimeError(f'`eras` entry type {type(self._eras)} not understood')

        # Get correct lumi format #
        lumis = {'2016':36.330,'2017':41.530,'2018':59.740,'HL':3000.}
        lumi = 0.
        for era in self._eras:
            if era not in lumis.keys():
                raise RuntimeError(f'Era {era} not understood')
            lumi += lumis[era]
        # PC recommendation : always three digits
        if lumi > 100:
            lumi = str(round(lumi))
        else:
            lumi = str(round(lumi,1))
        lumi += ' fb^{-1} '+f'({energy})'

        # Bin edges #
        if self._bin_edges is not None:
            if not isinstance(self._bin_edges[0],list):
                self._bin_edges = [self._bin_edges]

        # Open root file and get folder subdir # 
        if not os.path.exists(self._fit_diagnostics_path):
            raise RuntimeError(f'`{self._fit_diagnostics_path}` does not exist')

        self._histograms = {}
        logging.info(f'Opening file {self._fit_diagnostics_path}')
        with TFileOpen(self._fit_diagnostics_path,'r') as F:
            logging.debug(f'Opened {self._fit_diagnostics_path}')
            if not hasattr(F,folderName):
                raise RuntimeError(f'Folder `{folderName}` cannot be found in {self._fit_diagnostics_path}')
            folder = getattr(F,folderName)
            logging.debug(f'Entered folder {folderName}')
            for cat in self._categories:
                for era in self._eras:
                    catEra = f'{cat}_{era}'
                    if not hasattr(folder,catEra):
                        raise RuntimeError(f'Folder `{folderName}` does not contain category {catEra}')
                    folderCat = getattr(folder,catEra)
                    logging.debug(f'Opened category {catEra}')
                    if cat not in self._histograms.keys():
                        self._histograms[cat] = {}
                    self._getProcesses(folderCat,cat)
                    logging.info(f'Loaded from {folderCat.GetTitle()}')
        logging.info('Importation of histograms done')

        # "un-square" the data errors # 
        if self._unblind:
            for cat in self._histograms.keys():
                for i in range(self._histograms[cat]['data'].GetN()):
                    self._histograms[cat]['data'].SetPointError(i,
                       self._histograms[cat]['data'].GetErrorXlow(i),
                       self._histograms[cat]['data'].GetErrorXhigh(i),
                       math.sqrt(self._histograms[cat]['data'].GetErrorYlow(i)),
                       math.sqrt(self._histograms[cat]['data'].GetErrorYhigh(i)))

        # Get the number of bins for each category #
        self._bins = []
        for cat in self._histograms.keys():
            for h in self._histograms[cat].values():
                if h.__class__.__name__.startswith('TH1'):
                    self._bins.append(h.GetNbinsX())
                    break

        # Declare legend #
        legend = self._getLegend()

        # Process histograms per group #
        logging.debug('Processing histograms')
        self._histograms['__combined__'] = {}
        for group,optCfg in self._options.items():
            list_hist = [self._histograms[cat][group] for cat in categories if group in self._histograms[cat].keys()]
            if len(list_hist) == 0:
                continue
            if group == 'data':
                # Get combined data #
                self._histograms['__combined__']['data'] = self._processDataGraphs(list_hist)
                # Add to legend #
                # Esthetics #
                self._histograms['__combined__']['data'].SetMarkerColor(1)
                self._histograms['__combined__']['data'].SetMarkerStyle(20)
                self._histograms['__combined__']['data'].SetMarkerSize(1.)
                self._histograms['__combined__']['data'].SetLineColor(1)
                self._histograms['__combined__']['data'].SetLineWidth(1)
                self._histograms['__combined__']['data'].SetLineStyle(1)
            elif optCfg['type'] in ['mc','signal']:
                # Get combined hist #
                combined_hist = self._processBackgroundHistograms(list_hist)
                if combined_hist is not None:
                    self._histograms['__combined__'][group]  = combined_hist
                else:
                    continue
                # Esthetics #
                color = optCfg['color']
                if isinstance(color,str) and color.startswith('#'):
                    # Color is a hex string #
                    color = ROOT.TColor.GetColor(color)
                if optCfg['type'] == 'signal':
                    self._histograms['__combined__'][group].SetMarkerSize(0)
                    self._histograms['__combined__'][group].SetLineWidth(2)
                    self._histograms['__combined__'][group].SetLineColor(color)
                    self._histograms['__combined__'][group].SetFillColorAlpha(color, 0.50)
                if optCfg['type'] == 'mc':
                    self._histograms['__combined__'][group].SetMarkerSize(0)
                    self._histograms['__combined__'][group].SetLineWidth(0)
                    self._histograms['__combined__'][group].SetLineColor(color)
                    self._histograms['__combined__'][group].SetFillColor(color)
                if 'fill_style' in optCfg:
                    self._histograms['__combined__'][group].SetFillStyle(optCfg['fill_style'])
                # Scale in case signal #
                if optCfg['type'] == 'signal' and 'scale' in optCfg.keys():
                    self._histograms['__combined__'][group].Scale(float(optCfg['scale']))
            else:
                raise RuntimeError(f'Unknown type {optCfg["type"]} for group {group}')
        logging.debug('... done')


        # Make stack background #
        logging.debug('Stacking the MC backgrounds')
        stack_MC = self._getBackgroundStack()
        logging.debug('... done')
                
        # Add to legend #
        if self._unblind:
            legend.AddEntry(self._histograms['__combined__']['data'],"Data","p")
        for group in self._order:
            # Get options #
            optCfg = self._options[group]
            if optCfg['type'] == 'mc' and group != 'total':
                legend.AddEntry(self._histograms['__combined__'][group],optCfg['label'],'f')
        legend.AddEntry(self._histograms['__combined__']['total'],"Uncertainty","f")

        # Start plotting #
        logging.debug('Starting plotting')
        canvas = self._getCanvas()
        topPad = self._getTopPad()
        topPad.Draw()

        # Top pad #
        logging.debug('Plotting top pad')
        topPad.cd()
        opt = 'lin'
        if 'logy' in self._plot_options.keys() and self._plot_options['logy']:
            topPad.SetLogy() 
            opt = 'log'
        if 'logx' in self._plot_options.keys() and self._plot_options['logx']:
            topPad.SetLogx() 
            opt = 'log'

        # Make template #
        template = self._getTemplate()
        self._changeLabels(template)
        template.Draw('')
        # Draw stack #
        stack_MC.Draw('hist same')
        stack_MC.Draw('hist same axis') # otherwise erases the axes
        # Draw uncertainty #
        self._histograms['__combined__']['total'].Draw('E2 same')
        # Draw signals 
        for group in self._order:
            # Get options #
            optCfg = self._options[group]
            if optCfg['type'] == 'signal':
                self._histograms['__combined__'][group].Draw('hist same')
                # Add to legend #
                legend.AddEntry(self._histograms['__combined__'][group],optCfg['label'],'f')

        # Draw data if unblinded #
        if self._unblind:
            self._histograms['__combined__']['data'].Draw('E1P same')
        # Draw lumi labels #
        CMS_labels = self._getCMSLabels(lumi)
        for label in CMS_labels:
            label.Draw()
        # Draw lines #
        lines,labels = self._getSeparations()
        for line in lines:
            line.Draw()
        for label in labels:
            label.Draw()
        # Draw legend #
        legend.Draw('same')
        logging.debug('... done')

        # Bottom pad #
        if self._show_ratio:
            logging.debug('Plotting bottom pad')
            canvas.cd()
            bottomPad = self._getBottomPad()
            bottomPad.Draw()
            bottomPad.cd()
            # Get ratios #
            if self._unblind:
                err_data,maxabsr = self._getDataError(self._histograms['__combined__']['total'],
                                                      self._histograms['__combined__']['data'])
                # Need to get first to get range of data points for the ratio #
            else:
                maxabsr = 0.
            err_hist = self._getTotalHistError(self._histograms['__combined__']['total'],maxabsr)
            self._changeLabels(err_hist)
            err_hist.Draw('E2 hist')
            if self._unblind:
                err_data.Draw('E1P same')
            logging.debug('... done')

        # Save to pdf # 
        if not os.path.exists(self._output_path):
            logging.info(f'Output path {self._output_path} did not exist, will create it')
            os.makedirs(self._output_path)
        pdfName = f'{self._bin_name}_{self._fit_type}_{opt}_unblinded_{self._unblind}_{"_".join([str(era) for era in self._eras])}.pdf'
        pdfPath = os.path.join(self._output_path,pdfName)
                                
        canvas.Print(pdfPath)
        logging.info(f'Plot saved as {pdfPath}')

        # Give ownership of the histograms to python so it can use the garbage cleaning #
        for cat in self._histograms.keys():
            for group in self._histograms[cat].keys():
                if self._histograms[cat][group] is not None:
                    ROOT.SetOwnership(self._histograms[cat][group], True)
                                         

    def _getProcesses(self,folder,cat):
        n_bins = 0
        for process,processCfg in self._processes.items():
            if not folder.GetListOfKeys().FindObject(process):
                logging.warning(f'Process `{process}` not found in folder `{folder.GetTitle()}`')
                if processCfg['group'] not in self._histograms[cat].keys():
                    self._histograms[cat][processCfg['group']] = None
                continue
            if process == 'data':
                continue # Done in the end 
            # Get non zero bins #
            h = self._getNonZeroHistogram(folder.Get(process))
            if h.GetNbinsX() > n_bins:
                n_bins = h.GetNbinsX()
            p_max_len = max([len(name) for name in self._processes.keys()]) + 3
            logging.debug(f'Looking at process {process:{p_max_len}s} - yield = {h.Integral():12.5f}')
            
            # Check the sum of weight errors #
            if not h.GetSumw2N():
                h.Sumw2()

            # In case group no there yet, copy it (to avoid disappearance after file closed) #
            if processCfg['group'] not in self._histograms[cat].keys() or self._histograms[cat][processCfg['group']] is None:
                self._histograms[cat][processCfg['group']] = copy.deepcopy(h)
            # Group already there, add it #
            else:
                N1 = self._histograms[cat][processCfg['group']].GetNbinsX()
                N2 = h.GetNbinsX()
                if N1 != N2:
                    raise RuntimeError(f"Process {process} in category {cat} has {N1} bins, while in folder {folder.GetTitle()} it has {N2}, merging will not be possible. If the problems lies in eras having different binning, maybe consider running postfits separated per era")
                # Merge #
                self._histograms[cat][processCfg['group']].Add(h)
        # Add data #
        data = self._getNonZeroGraph(copy.deepcopy(folder.Get('data')),n_bins)

        # Need to square the errors so we can add them if needed, and use the sqrt later 
        for i in range(data.GetN()):
            data.SetPointError(i,
                   data.GetErrorXlow(i),
                   data.GetErrorXhigh(i),
                   data.GetErrorYlow(i)**2,
                   data.GetErrorYhigh(i)**2)
            
        # If data is there, need to manually add it
        if 'data' in self._histograms[cat].keys():
            # Check the x coordinates of the points #
            x1 = [x for x in data.GetX()]
            x2 = [x for x in self._histograms[cat]['data'].GetX()]
            y1 = [y for y in data.GetY()]
            y2 = [y for y in self._histograms[cat]['data'].GetY()]
            if len(set(x1).intersection(set(x2))) != len(x1):
                raise RuntimeError(f"Data of category {cat} has x positons {x1} while in folder {folder.GetTitle()} it has {x2}, merging will not be possible. If the problems lies in eras having different binning, maybe consider running postfits separated per era")
            # Add manually the content #
            for i in range(data.GetN()):
                data.SetPoint(i,x1[i],y1[i]+y2[i])
                data.SetPointError(i,
                   data.GetErrorXlow(i), 
                   data.GetErrorXhigh(i), 
                   data.GetErrorYlow(i) + self._histograms[cat]['data'].GetErrorYlow(i), 
                   data.GetErrorYhigh(i) + self._histograms[cat]['data'].GetErrorYhigh(i))
                        # the errors are encoded as quadratic here 
        # Save the new data #
        self._histograms[cat]['data'] = copy.deepcopy(data)
                                       
    @staticmethod
    def _getNonZeroHistogram(h):
        """
            When several categories with different ranges are given to FitDiagnostic
            it zero padds the categories with fewer bins 
            -> we need to cut out this range of 0-content on the right of the histogram
            BUT we do not want to exclude valid bins that just happen to have 0 content
        """
        # Get content #
        content = np.zeros(h.GetNbinsX())
        error   = np.zeros(h.GetNbinsX())
        for i in range(1,h.GetNbinsX()+1):
            content[i-1] = h.GetBinContent(i)
            error[i-1]   = h.GetBinError(i)
        # Find the last bin before a continuous range of 0 until end of content
        if content[-1] == 0.:
            cut_idx = np.where(np.diff((content==0.)*1) != 0)[0][-1] + 1
            content = content[:cut_idx]
            error   = error[:cut_idx]
        h_tmp = getattr(ROOT,h.__class__.__name__)(h.GetName(),
                                                   h.GetTitle(),
                                                   len(content),
                                                   0,
                                                   len(content))
        for i,(c,e) in enumerate(zip(content,error),1):
            h_tmp.SetBinContent(i,c)
            h_tmp.SetBinError(i,e)

        return h_tmp
        
    @staticmethod
    def _getNonZeroGraph(g,N):
        x = np.array(g.GetX())[:N]
        y = np.array(g.GetY())[:N]
        # Hide zero values of the graph #
        y[np.where(y==0.)] = -100.
        g_tmp = getattr(ROOT,g.__class__.__name__)(len(y))
        for i in range(len(x)):
            g_tmp.SetPoint(i,x[i],y[i])
            g_tmp.SetPointError(i,
                   g.GetErrorXlow(i),
                   g.GetErrorXhigh(i),
                   g.GetErrorYlow(i) if y[i] > 0. else 0.,
                   g.GetErrorYhigh(i) if y[i] > 0. else 0.)
        return g_tmp
        


    @staticmethod
    def _getBinning(h):
        return [h.GetXaxis().GetBinUpEdge(i) for i in range(h.GetNbinsX()+1)]


    def _processDataGraphs(self,list_data):
        # Concatenate several tgraphs into one #
        Ns = [g.GetN() for g in list_data]
        if self._bin_edges is not None:
            if len(self._bin_edges) != len(list_data):
                raise RuntimeError(f'There are {len(list_hist)} histograms but only {len(self._bin_edges)} bin edges sets have been provided')
            widths = np.concatenate([np.diff(bin_edges) for bin_edges in self._bin_edges],axis=0)
        else:
            widths = np.ones(sum(Ns))
        edges = np.concatenate([np.array([0]),np.cumsum(widths)],axis=0)
        xvals = (edges[1:]+edges[:-1])/2
        xerror_low = xvals-edges[:-1]
        xerror_high = edges[1:]-xvals
        yvals = np.concatenate([np.array(g.GetY()) for g in list_data],axis=0)
        yerror_low  = [g.GetErrorYlow(i) for g in list_data for i in range(0,g.GetN())]
        yerror_high = [g.GetErrorYhigh(i) for g in list_data for i in range(0,g.GetN())]
        gtot = ROOT.TGraphAsymmErrors(sum(Ns))
        embed()
        for i in range(0,sum(Ns)):
            gtot.SetPoint(i,xvals[i],yvals[i])
            gtot.SetPointError(i,
                               0., #xerror_low[i],
                               0., #xerror_high[i],
                               yerror_low[i],
                               yerror_high[i])
        # Return #
        return gtot
        
    def _processBackgroundHistograms(self,list_hist):
        # Concatenate several hists into one #
        if self._bin_edges is not None:
            if len(self._bin_edges) != len(list_hist):
                raise RuntimeError(f'There are {len(list_hist)} histograms but only {len(self._bin_edges)} bin edges sets have been provided')
            Ns = [len(be)-1 for be in self._bin_edges]
            widths = np.concatenate([np.diff(bin_edges) for bin_edges in self._bin_edges],axis=0)
            edges = np.concatenate([np.array([0]),np.cumsum(widths)],axis=0)
        else:
            Ns = self._bins
            edges = np.arange(sum(Ns)+1,dtype=np.float32)
        # Get characteristics of one of the histograms #
        h_dummy = None
        for h in list_hist:
            if h is not None:
                h_dummy = h
        if h_dummy is None: # No histogram taken from fitdiag file #
            return None
        htot = getattr(ROOT,h_dummy.__class__.__name__)(
                         h_dummy.GetName()+'tot',
                         h_dummy.GetTitle()+'tot',
                         edges.shape[0]-1,
                         edges)
        # Fill the histogram #
        i = 1
        for h,N in zip(list_hist,Ns):
            for j in range(1,N+1):
                if h is None:
                    htot.SetBinContent(i,0.)
                    htot.SetBinError(i,0.)
                else:
                    htot.SetBinContent(i,h.GetBinContent(j))
                    htot.SetBinError(i,h.GetBinError(j))
                i += 1
        # Return #
        return htot
        
    def _getTotalHistError(self,total_hist,maxabsr=0.):
        # Declare hist #
        edges = np.array([total_hist.GetXaxis().GetBinLowEdge(i) 
                    for i in range(1,total_hist.GetNbinsX()+2)], dtype=np.float32)
        total_err = getattr(ROOT,total_hist.__class__.__name__)(
                    total_hist.GetName()+'err',
                    '',
                    edges.shape[0]-1,
                    edges)
        # Esthetics #
        total_err.GetYaxis().SetTitle("#frac{Data - Expectation}{Expectation}")
        total_err.GetXaxis().SetTitleOffset(1.25)
        total_err.GetYaxis().SetTitleOffset(1.0)
        total_err.GetXaxis().SetTitleSize(0.14)
        total_err.GetYaxis().SetTitleSize(0.075)
        total_err.GetYaxis().SetLabelSize(0.105)
        total_err.GetXaxis().SetLabelSize(0.10)
        total_err.GetXaxis().SetLabelColor(1)
        total_err.SetMarkerSize(0)
        total_err.SetMarkerColor(12)
        total_err.SetFillColorAlpha(12, 0.40)
        total_err.SetMarkerColorAlpha(12,0.40)
        total_err.SetLineWidth(2)
        total_err.SetLineStyle(2)

        total_hist.SetMarkerSize(0)
        total_hist.SetMarkerColorAlpha(12,0.40)
        total_hist.SetLineWidth(0)
        total_hist.SetFillColorAlpha(12, 0.40)

        # Fill #
        for i in range(1,total_hist.GetNbinsX()+1):
            total_err.SetBinContent(i,0.)
            err = total_hist.GetBinError(i)/total_hist.GetBinContent(i)
            total_err.SetBinError(i,err)
            if err > maxabsr:
                maxabsr = err

        # Custom #
        minr = - maxabsr * 1.2
        maxr = + maxabsr * 1.2
        optx = {}
        opty = {}
        if 'x-axis' in self._plot_options.keys():
            optx.update(self._plot_options['x-axis'])
        if 'y-axis' in self._plot_options.keys():
            opty.update(self._plot_options['y-axis'])
            if 'label' in opty:
                del opty['label']
        if 'ratio' in self._plot_options.keys():
            opty.update(self._plot_options['ratio'])
        self._applyCustomAxisOptions(total_err.GetXaxis(),optx)
        self._applyCustomAxisOptions(total_err.GetYaxis(),opty,minr,maxr)

        # Return #
        return total_err

    def _getDataError(self,total_hist,data):
        # Initialize #
        xs = list(data.GetX())
        ys = list(data.GetY())
        err = ROOT.TGraphAsymmErrors(data.GetN())

        # Bin content loop #
        maxabsr = 0.
        for i in range(data.GetN()):
            bin_width = total_hist.GetBinWidth(i+1)
            dividend = total_hist.GetBinContent(i+1) * bin_width
            # Set point : data - expectation / expectation = data/expectation -1 #
            if ys[i] > 0:
                if dividend > 0:
                    err.SetPoint(i,total_hist.GetBinCenter(i+1), ys[i] / dividend -1)
                else:
                    err.SetPoint(i,total_hist.GetBinCenter(i+1), -1. )
            else:
                    # Hide it #
                    err.SetPoint(i,total_hist.GetBinCenter(i+1), -100. )

            # Set point error #
            err_up = data.GetErrorYhigh(i) / dividend
            err_down = data.GetErrorYlow(i) / dividend
            err.SetPointEYlow(i,  err_down)
            err.SetPointEYhigh(i, err_up)
            #err.SetPointEXlow(i,  bin_width / 2.0)
            #err.SetPointEXhigh(i, bin_width / 2.0)
            err.SetPointEXlow(i,  0.)
            err.SetPointEXhigh(i, 0.)
            # Check max variation #
            if err_up > maxabsr:
                maxabsr = err_up
            if err_down > maxabsr:
                maxabsr = err_down
            
        # Esthetics #
        err.SetMarkerColor(1)
        err.SetMarkerStyle(20)
        err.SetMarkerSize(1.)
        err.SetLineColor(1)
        err.SetLineWidth(1)
        err.SetLineStyle(1)

        # Return #
        return err,maxabsr

    def _getTemplate(self):
        # Set bin content #
        h_tot = self._histograms['__combined__']['total']
        bins = np.array([h_tot.GetXaxis().GetBinLowEdge(i) for i in range(1,h_tot.GetNbinsX()+2)],dtype=np.float32)
        template = ROOT.TH1F('template','',bins.shape[0]-1,bins)
        # Esthetics #
        xaxis = template.GetXaxis()
        yaxis = template.GetYaxis()

        # Custom #
        maxy = h_tot.GetMaximum()
        if 'logy' in self._plot_options.keys() and self._plot_options['logy']:
            # If log, need to adapt min and max
            miny = 1e-2 
            maxy *= 1e5
        else:
            miny = 0.
            maxy *= 2.0
        optx = self._plot_options['x-axis'] if 'x-axis' in self._plot_options.keys() else {}
        opty = self._plot_options['y-axis'] if 'y-axis' in self._plot_options.keys() else {}

        self._applyCustomAxisOptions(xaxis,optx)
        self._applyCustomAxisOptions(yaxis,opty,miny,maxy)

        # Need to kill bottom part of showing ratio #
        if self._show_ratio:
            xaxis.SetLabelSize(0.)
            xaxis.SetTitleSize(0.)

        return template

    def _changeLabels(self,h):
        if self._bin_edges is not None:
            # Get current labels #
            labels = self._getLabels(h)
            # Get current bin edges #
            edges = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1,h.GetNbinsX()+2)])
            # Make new list of labels per bin #
            new_labels = []
            for idx,cat_edges in enumerate(self._bin_edges):
                if idx < len(self._bin_edges) - 1:
                    new_labels.extend(cat_edges[:-1])    
                else:
                    new_labels.extend(cat_edges)    
            
            # Change labels # 
            for i,label in enumerate(labels):
                # Find bin edge -> get the new label
                idx = np.argmin(np.abs(edges-label))
                new_label = new_labels[idx]
                if new_label==int(new_label): # format as an int if no decimal
                    new_label = int(new_label)
                h.GetXaxis().ChangeLabel(i+1,-1,-1,-1,-1,-1,str(new_label))
                

    @staticmethod
    def _getLabels(h):
        # No way to get labels when they are numeric values
        # Need to redo the method in root that optimizes it
        # See : https://root-forum.cern.ch/t/changing-only-the-displayed-labels/40584
        x1 = h.GetXaxis().GetBinLowEdge(1)
        x2 = h.GetXaxis().GetBinUpEdge(h.GetNbinsX())
        div = h.GetXaxis().GetNdivisions()
        if div > 0:
            ndiv = div%100
        else:
            raise ValueError('Not supported : GetNdivisions() = {div}')
        ndivo = ctypes.c_int(0)
        x1o = ctypes.c_double(0.)
        x2o = ctypes.c_double(0.)
        bw  = ctypes.c_double(0.)
        ROOT.THLimitsFinder.Optimize(x1,x2,ndiv,x1o,x2o,ndivo,bw,"")
    
        labels = [x1o.value]
        for i in range(ndivo.value):
            labels.append(labels[-1]+bw.value)
    
        return labels
    

    def _getSeparations(self):
        # If single category, no need for separation #
        if len(self._categories) == 1:
            return [],[]
        if self._bin_edges is not None:
            # Binning is taken from provided bin edges 
            bins = [[be-bin_edges[0] for be in bin_edges] for bin_edges in self._bin_edges] 
        else:
            # Need to extract binning from self._histograms
            bins = []
            for cat in self._categories:
                # Take first histogram of category
                h = self._histograms[cat][list(self._histograms[cat].keys())[0]]
                bins.append([h.GetXaxis().GetBinLowEdge(i) for i in range(1,h.GetNbinsX()+2)])
        # Multiples categories, need to find their x position #
        lines_xpos = [0.]
        for i in range(len(bins)):
            lines_xpos.append(bins[i][-1] + lines_xpos[-1])
        # Find ymax from total histogram #
        ymax = self._histograms['__combined__']['total'].GetMaximum()
        if 'logy' in self._plot_options.keys() and self._plot_options['logy']:
            ymax *= 10
        else:
            ymax *= 1.2
        # Make lines objects #
        lines = []
        for line_xpos in lines_xpos[1:-1]:
            line = ROOT.TLine(line_xpos,0.0,line_xpos,ymax)
            line.SetLineColor(1)
            line.SetLineStyle(1) 
            line.SetLineWidth(2) 
            lines.append(line)
        # Add labels if requested #
        labels = []
        if self._labels is not None:
            assert isinstance(self._labels,list)
            if len(self._labels) != len(self._categories):
                raise RuntimeError(f'You want to use {len(self._labels)} labels but defined {len(self._categories)} categories')
            # Check if positions are wanted #
            if self._label_positions is not None:
                if len(self._label_positions) != len(self._categories):
                    raise RuntimeError(f'You want to use {len(self._label_positions)} label positions but defined {len(self._categories)} categories')
                label_pos = self._label_positions
            # if not, make educated guesses #
            else:
                label_pos = []
                for i in range(1,len(lines_xpos)):
                    xl = lines_xpos[i-1]
                    xr = lines_xpos[i]
                    label_pos.append(xl + 0.2 * (xr-xl))
            # Make labels #
            for i,labelName in enumerate(self._labels):
                # Label intiialization #
                label = ROOT.TPaveText(label_pos[i],ymax,label_pos[i],ymax)
                label.SetBorderSize(0)
                label.SetFillStyle(4000)
                # Add text #
                text = label.AddText(labelName)
                text.SetTextFont(50)
                text.SetTextAlign(12)
                text.SetTextSize(0.05)
                text.SetTextColor(1)
                # Custom #
                if 'labels' in self._plot_options.keys():
                    opt = self._plot_options['labels']
                    if 'textsize' in opt:
                        text.SetTextSize(opt['textsize'])
                    if 'textfont' in opt:
                        text.SetTextFont(opt['textfont'])
                labels.append(label)

        return lines,labels

    def _applyCustomAxisOptions(self,axis,axis_opt,axis_min=None,axis_max=None):
        # label title options #
        if 'label' in axis_opt.keys():
            axis.SetTitle(axis_opt['label'])
        if 'textsize' in axis_opt.keys():
            axis.SetTitleSize(axis_opt['textsize'])
        if 'textfont' in axis_opt.keys():
            axis.SetTitleFont(axis_opt['textfont'])
        if 'offset' in axis_opt.keys():
            axis.SetTitleOffset(axis_opt['offset'])
        # label ticks options #
        if 'labelsize' in axis_opt.keys():
            axis.SetLabelSize(axis_opt['labelsize'])
        if 'labelfont' in axis_opt.keys():
            axis.SetLabelFont(axis_opt['labelfont'])
        if 'divisions' in axis_opt.keys():
            axis.SetNdivisions(axis_opt['divisions'])
        # Overriding min and max #
        if 'min' in axis_opt.keys():
            axis_min = axis_opt['min']
        if 'max' in axis_opt.keys():
            axis_max = axis_opt['max']
        if axis_max is not None and axis_min is not None:
            axis.SetRangeUser(axis_min,axis_max)
        # Generic #
        axis.SetTickLength(0.04)
        axis.SetTickSize(0.055)

    def _getBackgroundStack(self):
        stack_MC = ROOT.THStack()
        integrals = {group:hist.Integral() for group,hist in self._histograms['__combined__'].items()} 
        if self._sort_by_yield:
            self._order = [tup[0] for tup in sorted(integrals.items(), key=lambda item: item[1],reverse=True)]
            # Order = from largest to smallest integral
        else:
            self._order =  list(self._options.keys())
        for group in reversed(self._order): # Need to ass small first
            # Get options #
            optCfg = self._options[group]
            # Add to stack if background #
            if optCfg['type'] == 'mc' and group != 'total':
                stack_MC.Add(self._histograms['__combined__'][group])

        optx = self._plot_options['x-axis'] if 'x-axis' in self._plot_options.keys() else {}
        opty = self._plot_options['y-axis'] if 'y-axis' in self._plot_options.keys() else {}

        self._applyCustomAxisOptions(stack_MC.GetStack().First().GetXaxis(),optx)
        self._applyCustomAxisOptions(stack_MC.GetStack().First().GetYaxis(),opty)

        return stack_MC

    def _getLegend(self):
        if "splitline" in self._header_legend:
            bottom_legend = 0.52
        else :
            bottom_legend = 0.64

        # Instantiate legend #
        if 'legend' in self._plot_options.keys() and 'position' in self._plot_options['legend'].keys():
            assert isinstance(self._plot_options['legend']['position'],list)
            assert len(self._plot_options['legend']['position']) == 4
            legend = ROOT.TLegend(*self._plot_options['legend']['position'])
        else:
            legend = ROOT.TLegend(0.2,bottom_legend,0.9450, 0.90)

        # Esthetics #
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)
        legend.SetFillColor(10)
        legend.SetTextSize(0.040 if self._show_ratio else 0.03)
        legend.SetHeader(self._header_legend)
        header = legend.GetListOfPrimitives().First()
        header.SetTextSize(0.05 if self._show_ratio else 0.04)
        header.SetTextColor(1)
        header.SetTextFont(62)

        # Possible custom choices #
        if 'legend' in self._plot_options.keys():
            if 'columns' in self._plot_options['legend'].keys():
                legend.SetNColumns(self._plot_options['legend']['columns'])
            if 'textsize' in self._plot_options['legend'].keys():
                legend.SetTextSize(self._plot_options['legend']['textsize'])
            if 'textfont' in self._plot_options['legend'].keys():
                legend.SetTextFont(self._plot_options['legend']['textfont'])
            if 'headersize' in self._plot_options['legend'].keys():
                header.SetTextSize(self._plot_options['legend']['headersize'])
            if 'headerfont' in self._plot_options['legend'].keys():
                header.SetTextFont(self._plot_options['legend']['headerfont'])

        return legend

    def _getCanvas(self):
        WW = 900
        HH = 800
        TT = 0.08 * HH
        BB = 0.10 * HH
        RR = 0.04 * WW
        if self._show_ratio:
            LL = 0.12 * WW
            canvas = ROOT.TCanvas("canvas", "canvas", WW, HH)
            canvas.SetBorderMode(0)
            canvas.SetLeftMargin(LL / WW)
            canvas.SetRightMargin(RR / WW)
            canvas.SetTopMargin(TT / HH)
            canvas.SetBottomMargin(BB / HH)
            canvas.SetTickx(0)
            canvas.SetTicky(0)
        else:
            LL = 0.13 * WW
            canvas = ROOT.TCanvas("canvas", "canvas", WW, WW)
            canvas.SetBorderMode(0)
            canvas.SetLeftMargin(LL / WW)
            canvas.SetRightMargin(RR / WW)
            canvas.SetTopMargin(TT / HH)
            canvas.SetBottomMargin(TT / HH)
            canvas.SetTickx(0)
        canvas.SetFillColor(0)
        canvas.SetFrameFillStyle(0)
        canvas.SetFrameBorderMode(0)

        return canvas

    def _getTopPad(self):
        # Based on ratio present or not #
        if self._show_ratio:
            topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.34, 1.00, 0.995)
            topPad.SetBottomMargin(0.053)
        else:
            topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.0, 1.00, 0.995)
            topPad.SetRightMargin(0.04)
            topPad.SetBottomMargin(0.1)
        # Generic #
        topPad.SetFillColor(10)
        topPad.SetTopMargin(0.075)
        topPad.SetLeftMargin(0.15)
        topPad.SetRightMargin(0.04)

        return topPad

    def _getBottomPad(self):
        bottomPad = ROOT.TPad("bottomPad", "bottomPad", 0.00, 0.05, 1.00, 0.34)
        bottomPad.SetFillColor(10)
        bottomPad.SetTopMargin(0.03)
        bottomPad.SetLeftMargin(0.15)
        bottomPad.SetBottomMargin(0.25)
        bottomPad.SetRightMargin(0.04)
        
        return bottomPad


    def _getCMSLabels(self,lumi):
        x0 = 0.22
        y0 = 0.953 if self._show_ratio else 0.935
        ypreliminary = 0.95 if self._show_ratio else 0.935
        xpreliminary = 0.08 if self._show_ratio else 0.085
        ylumi = 0.95 if self._show_ratio else 0.965
        xlumi = 0.78 if self._show_ratio else 0.73
        title_size_CMS = 0.0575 if self._show_ratio else 0.04
        title_size_Preliminary = 0.048 if self._show_ratio else 0.03
        title_size_lumi = 0.045 if self._show_ratio else 0.03
        label_cms = ROOT.TPaveText(x0, y0, x0 + 0.0950, y0 + 0.0600, "NDC")
        label_cms.AddText("CMS")
        label_cms.SetTextFont(61)
        label_cms.SetTextAlign(13)
        label_cms.SetTextSize(title_size_CMS)
        label_cms.SetTextColor(1)
        label_cms.SetFillStyle(0)
        label_cms.SetBorderSize(0)
        label_preliminary = ROOT.TPaveText(
            x0 + xpreliminary, y0 - 0.005, x0 + 0.0980 + 0.12, y0 + 0.0600 - 0.005, "NDC"
        )
        label_preliminary.AddText("Preliminary")
        label_preliminary.SetTextFont(50)
        label_preliminary.SetTextAlign(13)
        label_preliminary.SetTextSize(title_size_Preliminary)
        label_preliminary.SetTextColor(1)
        label_preliminary.SetFillStyle(0)
        label_preliminary.SetBorderSize(0)
        label_luminosity = ROOT.TPaveText(xlumi, y0 + 0.0035, xlumi + 0.0900, y0 + 0.040, "NDC")
        label_luminosity.AddText(lumi)
        label_luminosity.SetTextFont(42)
        label_luminosity.SetTextAlign(13)
        label_luminosity.SetTextSize(title_size_lumi)
        label_luminosity.SetTextColor(1)
        label_luminosity.SetFillStyle(0)
        label_luminosity.SetBorderSize(0)

        return [label_cms, label_preliminary, label_luminosity]


