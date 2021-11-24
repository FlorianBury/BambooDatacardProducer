import os
import sys
import copy
import math
import ROOT
import logging
import array
from IPython import embed

from context import TFileOpen

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

class PostfitPlots:
    def __init__(self,bin_name,output_path,fit_diagnostics_path,processes,fit_type,header,analysis,eras,categories,labels=None,label_positions=None,plot_options={},unblind=False,show_ratio=False,sort_by_yield=True,verbose=False):
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
            - labels [list(str)]            : list of categories labels to be put on top of each category (if several) [OPTIONAL]
            - label_positions [list(float)] : x positions of the labels [OPTIONAL], if not provided and labels is, will find some "smart" positions
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
                           'min': <min x-axis>,
                           'max': <max x-axis>,
                },
                'y-axis': {'textsize': <textsize>,
                           'textfont': <textfont>, 
                           'labelsize': <labelsize>,
                           'offset': <offset>,
                           'label': <y-axis label>,
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
        lumis = {'2016':36.330,'2017':41.530,'2018':59.740}
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
        lumi += ' fb^{-1} (13 TeV)'


        # Open root file and get folder subdir # 
        if not os.path.exists(self._fit_diagnostics_path):
            raise RuntimeError(f'`{self._fit_diagnostics_path}` does not exist')

        self._histograms = {}
        self._binning = {} 
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
                    
        # Declare legend #
        legend = self._getLegend()

        # Process histograms per group #
        logging.debug('Processing histograms')
        self._histograms['__combined__'] = {}
        embed()
        for group,optCfg in self._options.items():
            list_hist = [self._histograms[cat][group] for cat in categories]
            if group == 'data':
                # Get combined data #
                self._histograms['__combined__']['data'] = self._processDataGraphs(list_hist)
                # Add to legend #
                # Esthetics #
                self._histograms['__combined__']['data'].SetMarkerColor(1)
                self._histograms['__combined__']['data'].SetMarkerStyle(20)
                self._histograms['__combined__']['data'].SetMarkerSize(0.8)
                self._histograms['__combined__']['data'].SetLineColor(1)
                self._histograms['__combined__']['data'].SetLineWidth(1)
                self._histograms['__combined__']['data'].SetLineStyle(1)
            elif optCfg['type'] in ['mc','signal']:
                # Get combined hist #
                self._histograms['__combined__'][group]  = self._processBackgroundHistograms(list_hist)
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
        template.Draw('')
        # Draw stack #
        stack_MC.Draw('hist same')
        stack_MC.Draw('hist same axis') # otherwise erases the axes
        # Draw uncertainty #
        self._histograms['__combined__']['total'].Draw('e2 same')
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
            self._histograms['__combined__']['data'].Draw('e1p same')
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
            err_hist = self._getTotalHistError(self._histograms['__combined__']['total'])
            err_hist.Draw('e2 hist')
            if self._unblind:
                err_data = self._getDataError(self._histograms['__combined__']['total'],
                                              self._histograms['__combined__']['data'])
                err_data.Draw('e1P same')
            logging.debug('... done')

        # Save to pdf # 
        if not os.path.exists(self._output_path):
            logging.info(f'Output path {self._output_path} did not exist, will create it')
            os.makedirs(self._output_path)
        pdfName = f'{self._bin_name}_{self._fit_type}_{opt}_unblinded_{self._unblind}_{"_".join([str(era) for era in self._eras])}.pdf'
        pdfPath = os.path.join(self._output_path,pdfName)
                                
        canvas.Print(pdfPath)
        logging.info(f'Plot saved as {pdfPath}')
                                         

    def _getProcesses(self,folder,cat):
        for process,processCfg in self._processes.items():
            if not folder.GetListOfKeys().FindObject(process):
                logging.debug(f'Process `{process}` not found in folder `{folder.GetTitle()}`')
                continue
            if process == 'data':
                continue # Done in the end 
            # Get non zero bins #
            h = self._getNonZeroHistogram(folder.Get(process))
            p_max_len = max([len(name) for name in self._processes.keys()]) + 3
            logging.debug(f'Looking at process {process:{p_max_len}s} - yield = {h.Integral():12.5f}')
            
            # Check the sum of weight errors #
            if not h.GetSumw2N():
                h.Sumw2()

            # In case group no there yet, copy it (to avoid disappearance after file closed #
            if processCfg['group'] not in self._histograms[cat].keys():
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
        data = self._getNonZeroGraph(copy.deepcopy(folder.Get('data')))
        # Get binning from data graph #
        xs = list(data.GetX())
        bins = [xs[0]-data.GetErrorXlow(0)]
        for i in range(data.GetN()):
            bins.append(xs[i]+data.GetErrorXhigh(i))
            if i > 0:
                assert xs[i-1]+data.GetErrorXhigh(i) == xs[i]-data.GetErrorXlow(i)
        self._binning[cat] = bins

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
        content = []
        error   = []
        for i in range(1,h.GetNbinsX()+1):
            if h.GetBinContent(i) > 0 and h.GetBinError(i)>0:
                content.append(h.GetBinContent(i))
                error.append(h.GetBinError(i))
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
    def _getNonZeroGraph(g):
        x = list(g.GetX())
        y = list(g.GetY())
        g_tmp = getattr(ROOT,g.__class__.__name__)(sum([_y>0 for _y in y]))
        j = 0
        for i in range(len(x)):
            if y[i] > 0:
                g_tmp.SetPoint(j,x[i],y[i])
                g_tmp.SetPointError(j,
                       g.GetErrorXlow(i),
                       g.GetErrorXhigh(i),
                       g.GetErrorYlow(i),
                       g.GetErrorYhigh(i))
                j += 1
        return g_tmp
        


    @staticmethod
    def _getBinning(h):
        return [h.GetXaxis().GetBinUpEdge(i) for i in range(h.GetNbinsX()+1)]


    def _processDataGraphs(self,list_data):
        # Concatenate several tgraphs into one #
        Ns = [g.GetN() for g in list_data]
        xvals = [list(g.GetX()) for g in list_data]
        yvals = [list(g.GetY()) for g in list_data]
        gtot = ROOT.TGraphAsymmErrors(sum(Ns))
        i = 0
        for xs,ys,g in zip(xvals,yvals,list_data):
            j = 0
            for x,y in zip(xs,ys):
                gtot.SetPoint(i,i,y)
                gtot.SetPointError(i,
                       g.GetErrorXlow(j),
                       g.GetErrorXhigh(j),
                       g.GetErrorYlow(j),
                       g.GetErrorYhigh(j))
                j += 1
                i += 1
        # Return #
        return gtot
        
    def _processBackgroundHistograms(self,list_hist):
        # Concatenate several hists into one #
        Ns = [h.GetNbinsX() for h in list_hist]
        htot = getattr(ROOT,list_hist[0].__class__.__name__)(
                    list_hist[0].GetName()+'tot',
                    list_hist[0].GetTitle()+'tot',
                    sum(Ns),
                    0.,
                    sum(Ns))
        indices = [list(range(1,h.GetNbinsX()+1)) for h in list_hist]
        j = 1
        for ind,h in zip(indices,list_hist):
            for i in ind:
                htot.SetBinContent(j,h.GetBinContent(i))
                htot.SetBinError(j,h.GetBinError(i))
                j += 1
        # Return #
        return htot
        
    def _getTotalHistError(self,total_hist):
        # Declare hist #
        total_err = getattr(ROOT,total_hist.__class__.__name__)(
                    total_hist.GetName()+'err',
                    '',
                    total_hist.GetNbinsX(),
                    total_hist.GetXaxis().GetBinLowEdge(1),
                    total_hist.GetXaxis().GetBinUpEdge(total_hist.GetNbinsX()))
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
        total_err.SetFillColorAlpha(12, 0.40)
        total_err.SetLineWidth(2)
        total_err.SetLineStyle(2)

        total_hist.SetMarkerSize(0)
        total_hist.SetLineWidth(0)
        total_hist.SetFillColorAlpha(12, 0.40)

        # Fill #
        maxabsr = 0.
        for i in range(1,total_hist.GetNbinsX()+1):
            total_err.SetBinContent(i,0.)
            err = total_hist.GetBinError(i)/total_hist.GetBinContent(i)
            total_err.SetBinError(i,err)
            if err > maxabsr:
                maxabsr = err

        # Custom #
        minr = - maxabsr * 1.2
        maxr = + maxabsr * 1.2
        if 'ratio' in self._plot_options.keys():
            opt = self._plot_options['ratio']
            self._applyCustomAxisOptions(total_err.GetYaxis(),opt,minr,maxr)
        if 'x-axis' in self._plot_options.keys():
            self._applyCustomAxisOptions(total_err.GetXaxis(),self._plot_options['x-axis'])

        # Return #
        return total_err

    def _getDataError(self,total_hist,data):
        # Initialize #
        xs = list(data.GetX())
        ys = list(data.GetY())
        err = ROOT.TGraphAsymmErrors(data.GetN())

        # Bin content loop #
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
                    err.SetPoint(i,total_hist.GetBinCenter(i+1), -100. )

            # Set point error #
            err.SetPointEYlow(i,  data.GetErrorYlow(i) / dividend)
            err.SetPointEYhigh(i, data.GetErrorYhigh(i) / dividend)
            err.SetPointEXlow(i,  bin_width / 2.0)
            err.SetPointEXhigh(i, bin_width / 2.0)
            
        # Esthetics #
        err.SetMarkerColor(1)
        err.SetMarkerStyle(20)
        err.SetMarkerSize(0.8)
        err.SetLineColor(1)
        err.SetLineWidth(1)
        err.SetLineStyle(1)

        # Return #
        return err

    def _getTemplate(self):
        # Set bin content #
        bins = list(self._binning.values())
        template_bins = bins[0][:]
        for i in range(1,len(bins)):
            template_bins += [b+template_bins[-1] for b in bins[i]]
        template = ROOT.TH1F('template','',len(template_bins)-1,array.array('d',template_bins))
        # Esthetics #
        xaxis = template.GetXaxis()
        yaxis = template.GetYaxis()

        # Custom #
        maxy = self._histograms['__combined__']['total'].GetMaximum()
        if 'logy' in self._plot_options.keys() and self._plot_options['logy']:
            # If log, need to adapt min and max
            miny = 1e-2 
            maxy *= 1e6
        else:
            miny = 0.
            maxy *= 2.0
        xaxis_opt = self._plot_options['x-axis'] if 'x-axis' in self._plot_options.keys() else {}
        yaxis_opt = self._plot_options['y-axis'] if 'y-axis' in self._plot_options.keys() else {}

        self._applyCustomAxisOptions(xaxis,xaxis_opt)
        self._applyCustomAxisOptions(yaxis,yaxis_opt,miny,maxy)

        # Need to kill bottom part of showing ratio #
        if self._show_ratio:
            xaxis.SetLabelSize(0.)
            xaxis.SetTitleSize(0.)

        return template

    def _getSeparations(self):
        # If single category, no need for separation #
        if len(self._binning) == 1:
            return [],[]
        bins = list(self._binning.values())
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




if __name__ == '__main__':
    plot_options = {
        'legend': {
            'columns': 3,
            'textsize': 0.05,
            'textfont': 42,
            'headersize': 0.07,
            'headerfont': 62,
         },
        'x-axis': {'textsize': 0.1,
                   'labelsize': 0.1,
                   'offset': 1.2,
                   'label': 'DNN score bin #',
        },
        'y-axis': {'textsize': 0.06,
                   'labelsize': 0.05,
                   'offset': 1.2,
                   'label': 'Events',
                   #'min': 1e-3,
                   #'max': 1e11,
        },
        'logy': True,
        'logx': False,
        'ratio':{#'min': -0.60,
                 #'max': 0.60,
                 'offset': 0.8,
                 'titlesize': 0.08,
                 'labelsize': 0.1,
        },
        'labels':{
            'textsize': 0.06,
        }
    }



                
                
    PostfitPlots(bin_name   = 'other',
                 output_path = 'test',
                 fit_diagnostics_path = '/nfs/scratch/fynu/fbury/Datacards/bbww_dl/Resonant/datacard_fit_Resonant_HighMass_Graviton_2D_syst_M_650_FR2/prefit/fitDiagnosticsHHbbWW.root',
                 processes = {'TT':{'group':'TT','type':'mc','color':'#992233','label':'TT'},
                              'Other_bbWW':{'group':'other','type':'mc','color':5,'label':'Other'},
                              'DY':{'group':'DY','type':'mc','color':4,'label':'DY'},
                              'ST':{'group':'ST','type':'mc','color':3,'label':'ST'},
                              'Fakes':{'group':'Fakes','type':'mc','color':6,'label':'Fakes'},
                              'signal_ggf_spin2_650_hbbhwwdl':{'group':'HH','type':'signal','color':4,'label':'signal'},
                              },
                 fit_type = 'prefit',
                 header   = 'test',
                 analysis = 'HH',
                 eras     = ['2016'],
                 categories = ['HH_DL_650_resolved_other','HH_DL_650_boosted_other','HH_DL_650_inclusive_DY_VVV'],
                 plot_options = plot_options,
                 labels   = ['resolved','boosted','DY'],
                 label_positions = None,
                 unblind= False,
                 show_ratio = True,
                 verbose = True,
                 sort_by_yield=True)


    
