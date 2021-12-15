import os
import sys
import csv
import json
import yaml
import copy
import shutil
import argparse
import subprocess
import ROOT
import numpy as np
import multiprocess as mp
from IPython import embed

from yamlLoader import YMLIncludeLoader

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(55)
ROOT.gErrorIgnoreLevel = 2001


def runCommand(command,combine_args={},subdirs={},plotIt=False,debug=False):
    # Modify command #
    command = command.split()
    if '--combine' in command:
        raise RuntimeError('Please do not use --combine in your commands')
    if '-u' not in command:
        command.insert(1,'-u')
    if combine_args is not None and isinstance(combine_args,dict) and len(combine_args) > 0:
        combine_cmd = ['--combine']
        for combine_arg in combine_args.keys():
            if combine_arg in subdirs.keys():
                combine_cmd.append(combine_arg)
        if len(combine_cmd) > 1:
            command.extend(combine_cmd)
    if debug:
        command.append('--debug')
    if plotIt:
        command += ['--plotIt']


    # Run command #
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    # Browse output #
    output_path = None
    while True:
        nextline = process.stdout.readline().strip()
        if "Output path" in nextline:
            output_path = nextline.split(' ')[-1]
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline+'\n')
        sys.stdout.flush()
    process.communicate()

    # Finalize #
    results = {}
    if output_path is None:
        print ('Could not find output path from command')
    elif not os.path.isdir(output_path):
        print (f'Output path {output_path} not valid')
    else:
        # Get results based on subdirs #
        for combineMode,subdirNames in subdirs.items(): 
            if len(subdirNames) == 0:
                subdirNames = ['']
            if combineMode not in combine_args.keys():
                continue
            combineType = combine_args[combineMode]
            results[combineMode] = {}
            for subdirName in subdirNames:
                json_file = os.path.join(output_path,combineMode,subdirName,f'{combineType}.json')
                subdirName = os.path.basename(subdirName)
                if os.path.exists(json_file):
                    with open(json_file,'r') as handle:
                        result = json.load(handle)
                    results[combineMode][subdirName] = result
                else:
                    print (f'Combine type {combineMode} file {json_file} does not exist')
                    results[combineMode][subdirName] = None
    return results
            
            

class Scan:
    def __init__(self,combine_args,plotIt,jobs,force,no_save,unblind,debug,**kwargs):
        self.curves         = kwargs['curves']
        self.combine        = kwargs['combine']
        self.outputDir      = kwargs['outputDir']
        self.plots          = kwargs['plots'] if 'plots' in kwargs.keys() else None
        self.combine_args   = combine_args
        self.plotIt         = plotIt
        self.jobs           = jobs
        self.force          = force
        self.no_save        = no_save
        self.unblind        = unblind
        self.debug          = debug

        # Make directory #
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)

        path_save = os.path.join(self.outputDir,'graphs.root')
        is_save = os.path.exists(path_save) and not self.force

        if self.combine_args is None:
            self.combine_args = []

        if not is_save:
            if self.force:
                print ('Root save found, but will run anyway')
            else:
                print ('Root save not found, will produce it')
            self.runTasks()
        else:
            print ('Root save found, will load it')
            self.loadSave(path_save)


        # Producing graphs #
        if not is_save:
            print ('Producing graphs')
            self.produceGraphs()
            print ('... done')
            if not self.no_save:
                print (f'Creating save file {path_save}')
                self.produceSave(path_save)
                print ('... done')

        # PLOTS #
        if self.plots:
            for plotCfg in self.plots:
                lines = []
                legends = []
                graphs = []
                attributes = []
                title = ""
                # Name #
                if 'name' not in plotCfg.keys():
                    raise RuntimeError('Missing `name`')
                name = plotCfg['name']
                # Check if single limit or not #
                single_limit = False
                # Find the graphs #
                for line,names in plotCfg['curves'].items():
                    assert len(names) == 3
                    if names[0] not in self.curves.keys() or not names[1] in self.curves[names[0]]['graphs'].keys():
                        print (f'Curve {names} not found in save')
                        continue
                    lines.append(line)
                    curveName,combineMode,subname = names
                    combineType = self.combine[combineMode]
                    if len(plotCfg['curves']) == 1 and combineType == 'limits':
                        single_limit = True
                        # Single limit plot #
                        graphs = self.curves[curveName]['graphs'][combineMode][subname]
                        if 'labelConv' in self.curves[curveName].keys():
                            labelConv = self.curves[curveName]['labelConv'][combineMode][subname]
                        else:
                            labelConv = None
                    else:
                        # Multi graphs plots #
                        if len(self.curves[curveName]['graphs'][combineMode]) > 0:
                            if combineType == 'limits':
                                graph = self.curves[curveName]['graphs'][combineMode][subname][1]
                            if combineType == 'gof':
                                graph = self.curves[curveName]['graphs'][combineMode][subname][0]
                        else:
                            graph = None
                        graphs.append(graph)
                if len(graphs) == 0:
                    continue
                # Check for attributes #
                if 'options' in plotCfg.keys():
                    options = plotCfg['options']
                # Get attributes #
                if 'lines' in plotCfg.keys():
                    for line in lines:
                        if line in plotCfg['lines'].keys():
                            attributes.append(plotCfg['lines'][line])
                        else:
                            attributes.append({})
                # Get legend entries #
                if 'legend' in plotCfg.keys():
                    for line in lines:
                        legends.append(plotCfg['legend'][line]) 
                # Title #
                if 'title' in plotCfg.keys():
                    title = plotCfg['title']
                # Run the plot #
                if single_limit:
                    self.plotLimits(graphs      = graphs,
                                    suffix      = name,
                                    options     = options,
                                    labelConv   = labelConv)
                else:
                    self.plotMultipleGraphs(graphs      = graphs,
                                            legends     = legends,
                                            options     = options,
                                            attributes  = attributes,
                                            title       = title,
                                            name        = name)
        # Save as CSV #
        for curveName in self.curves.keys():
            graphs = self.curves[curveName]
            for combineMode in self.curves[curveName]['graphs'].keys():
                combineType = self.combine[combineMode]
                for subdir in self.curves[curveName]['graphs'][combineMode].keys():
                    graphs = self.curves[curveName]['graphs'][combineMode][subdir]
                    csv_path = os.path.join(self.outputDir,f'{curveName}_{combineMode}_{subdir}.csv').replace(' ','_')
                    if combineType == 'limits':
                        headers = ['Data','Central','1 sigma','2 sigma']
                    if combineType == 'gof':
                        headers = ['Gof']
                    self.saveCSV(csv_path,graphs,headers)
                    
    def loadSave(self,path_save):
        F = ROOT.TFile(path_save,'READ')
        for curveName in self.curves.keys():
            # Get curve directory #
            if not hasattr(F,curveName):
                raise RuntimeError(f'Curve name `{curveName}` not found in {path_save}')
            curveDir = getattr(F,curveName)
            self.curves[curveName]['graphs'] = {}
            for kCurve in curveDir.GetListOfKeys():
                if kCurve.GetClassName() != "TDirectoryFile":
                    continue
                combineMode = kCurve.GetName()
                # Check if curve has the mode in the config #
                if not combineMode in self.combine.keys():
                    continue
                if combineMode not in self.combine_args:
                    self.combine_args.append(combineMode)
                if not hasattr(curveDir,combineMode):
                    print (f'Curve name `{curveName}` with combine type `{combineMode}` not found in {path_save}')
                    continue
                combineModeDir = getattr(curveDir,combineMode)
                self.curves[curveName]['graphs'][combineMode] = {}
                for key in combineModeDir.GetListOfKeys():
                    # Get subname directory #
                    if not hasattr(combineModeDir,key.GetName()):
                        raise RuntimeError(f'Curve name `{curveName}` combine type {combineMode} entry {key.GetName()} not found in {path_save}')
                    if key.GetClassName() == "TDirectoryFile":
                        subnameDir = getattr(combineModeDir,key.GetName())
                        self.curves[curveName]['graphs'][combineMode][key.GetName()] = []
                        for g in subnameDir.GetListOfKeys():
                            self.curves[curveName]['graphs'][combineMode][key.GetName()].append(subnameDir.Get(g.GetName()))
                    elif key.GetClassName() == "TGraph":
                        if '' not in self.curves[curveName]['graphs'][combineMode].keys():
                            self.curves[curveName]['graphs'][combineMode][''] = [combineModeDir.Get(key.GetName())]
                        else:
                            self.curves[curveName]['graphs'][combineMode][''].append(combineModeDir.Get(key.GetName()))
                    else:
                        raise RuntimeError(f'Curve name `{curveName}` combine type {combineMode} entry {key.GetName()} type {key.GetClassName()} not understood in {path_save}')


    def produceSave(self,path_save):
        F = ROOT.TFile(path_save,'RECREATE')
        for curveName in self.curves.keys():
            curveDir = F.mkdir(curveName,curveName)
            curveDir.cd()
            for combineMode in self.combine_args:
                if combineMode in self.curves[curveName]['graphs'].keys():
                    combineModeDir = curveDir.mkdir(combineMode,combineMode)
                    combineModeDir.cd()
                    for subname in self.curves[curveName]['graphs'][combineMode]:
                        if len(subname) == 0:
                            subnameDir = combineModeDir
                        else:
                            subnameDir = combineModeDir.mkdir(subname,subname)
                        subnameDir.cd()
                        if self.curves[curveName]['graphs'][combineMode][subname] is not None:
                            graphs = self.curves[curveName]['graphs'][combineMode][subname]
                            if not isinstance(graphs,tuple) and not isinstance(graphs,list):
                                graphs = [graphs]
                            for graph in graphs:
                                graph.Write(graph.GetName(),ROOT.TObject.kOverwrite)
                    combineModeDir.cd()
        F.Close()
 
    def runTasks(self):
        # Create list of args #
        pool_cmds = []
        for curveName,values in self.curves.items():
            subdirs = {}
            combine_args = {}
            for combineMode in self.combine_args:
                if combineMode not in values['modes']:
                    print(f'Key {combineMode} missing in curve {curveName} : will not run it')
                else:
                    subdirs[combineMode] = values['modes'][combineMode]
                    combine_args[combineMode] = self.combine[combineMode]

            mainSubdDirs = copy.deepcopy(subdirs)
            for ipoint,point in enumerate(values['points']):
                if not 'command' in point.keys():
                    continue
                if 'custom' in point.keys():
                    subdirs = copy.deepcopy(mainSubdDirs)
                    for subkey, subvalues in subdirs.items():
                        subdirs[subkey] = [subval.format(**point['custom']) for subval in subvalues]

                pool_cmds.append((point['command'],     # command 
                                  combine_args,         # combine args 
                                  subdirs,              # subdirs
                                  self.plotIt,          # plotIt
                                  self.debug))          # debug

        # Run the pool #
        with mp.Pool(processes=min(mp.cpu_count(),self.jobs)) as pool:
            results = pool.starmap(runCommand, pool_cmds)

        # Add values to curves dict #
        idx = 0
        for icurve,curveName in enumerate(self.curves.keys()):
            for ipoint in range(len(self.curves[curveName]['points'])):
                result = None
                custom = None
                scale  = 1.
                if 'values' in self.curves[curveName]['points'][ipoint].keys():
                    combineMode = list(self.curves[curveName]['modes'])[0]
                    subname = ''
                    result = {combineMode:{subname:self.curves[curveName]['points'][ipoint]['values']}}
                if 'command' in self.curves[curveName]['points'][ipoint].keys():
                    result = results[idx]
                    # If custom defined, save it to reverse #
                    if 'custom' in self.curves[curveName]['points'][ipoint].keys():
                        custom = self.curves[curveName]['points'][ipoint]['custom']
                if 'scale' in self.curves[curveName].keys():
                    scale = self.curves[curveName]['scale']
                if result is not None:
                    presult = {}  # processed result
                    # Loop through results dict # 
                    for combineMode in result.keys():
                        combineType = self.combine[combineMode]
                        presult[combineMode] = {}
                        for subname in result[combineMode].keys():
                            if result[combineMode][subname] is None:
                                continue
                            subres = result[combineMode][subname]
                            # Reverse the custom #
                            if custom is not None:
                                for key,val in custom.items():
                                    subname = subname.replace(str(val),f'{{{key}}}')
                            # Format the results #
                            if combineType == 'limits':
                                # Use floats for both keys and values #
                                presult[combineMode][subname] = {float(k):float(v)*scale for k,v in subres.items()}
                            if combineType == 'gof':
                                # From toys and data, produce p value #
                                if isinstance(subres['data'],list):
                                    for item in subres['data']:
                                        name = item['name']
                                        if custom is not None:
                                            for key,val in custom.items():
                                                name = name.replace(str(val),f'{{{key}}}')
                                        presult[combineMode][name] = self.produceGofValue(item)
                                else:
                                    presult[combineMode][subname] = self.produceGofValue(subres)
                    # Add to the curves content #
                    self.curves[curveName]['points'][ipoint]['results'] = presult
                    # Iterate through results #
                    idx += 1
                else:
                    raise ValueError(f'Curve name {curveName} at point {ipoint}, no result produced : is there a `values` or command `entry` ?')

    def produceGraphs(self):
        for curveName in self.curves.keys():
            self.curves[curveName]['graphs'] = {}
            self.curves[curveName]['labelConv'] = {}
            resultsPerPoint = {}
            for point in self.curves[curveName]['points']:
                for combineMode, results in point['results'].items():
                    if combineMode not in resultsPerPoint.keys():
                        self.curves[curveName]['graphs'][combineMode] = {}
                        self.curves[curveName]['labelConv'][combineMode] = {}
                        resultsPerPoint[combineMode] = {}
                    for resultName,resultVals in results.items():
                        if resultName not in resultsPerPoint[combineMode].keys():
                            resultsPerPoint[combineMode][resultName] = {}
                        resultsPerPoint[combineMode][resultName][point['label']] = resultVals
                            
            for combineMode in resultsPerPoint.keys():
                combineType = self.combine[combineMode]
                if combineType == 'limits':
                    for resultName,resultVals in resultsPerPoint[combineMode].items():
                        graphs,labelConv = self.produceLimitGraphs(resultsPerPoint[combineMode][resultName]) 
                        self.curves[curveName]['graphs'][combineMode][resultName] = graphs
                        self.curves[curveName]['labelConv'][combineMode][resultName] = labelConv
                elif combineType == 'gof':
                    for resultName,resultVals in resultsPerPoint[combineMode].items():
                        self.curves[curveName]['graphs'][combineMode][resultName] = [self.produceGofGraph(resultsPerPoint[combineMode][resultName])]
                elif combineType == 'pulls_impacts':
                    pass
                else:
                    raise RuntimeError(f'Combine type {combineType} from mode {combineMode} not understood')


    def produceGofValue(self,gofResult):
        data = gofResult['data']
        toys = np.sort(np.array(gofResult['toys']))
        if toys.shape[0] > 0:
            return round((1-(np.abs(toys - data)).argmin()/toys.shape[0]) * 100,6)
        else:
            return -1.

    def produceGofGraph(self,gof):
        g = ROOT.TGraph(len(gof))
        for i,(x,y) in enumerate(gof.items()):
            if y is not None:
                g.SetPoint(i,x,y)
        return g
            
        
    def produceLimitGraphs(self,limitsPerPoint):
        if len(limitsPerPoint) == 0:
            return None,None

        # Init #
        onesigma_up = []
        twosigma_up = []
        onesigma_low = []
        twosigma_low = []
        central = []
        data = []
        xpoints = []

        # Check if labels are all float or string labels #
        labels = list(limitsPerPoint.keys())
        floatLabels = True
        for label in labels:
            try:
                float(label)
            except:
                floatLabels = False

        if floatLabels:
            labelConv = {label:float(label) for label in labels}
        else:
            labelConv = {label:float(i) for i,label in enumerate(labels)}
            
        # Loop over limits #    
        for label,limits in limitsPerPoint.items():
            if limits is None:
                continue
            if len(limits.keys()) < 6:
                print(f'[WARNING] Not all limits found for label {label}')
                continue
            central.append(limits[50.0])
            twosigma_low.append(limits[2.5])
            onesigma_low.append(limits[16.0])
            onesigma_up.append(limits[84.0])
            twosigma_up.append(limits[97.5])
            data.append(limits[-1.])
            xpoints.append(labelConv[label])

        # Bands #
        onesigma_low.reverse()
        twosigma_low.reverse()
        onesigma_all = onesigma_up + onesigma_low
        twosigma_all = twosigma_up + twosigma_low

        xpoints_all = xpoints + list(reversed(xpoints))
        xpoints_f =  np.array(xpoints) 

        # Graphs #
        if len(xpoints) == 0:
            g_data = ROOT.TGraph()
            g_central = ROOT.TGraph()
            g_onesigma = ROOT.TGraph()
            g_twosigma = ROOT.TGraph()
        else:
            g_data = ROOT.TGraph(len(xpoints), np.array(xpoints),  np.array(data))
            g_central = ROOT.TGraph(len(xpoints),  np.array(xpoints),  np.array(central))
            g_onesigma = ROOT.TGraph(len(xpoints)*2,  np.array(xpoints_all),  np.array(onesigma_all))
            g_twosigma = ROOT.TGraph(len(xpoints)*2,  np.array(xpoints_all),  np.array(twosigma_all))

        if floatLabels:
            labelConv = None

        g_data.SetName("data")
        g_central.SetName("central")
        g_onesigma.SetName("onesigma")
        g_twosigma.SetName("twosigma")
        return (g_data,g_central,g_onesigma,g_twosigma),labelConv

    @staticmethod
    def saveCSV(csv_path,graphs,headers):
        xs = [list(g.GetX()) for g in graphs]
        ys = [list(g.GetY()) for g in graphs]
        allxs = set([x for xi in xs for x in xi])
        with open(csv_path,'w',newline='') as csvfile:
            writer = csv.writer(csvfile) #delimiter=' ',
                                        #quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # Header #
            writer.writerow(['x']+headers)
            for x in sorted(allxs):
                line = [x]
                for i in range(len(graphs)):
                    if x in xs[i]:
                        idx = xs[i].index(x)
                        line.append(ys[i][idx])
                    else:
                        line.append('-')
                writer.writerow(line)
        print(f'Wrote to file {csv_path}')
                    

    def plotLimits(self,graphs,options,suffix,labelConv=None):
        if graphs is None:
            print (f'No graph found for label {suffix}')
            return 

        suffix = str(suffix)
        suffix = suffix.replace(' ','_')
        g_data,g_central,g_onesigma,g_twosigma = graphs
        g_twosigma.SetFillColor(ROOT.TColor.GetColor('#fcd600'))
        g_twosigma.SetLineWidth(0)
        g_onesigma.SetFillColor(ROOT.TColor.GetColor('#17c42b'))
        g_onesigma.SetLineWidth(0)
        g_central.SetLineWidth(3)
        g_central.SetLineStyle(7)
        g_central.SetLineColor(ROOT.kBlack)
        g_central.SetMarkerStyle(20)
        g_central.SetMarkerSize(0.)
        g_central.SetMarkerColor(ROOT.kBlack)
        g_data.SetLineWidth(2)
        g_data.SetLineStyle(1)
        g_data.SetLineColor(ROOT.kBlack)
        g_data.SetMarkerStyle(20)
        g_data.SetMarkerSize(0.)
        g_data.SetMarkerColor(ROOT.kBlack)


        if g_central.GetN() == 0:
            return

        # Canvas #
        c1 = ROOT.TCanvas("c1","c1",800, 600)
        if 'canvas' in options.keys():
            self.useAttributes(c1,options['canvas'])
        if labelConv is not None:
            c1.SetBottomMargin(0.20)

        pdfPath = os.path.join(self.outputDir,f'{suffix}.pdf')
        c1.Print(pdfPath+'[')

        # Plot #
        xpoints = list(g_central.GetX())
        minx = min(xpoints)*0.9
        maxx = max(xpoints)*1.1
        b1 = ROOT.TH1F("b2","", len(xpoints)*2, minx, maxx)
        ylow = g_onesigma.GetHistogram().GetMinimum()
        yhigh = g_twosigma.GetHistogram().GetMaximum()
        b1.GetYaxis().SetRangeUser(ylow*0.9, yhigh*1.2)
        b1.SetStats(0)
        if 'lines' in options.keys():
            self.useAttributes(b1,options['lines'])
            b1.SetLineWidth(1)
        #b1.GetXaxis().SetNDivisions(len(xpoints))
        #if labelConv is not None:
        #    for label,x in labelConv.items():
        #        b1.GetXaxis().SetBinLabel(b1.GetXaxis().FindBin(x),label)
        b1.LabelsOption("v")
        b1.LabelsDeflate("X")
        b1.LabelsDeflate("Y")

        b1.Draw()
        g_twosigma.Draw("fe3same")
        g_onesigma.Draw("fe3same")
        g_central.Draw("lpsame")
        if self.unblind:
            g_data.Draw("lpsame")
        
        # Legend #
        if 'legend' in options.keys():
            leg = ROOT.TLegend()
            self.useAttributes(leg,options['legend'])
            if self.unblind:
                leg.AddEntry(g_data,"Observed","l")
            leg.AddEntry(g_central,"Expected (95% CL)","l")
            leg.AddEntry(g_onesigma,"#pm 1 #sigma Expected","f")
            leg.AddEntry(g_twosigma,"#pm 2 #sigma Expected","f")

        if 'texts' in options.keys():
            texts = []
            for headerCfg in options['texts']:
                texts.append(ROOT.TLatex(0.05,0.95,headerCfg['text']))
                self.useAttributes(texts[-1],headerCfg['options'])
                texts[-1].Draw("same")

        if 'additional' in options.keys():
            elementObj = []
            for element in options['additional']:
                if element['type'] == 'TGraph':
                    graph = ROOT.TGraph(len(element['data']))
                    for i,(x,y) in enumerate(element['data']):
                        graph.SetPoint(i,x,y)
                    graph.Draw('same')
                    elementObj.append(graph)
                else:
                    raise NotImplementedError
                self.useAttributes(elementObj[-1],element['options'])
                leg.AddEntry(elementObj[-1],element['name'])
        
        leg.Draw("same")
        c1.Print(pdfPath)

        c1.Print(pdfPath+']')

        print (f'Produced single limit plot : {pdfPath}')

    def useAttributes(self,obj,methodDict):
        for method,val in methodDict.items():
            if '.' in method:
                methods = method.split('.')
                nobj = obj
                for method in methods[:-1]:
                    nobj = getattr(obj,method)()
                self.applyAttribute(nobj,methods[-1],val)
            else:
                self.applyAttribute(obj,method,val)

    def applyAttribute(self,obj,method,val):
        if not hasattr(obj,method):
            raise RuntimeError(f'Object {obj.__class__.__name__} does not have method {method}')
        if isinstance(val,list) or isinstance(val,tuple):
            getattr(obj,method)(*val)
        elif isinstance(val,int) or isinstance(val,float):
            getattr(obj,method)(val)
        elif isinstance(val,str):
            if 'Color' in method and val.startswith('#'):
                val = ROOT.TColor.GetColor(val)
            getattr(obj,method)(val)
        else:
            raise RuntimeError(f'Type {type(val)} not understood')

                
    def plotMultipleGraphs(self,graphs,legends,options,title,name,attributes=[],labelConv=None):
        if len(graphs) == 0:
            print (f'No multi graph found for label {suffix}')
            return 

        # Indices of non 'None' graphs #
        valid_idx = [g is not None for g in graphs]

        # Canvas #
        c1 = ROOT.TCanvas("c1","c1",800, 600)
        if 'canvas' in options.keys():
            self.useAttributes(c1,options['canvas'])

        pdfPath = os.path.join(self.outputDir,f'{name.replace(" ","_")}.pdf')
        c1.Print(pdfPath+'[')

        # Add main `lines` attribute to per-line attributes #
        if len(attributes) == 0:
            attributes = [{}] * len(graphs)
        elif len(attributes) != len(graphs):
            raise RuntimeError(f'Plot {name} has {len(graphs)} but {len(attributes)} line plot options')
        if 'lines' in options.keys():
            for i in range(len(attributes)):
                for key,val in options['lines'].items():
                    if key not in attributes[i].keys():
                        attributes[i][key] = val

        # Check if some overrides are done anywhere, otherwise use default #
        custom_color = False
        custom_yrange = False
        for attribute in attributes:
            for key in attribute.keys():
                if 'Set' in key and 'Color' in key:
                    custom_color = True
                if 'SetRange' in key and 'GetYaxis' in key:
                    custom_yrange = True

        # Plot #
        xpoints = list()
        for graph in graphs:
            if graph is not None and graph.GetN()>0:
                for val in graph.GetX():
                    if val not in xpoints:
                        xpoints.append(val)
        xpoints = sorted(xpoints)
        if len(xpoints) > 0:
            minx = min(xpoints)*0.9
            maxx = max(xpoints)*1.1
        else:
            minx = 0.
            maxx = 1.
        b1 = ROOT.TH1F("b2","", min(1,len(xpoints)*2), minx, maxx)
        b1.SetTitle(title)
        if 'lines' in options.keys():
            self.useAttributes(b1,options['lines'])
        if not custom_yrange:
            ylow  = min([g.GetHistogram().GetMinimum() for g in graphs]) * 1.1
            yhigh = max([g.GetHistogram().GetMaximum() for g in graphs]) * 1.1
            b1.GetYaxis().SetRangeUser(ylow, yhigh)
        b1.SetStats(0)
        #b1.GetXaxis().SetNDivisions(len(xpoints))
        if labelConv is not None:
            for label,x in labelConv.items():
                b1.GetXaxis().SetBinLabel(b1.GetXaxis().FindBin(x),label)
        b1.LabelsDeflate("X")
        b1.LabelsDeflate("Y")


        # Legend #
        if 'legend' in options.keys():
            leg = ROOT.TLegend()
            self.useAttributes(leg,options['legend'])
            for g,legend in zip(graphs,legends):
                if g is not None:
                    leg.AddEntry(g,legend,"lp")

        # Legend #
        N = len(graphs)
        b1.Draw()

        for g in graphs:
            if g is not None:
                if custom_color:
                    g.Draw("lpsame")
                else:
                    g.Draw("lpsame PLC")

        # Attributes #
        for g,attribute in zip(graphs,attributes):
            if g is not None:
                self.useAttributes(g,attribute)

        if 'texts' in options.keys():
            texts = []
            for headerCfg in options['texts']:
                texts.append(ROOT.TLatex(0.05,0.95,headerCfg['text']))
                self.useAttributes(texts[-1],headerCfg['options'])
                texts[-1].Draw("same")


        if 'legend' in options.keys():
            leg.Draw()
        
        c1.Print(pdfPath)

        c1.Print(pdfPath+']')
        print (f'Produced multi graph plot : {pdfPath}')



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Limit scans')
    parser.add_argument('--yaml', action='store', required=True, type=str,
                        help='Yaml containing parameters')
    parser.add_argument('-j','--jobs', action='store', required=False, type=int, default=1,
                        help='Number of commands to run in parallel (default = 1), using -1 will spawn all the commands')
    parser.add_argument('--combine', action='store', nargs='*', required=False, type=str, default=[],
                        help='Combine modes to run (as defined in the config file')
    parser.add_argument('--plotIt', action='store_true', required=False, default=False,
                        help='Browse datacard files and produce plots via plotIt')
    parser.add_argument('--unblind', action='store_true', required=False, default=False,
                        help='Unblind data')
    parser.add_argument('--force', action='store_true', required=False, default=False,
                        help='Force recreation of save')
    parser.add_argument('--no_save', action='store_true', required=False, default=False,
                        help='Does not create the root save file (to be combined with --force probably)')
    parser.add_argument('--debug', action='store_true', required=False, default=False,
                        help='Avoids sending jobs')
    parser.add_argument('-a','--additional', action='store', required=False, default=None, type=str,
                        help='Additional arguments to pass to the commands [as a string, beware to include a space right before]')
    args = parser.parse_args()

    with open(args.yaml,'r') as handle:
        content = yaml.load(handle,Loader=YMLIncludeLoader)

   
    instance = Scan(**content,
                    combine_args    = args.combine,
                    plotIt          = args.plotIt,
                    jobs            = args.jobs,
                    force           = args.force,
                    no_save         = args.no_save,
                    debug           = args.debug,
                    unblind         = args.unblind)


