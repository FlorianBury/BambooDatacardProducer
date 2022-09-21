import os
import io
import sys
import re
import json
import time
import glob
import copy
import ctypes
import shlex
import yaml
import parse
import random
import shutil
import string
import argparse
import logging
import subprocess
import collections
import importlib
import itertools
import numpy as np
import math
import multiprocessing as mp
import numpy as np
from functools import partial
import enlighten
from contextlib import redirect_stdout
import ROOT

from CP3SlurmUtils.Configuration import Configuration
from CP3SlurmUtils.SubmitWorker import SubmitWorker
from CP3SlurmUtils.Exceptions import CP3SlurmUtilsException    

from context import TFileOpen
from yamlLoader import YMLIncludeLoader
from interpolation import InterpolateContent
from postfits import PostfitPlots
from txtwriter import Writer
from numpy_hist import NumpyHist

from IPython import embed

ROOT.gROOT.SetBatch(True) 

# Inference setup #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SETUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'inference')
SETUP_SCRIPT = 'setup.sh'

ULIMIT = "ulimit -S -s unlimited"

COMBINE_DEFAULT_ARGS = [
    #"--X-rtd MINIMIZER_freezeDisassociatedParams",
    #"--X-rtd MINIMIZER_multiMin_hideConstants",
    #"--X-rtd MINIMIZER_multiMin_maskConstraints",
    #"--X-rtd MINIMIZER_multiMin_maskChannels=2",
    "--X-rtd REMOVE_CONSTANT_ZERO_POINT=1",
    "--cminDefaultMinimizerType Minuit2",
    "--cminDefaultMinimizerStrategy 0",
    "--cminDefaultMinimizerTolerance 0.1",
    "--cminFallbackAlgo Minuit2,0:0.2",
    "--cminFallbackAlgo Minuit2,0:0.4"
]



class Datacard:
    def __init__(self,outputDir=None,configPath=None,path=None,yamlName=None,worker=None,groups=None,shapeSyst=None,normSyst=None,histConverter=None,era=None,use_syst=False,fix_histograms=True,include_overflow=False,root_subdir=None,histAdditionals=None,histCorrections=None,pseudodata=False,rebin=None,histEdit=None,histCut=None,textfiles=None,plotIt=None,combineConfigs=None,logName=None,custom_args=None,save_datacard=True,**kwargs):
        self.outputDir          = outputDir
        self.configPath         = configPath
        self.path               = path
        self.worker             = worker
        self.era                = era
        self.use_syst           = use_syst
        self.fix_histograms     = fix_histograms
        self.include_overflow   = include_overflow
        self.yamlName           = yamlName
        self.root_subdir        = root_subdir
        self.pseudodata         = pseudodata
        self.histConverter      = histConverter
        self.groups             = groups
        self.normSyst           = normSyst
        self.shapeSyst          = shapeSyst
        self.rebin              = rebin
        self.histEdit           = histEdit
        self.histCut            = histCut
        self.textfiles          = textfiles
        self.plotIt             = plotIt
        self.combineConfigs     = combineConfigs
        self.histAdditionals    = histAdditionals
        self.histCorrections    = histCorrections
        self.save_datacard      = save_datacard
        self.custom_args        = custom_args
        self.aggregation        = {}

        # Format eras as string #
        if isinstance(self.era,list) or isinstance(era,tuple):
            self.era = [str(era) for era in self.era]
        else:
            self.era = str(self.era)

        # Make sure few things are iterable #
        if not isinstance(self.path,list):
            self.path = [self.path]
        if not isinstance(self.yamlName,list):
            self.yamlName = [self.yamlName]

        # Add logging output to log file #
        #handler = logging.FileHandler(os.path.join(self.outputDir,logName),mode='w')
        #logger = logging.getLogger()
        #logger.addHandler(handler)


    def initialize(self):
        # Get entries that can either be dicts or list of dicts (eg with !include in yaml) #
        self.groups = self.includeEntry(self.groups,'Groups',eras=self.era)
        self.normSyst = self.includeEntry(self.normSyst,'Norm syst',eras=self.era)
        self.shapeSyst = self.includeEntry(self.shapeSyst,'Shape syst',eras=self.era)
        self.combineConfigs = self.includeEntry(self.combineConfigs,'CombineConfigs',eras=self.era)
        self.histCorrections = self.includeEntry(self.histCorrections,'HistCorrections',eras=self.era,keep_inner_list=True)
        self.histAdditionals = self.includeEntry(self.histAdditionals,'HistAdditionals',eras=self.era,keep_inner_list=True)

        # Make sure the histConverter keys are string (otherwise problems with eras, etc) #
        for k,v in self.histConverter.items():
            if not isinstance(k,str):
                self.histConverter[str(k)] = self.histConverter.pop(k)
            if isinstance(k,dict):
                for sk in v.keys():
                    if not isinstance(sk,str):
                        self.histConverter[str(k)][str(sk)] = self.histConverter[str(k)].pop(sk)

        # Initialise containers #
        self.content = {f'{histName}_{self.era}':{g:{} for g in self.groups.keys()} for histName in self.histConverter.keys()}
        self.systPresent = {histName:{group:[] for group in self.groups.keys()} for histName in self.content.keys()}
        self.yields = {histName:{g:None for g in self.groups.keys()} for histName in self.content.keys()}

    @staticmethod
    def includeEntry(entries,name,eras=None,keep_inner_list=False):
        if eras is not None:
            multi_era = False
            if isinstance(eras,list) or isinstance(eras,tuple):
                multi_era = len(eras) > 1
            else:
                eras = [eras]
        if isinstance(entries,dict):
            # Check if the entries include era dependency #
            keys = [str(key) for key in entries.keys()]
            if eras is not None and len(set(keys).intersection(set(eras))) > 0:
                combined = {}
                for era in eras:
                    subentry = entries[list(entries.keys())[keys.index(era)]]
                    if isinstance(subentry,dict):
                        combined[str(era)] = subentry
                    elif isinstance(subentry,tuple) or isinstance(subentry,list):
                        if keep_inner_list:
                            combined[str(era)] = []
                        else:
                            combined[str(era)] = {}
                        for ientry,eraCfg in enumerate(subentry):
                            if not isinstance(eraCfg,dict):
                                raise RuntimeError(f'Subentry index {ientry} from era {era} and {name} is not a dict')
                            if keep_inner_list:
                                combined[str(era)].append(eraCfg)
                            else:
                                combined[str(era)].update(eraCfg)
                    else:
                        raise RuntimeError(f'Subentry with era {era} and name {name} is nor a dict neither a list/tuple')
                if multi_era:
                    return combined
                else:
                    return combined[str(eras[0])]
            else:
                return entries
        elif isinstance(entries,list) or isinstance(entries,tuple):
            if keep_inner_list:
                combined = []
            else:
                combined = {}
            for ientries,subentries in enumerate(entries):
                if not isinstance(subentries,dict):
                    raise RuntimeError(f'Subentry index {ientries} from {name} is not a dict')
                if keep_inner_list:
                    combined.append(subentries)
                else:
                    combined.update(subentries)
            if keep_inner_list:
                return combined
            else:
                return {str(k):v for k,v in combined.items()}
        elif entries is None:
            return None
        else: 
            raise RuntimeError(f"{name} format {type(entries)} not understood")


    def run_production(self):
        # Initialize #
        self.initialize()

        # Load Yaml configs from bamboo #
        self.yaml_dict = self.loadYaml(self.path,self.yamlName)
        if self.histAdditionals is not None and 'yaml' in self.histAdditionals.keys():
            addDirs = [os.path.dirname(yamlFile) for yamlFile in self.histAdditionals['yaml']]
            addYaml = [os.path.basename(yamlFile) for yamlFile in self.histAdditionals['yaml']]
            for fileName,fileCfg in self.loadYaml(addDirs,addYaml)['samples'].items():
                if fileName not in self.yaml_dict['samples']:
                    self.yaml_dict['samples'][fileName] = fileCfg
                    logging.debug(f'\tAdding file {fileName} to yaml dictionary')

        # Print out #
        logging.info(f'Running production over following categories :')
        for cat in self.content.keys():
            logging.info(f'... {cat}')

        # Apply pseudo-data #
        if self.pseudodata:
            self.generatePseudoData()

        # Run production #
        if self.histCorrections:
            self.checkforIntermediateAggregation()
        self.loopOverFiles()
        if self.pseudodata:
            self.roundFakeData()
        if self.histCorrections is not None:
            self.applyCorrections(after_aggregation=False,after_rebinning=False)
        if self.aggregation:
            self.applyAggregation()
            if self.histCorrections is not None:
                self.applyCorrections(after_aggregation=True,after_rebinning=False)
        if self.histEdit is not None:
            self.applyEditing()
        if self.histCut is not None:
            self.applyCut()
        self.applyOverflow()
        if self.rebin is not None:
            self.applyRebinning()
            if self.histCorrections is not None:
                self.applyCorrections(after_aggregation=True,after_rebinning=True)
        self.yieldPrintout()
        if self.save_datacard:
            self.saveDatacard()
        return


    def generatePseudoData(self):
        logging.info('Will use sum of mc samples as pseudodata')
        if any([isinstance(files,dict) for groupCfg in self.groups.values() for files in groupCfg['files'] 
                                if groupCfg['type'] == 'mc' and isinstance(groupCfg['files'],list)]):
            # Some groups have conditional MC samples
            filesPerCat = {'__all__':[]}
            for group,groupCfg in self.groups.items():
                if groupCfg['type'] != 'mc':
                    continue
                files = copy.deepcopy(groupCfg['files'])
                if not isinstance(files,list):
                    files = [files] # Make it always a list
                for i,f in enumerate(files):
                    if isinstance(f,str) and f not in filesPerCat['__all__']:
                        filesPerCat['__all__'].append(f)
                    if isinstance(f,dict): 
                        if not 'files' in f.keys():
                            logging.warning(f'No `files` item in entry {i} of group {group}, will omit it')
                        elif not 'cat' in f.keys():
                            logging.warning(f'No `cat` item in entry {i} of group {group}, will omit it')
                        else:
                            if isinstance(f['cat'],str):
                                cat = (f['cat'],)
                            elif isinstance(f['cat'],list):
                                cat = tuple(f['cat'])
                                # Key needs to be immutable
                            else:
                                raise RuntimeError(f"Category {f['cat']} type {type(f['cat'])} in entry {i} of group {group} not understood")
                            if cat in filesPerCat.keys():
                                filesPerCat[cat].extend(f['files'])
                            else:
                                filesPerCat[cat] = f['files']
            # Make it a files content for the data_obs group from pseudodata #
            assert len(filesPerCat) > 1 # == 1 means only __all__, and that should not happen
            mc_samples = []
            for cat,files in filesPerCat.items():
                if cat == '__all__':
                    mc_samples.extend(filesPerCat['__all__'])
                else:
                    mc_samples.append({'cat'   : list(cat),
                                       'files' : files})
        else:
            # No conditional -> add all samples in data_obs group as pseudodata
            mc_samples = [sample for sample,sampleCfg in self.yaml_dict['samples'].items() if sampleCfg['type']=='mc']
        # Make the group #
        self.groups['data_obs'] = {'files'  : mc_samples,
                                   'legend' : 'pseudo-data',
                                   'type'   : 'data',
                                   'group'  : 'data'}
        self.findGroup(force=True) # Force the recreation of cached sampleToGroup to take into account the change of groups dict
            # One needs the yaml dict to be there as default, and yaml dict has called findGroup
        for mc_sample in mc_samples:
            if isinstance(mc_sample,str):
                files = [mc_sample]
            if isinstance(mc_sample,dict):
                files = mc_sample['files']
            for f in files:
                if 'data_obs' not in self.findGroup(f):
                    raise RuntimeError(f'Something went wrong : sample {f} not flagged as data_obs for pseudodata')
            # Need to add to containers #
        self.initialize()

    def loopOverFiles(self):
        # Save histogram names in rootfiles #
        self.fileHistList = []
        for val in self.histConverter.values():
            if isinstance(val,list):
                self.fileHistList.extend(val)
            else:
                self.fileHistList.append(val)
        # Get the list from all directories #
        if isinstance(self.path,list):
            files = set()
            for path in self.path:
                files = files.union(set([f for f in glob.glob(os.path.join(path,'results','*.root'))]))
            files = list(files)
        else:
            files = glob.glob(os.path.join(self.path,'results','*.root'))

        logging.info('Running over directories :')
        for path in self.path:
            logging.info(f'... {path}')

        # Start loop over samples #
        pbar = enlighten.Counter(total=len(files), desc='Progress', unit='files')
        for f in sorted(files):
            pbar.update()
            if '__skeleton__' in f:
                continue
            sample = os.path.basename(f)
            logging.debug(f'Looking at file {f}')
            # Check if in the group list #
            groups = self.findGroup(sample)
            if len(groups) == 0:
                logging.warning("Could not find sample %s in group list"%sample)
                continue
            if len(groups) != len(set(groups)): # One or more groups are repeated
                logging.error(f'Sample {f} will be present in several times in the same group :')
                for group, occurences in collections.Counter(groups).items():
                    logging.error(f'... {group:30s} : {occurences:2d}')
                raise RuntimeError(f'Repetitions of the same group is likely a bug, will stop here')
            # Get histograms #
            hist_dict = self.getHistograms(f)
            if len(hist_dict) == 0 or sum([len(val) for val in hist_dict.values()]) == 0:
                logging.debug('\tNo histogram taken from this sample')
                continue
            # Get additional files #
            if self.histAdditionals is not None and sample in self.histAdditionals['samples'].keys() and self.use_syst:
                for systEntry in self.histAdditionals['samples'][sample]:
                    files = systEntry['files']
                    if isinstance(files,str):
                        files = [files]
                    for f in files:
                        logging.debug(f'Looking at {f}')
                    file_hists = [self.getHistograms(f) for f in files]
                    for cat in hist_dict.keys():
                        additional_hists = [fh[cat]['nominal'] for fh in file_hists if cat in fh.keys()]
                        if len(additional_hists) == 1:
                            h_diff = additional_hists[0]
                            if h_diff.Integral() != 0:
                                h_diff.Scale(hist_dict[cat]['nominal'].Integral()/h_diff.Integral())
                            h_diff.Add(hist_dict[cat]['nominal'],-1)
                            h_up = copy.deepcopy(hist_dict[cat]['nominal'])
                            h_down = copy.deepcopy(hist_dict[cat]['nominal'])
                            h_up.Add(h_diff,+1)
                            h_down.Add(h_diff,-1)
                            hist_dict[cat][f'{systEntry["name"]}Up'] = h_up
                            hist_dict[cat][f'{systEntry["name"]}Down'] = h_down
                        elif len(additional_hists) == 2:
                            hist_dict[cat][f'{systEntry["name"]}Up'] = additional_hists[0]
                            hist_dict[cat][f'{systEntry["name"]}Down'] = additional_hists[1]
                        elif len(additional_hists) > 2:
                            raise RuntimeError
            # Print out #
            logging.debug("\tFound following histograms :")
            for histName in hist_dict.keys():
                # Printout and check if empty #
                logging.debug(f'\t... {histName} : {len(hist_dict[histName])} histograms')
                if len(hist_dict[histName]) == 0:
                    continue
                # Add to present systematics #
                for group in groups:
                    self.systPresent[histName][group].append({'systematics': [systname for systname in hist_dict[histName].keys() if systname != 'nominal'],
                                                              'nominal': copy.deepcopy(hist_dict[histName]['nominal'])})
            # Add to content by group #
            logging.debug("\tWill be used in groups : ")
            for group in groups:
                # if several groups, need to fetch for each group 
                # -> need to have different memory adress
                if self.addSampleToGroup(sample,group,copy.deepcopy(hist_dict)):
                    logging.debug(f"\t... {group}")
            del hist_dict

        self.checkForMissingNominals()
        self.yields = self.getYields()

        # Correct for missing systematics #
        for histName in self.systPresent.keys():
            for group in self.systPresent[histName].keys():
                for sampleDict in self.systPresent[histName][group]:
                    for systName in self.content[histName][group].keys():
                        if systName == "nominal":
                            continue
                        if systName not in sampleDict['systematics']:
                            self.content[histName][group][systName].Add(sampleDict['nominal'])

        # Renaming to avoid overwritting ROOT warnings #
        for histName in self.content.keys(): 
            for group in self.content[histName].keys():
                for systName,hist in self.content[histName][group].items():
                    if self.pseudodata and group == 'data_real':
                        continue
                    if hist is None:
                        continue
                    name = f'{histName}__{group}__{systName}'
                    hist.SetName(name)
                    hist.SetTitle(name)

    def addSampleToGroup(self,sample,group,hist_dict):
        used = False
        for histname,hists in hist_dict.items():
            # If no hist, continue #
            if len(hists) == 0:
                continue
            # If in condition dictionnary, check that it matches at least one condition #
            if sample in self.condSample.keys():  
                if not any([self.checkConditional(conditions=cond,group=group,cat=histname) for cond in self.condSample[sample]
]):                    
                    continue
            # Get nominal and add it #
            nominal = hists['nominal']
            if not 'nominal' in self.content[histname][group].keys():
                self.content[histname][group]['nominal'] = copy.deepcopy(nominal)
            else:
                self.content[histname][group]['nominal'].Add(nominal)
            used = True
            # Add the systematic shape #
            if self.use_syst:
                for systName in hists.keys():
                    if systName == 'nominal':
                        continue
                    hist = hists[systName]
                    if systName not in self.content[histname][group].keys():
                        self.content[histname][group][systName] = copy.deepcopy(hist)
                    else:
                        self.content[histname][group][systName].Add(hist)
        return used

    def findGroup(self,sample=None,force=False):
        """ force entry to force the recration of the cached sampleToGroup """
        # Creation of this attribute if not defined already (avoid lengthy recomputations all the time) # 
        if not hasattr(self,'sampleToGroup') or force:
            self.sampleToGroup = {} # Key = sample, value = group 
            self.condSample =  {}   # Key = sample, value = conditions (list of dicts)
            self.condGroup  =  {}   # Key = group,  value = conditons
            for group in self.groups.keys():
                self.condGroup[group] = []
                # Check files attribute # 
                if 'files' not in self.groups[group]:
                    logging.warning("No `files` item in group {}".format(group))
                    files = []
                else:
                    files = self.groups[group]['files']
                # Make it a list all the time #
                if not isinstance(files,list):
                    files = [files]
                # Conditional group (based on era or category) #
                if any([isinstance(f,dict) for f in files]): 
                    tmp_files = []
                    for i,f in enumerate(files):
                        if isinstance(f,dict):
                            if not 'files' in f.keys():
                                logging.warning(f'No `files` item in entry {i} of group {group}, will omit it')
                            else:
                                # Add to sampleToGroup and condSample #
                                if isinstance(f['files'],list):
                                    add_files = f['files']
                                else:
                                    add_files = [f['files']]
                                for add_file in add_files:
                                    if add_file not in self.condSample.keys():
                                        self.condSample[add_file] = []
                                    cond = {k:v for k,v in f.items() if k != 'files'}
                                    if cond not in self.condGroup[group]:
                                        self.condGroup[group].append(copy.deepcopy(cond))
                                    cond['group'] = group
                                    if cond not in self.condSample[add_file]:
                                        self.condSample[add_file].append(cond)
                                    if add_file not in tmp_files:
                                        tmp_files.append(add_file)
                        else:
                            tmp_files.append(f)
                            self.condGroup[group].append({})
                    files = tmp_files
                else:
                    for f in files:
                        if f not in self.condSample.keys():
                            self.condSample[f] = []
                        self.condSample[f].append({'group':group})
                    self.condGroup[group].append({})
                # Add the link between sample and group #
                for f in files:
                    if f in self.sampleToGroup.keys():
                        self.sampleToGroup[f].append(group)
                    else:
                        self.sampleToGroup[f] = [group]

        # Search the saved dictionnary #
        if sample is None:
            return None
        if sample in self.sampleToGroup.keys():
            return self.sampleToGroup[sample]
        else:
            return []
    
    def checkforIntermediateAggregation(self):
        for iconf,corrConfig in enumerate(self.histCorrections):
            if 'histConverter' in corrConfig.keys():
                if 'aggregation' not in corrConfig.keys():
                    raise RuntimeError(f"If you use `histConverter` in nonclosure entry {iconf} of era {self.era}, you need `aggregation`")
                histConverter = corrConfig['histConverter']
                aggregation = corrConfig['aggregation']
                intermediate = [v for values in aggregation.values() for v in values]
                if set(histConverter.keys()).intersection(set(intermediate)) != set(histConverter.keys()):
                    logging.info(f'Nonclosure entry {iconf} of era {self.era}')
                    logging.info('\thistConverter keys of intermediate :')
                    for key in histConverter.keys():
                        logging.info(f'\t... {key}')
                    logging.info('\taggregation values :')
                    for val in intermediate:
                        logging.info(f'\t... {val}')
                    raise RuntimeError(f'Nonclosure entry {iconf} of era {self.era} mismatch')
                for key in aggregation.keys():
                    if key in self.histConverter.keys(): # found a key to change
                        self.aggregation[f'{key}_{self.era}'] = [f'{val}_{self.era}' for val in aggregation[key]]
                        logging.info(f'Main category {key} split between :')
                        del self.histConverter[key]
                        for newCat in aggregation[key]:
                            self.histConverter[newCat] = histConverter[newCat]
                            logging.info(f'... {newCat}')
        if self.aggregation:
            self.initialize()

    @staticmethod
    def getRegexSet(patterns,listTest):
        """ returns all element of testList that pass the regex on at least one of the patterns """
        if not isinstance(patterns,list):
            patterns = [patterns]
        # Check if regular check or anti search #
        if any([pattern.startswith('\\') for pattern in patterns]):
            # Anti-search detected #
            if not all([pattern.startswith('\\') for pattern in patterns]):
                raise RuntimeError(f'You used `\\` for an anti search but it can only work if done for all items : {patterns}')
            anti_search = True
            outSet = set(listTest)
            patterns = [pattern.replace('\\','') for pattern in patterns]
        else:
            # Classic search #
            outSet = set()
            anti_search = False
        # Test all patterns #
        for pattern in patterns:
            regex = re.compile(pattern)
            matches = {element for element in listTest if regex.match(element) is not None}
            if anti_search:
                outSet -= matches
            else:
                outSet.update(matches)
        return outSet


    def checkConditional(self,conditions,group,cat):
        """ Check conditions for a give sample in certain category, returns True if OK """
        # Check if there is an era condition #
        if 'era' in conditions.keys() and str(conditions['era']) != str(self.era):
            return False
        # Check if there is a category condition #
        if 'cat' in conditions.keys():
            if f'_{self.era}' in cat:
                cat = cat.replace(f'_{self.era}','')
            if cat not in self.getRegexSet(conditions['cat'],self.histConverter.keys()):
                return False
        # Check if group condition #
        if 'group' in conditions.keys():
            if group not in self.getRegexSet(conditions['group'],self.groups.keys()):
                return False
        return True
                
    def getHistograms(self,rootfile):
        sample = os.path.basename(rootfile) 
        # Get config info #
        lumi = self.yaml_dict["luminosity"][str(self.era)]
        if sample not in self.yaml_dict["samples"].keys():
            logging.warning(f'Sample {sample} not found in yaml plot dict, will omit it')
            return {}
        sample_type = self.yaml_dict["samples"][sample]['type']
        if sample_type == "mc" or sample_type == "signal":
            xsec = self.yaml_dict["samples"][sample]['cross-section']
            sumweight = self.yaml_dict["samples"][sample]['generated-events']
            br = self.yaml_dict["samples"][sample]["branching-ratio"] if "branching-ratio" in self.yaml_dict["samples"][sample].keys() else 1.
        else:
            xsec = None
            sumweight = None
            br = None

        logging.debug(f"\tFound following information")
        for name,val in zip(['cross-section','Sum of weights','branching-ratio'],[xsec,sumweight,br]):
            if val is not None:
                logging.debug(f"\t... {name:20s} = {val:25.5f}")

        # Open ROOT file #
        with TFileOpen(rootfile,'r') as F:
            # Check that at least one histogram is there #
            # (to make computations faster, especially when using split and several dirs)
            if not any([F.GetListOfKeys().Contains(hName) for hName in self.fileHistList]):
                # Not a single histogram name has been found in the file, return nothing
                return {}
            # Get list of hist names #
            list_histnames = []
            for key in F.GetListOfKeys():
                keyName = key.GetName()
                if not self.use_syst and '__' in keyName:
                    continue
                if any([histname in keyName for histname in self.fileHistList]):
                    list_histnames.append(keyName)

            # Loop through hists #
            hist_dict = {}
            for datacardname, histnames in self.histConverter.items():
                # Check if conditional sample #
                if sample in self.condSample.keys():  
                    if not any([self.checkConditional(conditions=cond,group=group,cat=datacardname) \
                                    for group in self.sampleToGroup[sample] for cond in self.condSample[sample]]):
                        # Test all possible conditions for that sample, if any one is matched for the category use it
                        continue
                        
                datacardname += f'_{self.era}'
                hist_dict[datacardname] = {}
                if not isinstance(histnames,list):
                    histnames = [histnames]
                # Get all defined systematics in the file histogram #
                systPresentInFile = []
                if self.use_syst:
                    for histname in list_histnames:
                        if '__' not in histname:
                            continue
                        hName, sName = histname.split('__')
                        if hName not in histnames:
                            continue
                        if sName not in systPresentInFile:
                            systPresentInFile.append(sName)
                    # this is because if several histograms are to be summed in the conversion
                    # and one has a systematic that the other does not have, one needs to add 
                    # the nominal to fill up this missing systematic

                for histname in histnames:
                    # Check #
                    if not histname in list_histnames:
                        continue
                    # Nominal histogram #
                    hnom = self.getHistogram(F,histname,lumi,br,xsec,sumweight)
                    if not 'nominal' in hist_dict[datacardname].keys():
                        hist_dict[datacardname]['nominal'] = copy.deepcopy(hnom)
                    else:
                        hist_dict[datacardname]['nominal'].Add(hnom)
                    # Systematic histograms #
                    listsyst = [histname + '__' + systName for systName in systPresentInFile]
                    for syst in listsyst:
                        if syst in list_histnames: # systematic is there
                            h = self.getHistogram(F,syst,lumi,br,xsec,sumweight) 
                        else: # systematic for histogram is not present, use the nominal
                            h = copy.deepcopy(hnom) 
                        systName = syst.split('__')[-1]
                        if systName.endswith('up'):
                            systName = systName[:-2]+"Up"
                        elif systName.endswith('down'):
                            systName = systName[:-4]+"Down"
                        else:
                            raise RuntimeError("Could not understand systematics {}".format(systName))
                        if not systName in hist_dict[datacardname].keys():
                            hist_dict[datacardname][systName] = copy.deepcopy(h)
                        else:
                            hist_dict[datacardname][systName].Add(h)
        return hist_dict

    @staticmethod
    def getHistogram(F,histnom,lumi=None,xsec=None,br=None,sumweight=None):
        # Get hist #
        h = copy.deepcopy(F.Get(histnom))
        # Normalize hist to data #
        if lumi is not None and xsec is not None and br is not None and sumweight is not None:
            h.Scale(lumi*xsec*br/sumweight)
        return h
             
    def loadYaml(self,paths,yamlNames):
        yamlDict = {}
        for path in paths:
            if not os.path.isdir(path):
                logging.error(f"`{path}` is not a directory")
                continue
            for yamlName in yamlNames:
                yamlPath = os.path.join(path,yamlName)
                if not os.path.exists(yamlPath):
                    logging.warning(f"`{yamlPath}` -> not found, skipped")
                    continue
                # Parse YAML #
                with open(yamlPath,"r") as handle:
                    full_dict = yaml.load(handle,Loader=yaml.FullLoader)
                # Get Lumi per era #  
                lumi_dict = full_dict["configuration"]["luminosity"]
                if 'luminosity' not in yamlDict.keys():
                    yamlDict['luminosity'] = lumi_dict
                else:
                    for key,val in lumi_dict.items():
                        if key not in yamlDict['luminosity'].keys():
                            yamlDict['luminosity'][key] = val
                        else:
                            if yamlDict['luminosity'][key] != lumi_dict[key]:
                                logging.warning(f'Found different luminosity values for key {key} : {yamlDict["luminosity"][key]} != {lumi_dict[key]} ... this is suspicious') 

                # Get data per sample #
                info_to_keep = ['cross-section','generated-events','group','type','era','branching-ratio']
                if 'samples' not in yamlDict.keys():
                    sample_dict = {}
                    for sample,data in full_dict['files'].items():
                        sample_dict[sample] = {k:data[k] for k in data.keys() & info_to_keep}
                    yamlDict['samples'] = sample_dict
                else:
                    for sample,sampleCfg in full_dict['files'].items():
                        if sample not in yamlDict['samples'].keys():
                            yamlDict['samples'].update({sample:sampleCfg})
                        else:
                            for key in ['cross-section','generated-events','branching-ratio']:
                                if key in sampleCfg:
                                    diff = 2*abs(sampleCfg[key]-yamlDict['samples'][sample][key])/(sampleCfg[key]+yamlDict['samples'][sample][key])
                                    if diff > 1e-6:
                                        logging.warning(f'Found different value in sample `{sample}` for entry `{key}` : {sampleCfg[key]} != {yamlDict["samples"][sample][key]} [{diff*100:8.5f}%] ... this is suspicious')
                        
                # Few checks #
                for sample in yamlDict['samples'].keys():
                    for key in ['cross-section','generated-events']:
                        if key in yamlDict['samples'][sample].keys() and yamlDict['samples'][sample][key] < 0:
                            logging.warning(f'Sample {sample} has {key} = {yamlDict["samples"][sample][key]} (< 0), this might not be expected')

                # Get plot options #
                if 'plots' not in yamlDict.keys(): 
                    yamlDict['plots'] = full_dict['plots']
                else:
                    yamlDict['plots'].update(full_dict['plots'])
        # Check it actually found something #
        if len(yamlDict) == 0:
            raise RuntimeError('No info has been extracted from the plotIt Yaml files, maybe they all do not exist ?')

        # Overwrite some values based on config #
        info_to_override = ['cross-section','generated-events','branching-ratio']
        for sample in yamlDict['samples'].keys():
            groups = self.findGroup(sample)
            for key in info_to_override:
                overrideFlag = [i for i,group in enumerate(groups) if key in self.groups[group].keys()]
                if len(overrideFlag) == 1: # One group overrides the default values
                    yamlDict['samples'][sample][key] = self.groups[groups[overrideFlag[0]]][key]
                    logging.debug(f'Override of sample {sample} in group {groups} of {key} = {yamlDict["samples"][sample][key]}')
                    if len(groups) > 1:
                        logging.warning(f'Sample {sample} parameter {key} has been overwritten to the value {self.groups[groups[overrideFlag[0]]][key]} from the config but is present in several groups ({",".join(groups)}) that will all be impacted')
                elif len(overrideFlag) > 1: # Several groups override the default valeues 
                    values = [self.groups[groups[flag]][key] for flag in overrideFlag]
                    if values.count(values[0]) != len(values): # different override values 
                        raise RuntimeError(f'Sample {sample} parameter {key} will be overwritten from the config but is present in several groups ({",".join(groups)}) that have defined several values ({",".join([str(val) for val in values])}, will stop here as this can lead to undefined behaviour )')
                    else: # same override values
                        yamlDict['samples'][sample][key] = values[0]
                        logging.debug(f'Override of sample {sample} in group {groups} of {key} = {yamlDict["samples"][sample][key]}')

        return yamlDict

    def roundFakeData(self):
        for histName in self.content.keys():
            for group in self.content[histName].keys():
                if group == 'data_obs':
                    systToRemove = [systName for systName in self.content[histName][group].keys() if systName != 'nominal']
                    for systName in systToRemove:
                        del self.content[histName][group][systName]
                    if 'nominal' not in self.content[histName][group].keys():
                        raise RuntimeError(f'No nominal found for pseudodata of histogram {histName}')
                    hist = self.content[histName][group]['nominal']
                    for i in range(0,hist.GetNbinsX()+2):
                        if hist.GetBinContent(i) > 0:
                            hist.SetBinContent(i,round(hist.GetBinContent(i)))   # Round content 
                            hist.SetBinError(i,math.sqrt(hist.GetBinContent(i))) # Poisson error
                        else:
                            hist.SetBinContent(i,0)
                            hist.SetBinError(i,0)

    def getYields(self):
        yields = {}
        err = ctypes.c_double(0.)
        for histName in self.content.keys(): 
            yields[histName] = {}
            for group in self.content[histName].keys():
                # Check conditonials #
                condPassed = False
                for cond in self.condGroup[group]:
                    if self.checkConditional(conditions=cond,group=group,cat=histName):
                        condPassed = True
                        break
                if not condPassed:
                    continue

                # Get yields from histogram #
                h = self.content[histName][group]['nominal']
                if h.__class__.__name__.startswith('TH1'):
                    if self.include_overflow:
                        integral = h.IntegralAndError(0,h.GetNbinsX()+1,err)
                    else:
                        integral = h.IntegralAndError(1,h.GetNbinsX(),err)
                elif h.__class__.__name__.startswith('TH2'):
                    if self.include_overflow:
                        integral = h.IntegralAndError(0,h.GetNbinsX()+1,0,h.GetNbinsY()+1,err)
                    else:
                        integral = h.IntegralAndError(1,h.GetNbinsX(),1,h.GetNbinsY(),err)
                else:
                    raise ValueError
                yields[histName][group] = (integral,err.value)
        return yields

    def yieldPrintout(self):
        # Printout of yields before and after operations #
        logging.info('Yield printout : ')
        yields_after = self.getYields()
        len_groups = max([len(group) for group in self.groups.keys()]+[1])

        def printout(group,y_before,y_after):
            diff = (y_after[0]-y_before[0])/y_before[0]*100 if y_before[0] != 0 else 0.
            string = f'{group:{len_groups+3}s} : yield (before operations) = {y_before[0]:12.3f} +/- {y_before[1]:10.3f} -> yield (after) = {y_after[0]:12.3f} +/- {y_after[1]:10.3f} [{diff:+5.2f}%]'
            logging.info(f'    {string}')
            return len(string)

        for histName in self.content.keys():
            logging.info(f'   Category {histName}')
            yields_tot_before = {'mc': [0.,0.],'signal': [0.,0.]}
            yields_tot_after = copy.deepcopy(yields_tot_before)
            group_to_split = {key:[group for group in self.content[histName].keys() 
                                        if self.groups[group]['type']==key and len(self.content[histName][group])>0] 
                                                for key in yields_tot_before.keys()}
            max_length = 0
            for key in yields_tot_before.keys():
                for group in group_to_split[key]:
                    if self.groups[group]['type'] != key:
                        continue
                    y_before = self.yields[histName][group]
                    y_after  = yields_after[histName][group]
                    max_length = max(max_length,printout(group,y_before,y_after))
                    yields_tot_before[key][0] += y_before[0]
                    yields_tot_before[key][1] += y_before[1]**2
                    yields_tot_after[key][0] += y_after[0]
                    yields_tot_after[key][1] += y_after[1]**2
                logging.info(f'   {"-"*max_length}')
                yields_tot_before[key][1] = math.sqrt(yields_tot_before[key][1])
                yields_tot_after[key][1] = math.sqrt(yields_tot_after[key][1])
                printout(f"Total {key}",yields_tot_before[key],yields_tot_after[key])
                logging.info(f'   {"="*max_length}')
            if 'data_obs' in self.yields[histName].keys():
                y_before = self.yields[histName]['data_obs']
                y_after  = yields_after[histName]['data_obs']
                printout('Total data',y_before,y_after)

    def findSystematicName(self,systName,histName,group):
        if systName not in self.shapeSyst.keys():
            raise RuntimeError(f"Could not find {systName} in systematic dict")
        CMSName = self.shapeSyst[systName]

        # If syst to be discarded from the datacard #
        if CMSName == 'discard':
            logging.debug(f'Discarding systematic {systName}')
            return None

        if isinstance(CMSName,str):
            pass
        elif isinstance(CMSName,dict):
            CMSNames = [] 
            for key,values in CMSName.items():
                assert isinstance(values,dict)
                if self.checkConditional(values,group,histName):
                    CMSNames.append(key)
            if len(CMSNames) == 0:
                return None
            elif len(CMSNames) > 1:
                raise RuntimeError(f'Systematic {systName} was found with more than one match in the shape dict : {CMSNames}')
            else:
                CMSName = CMSNames[0]
        else:
            raise RuntimeError(f'Could not understand type {type(CMSName)} in shape systematics dict')
        if '{era}' in CMSName:
            CMSName = CMSName.format(era=self.era)
        return CMSName

    def saveDatacard(self):
        self.checkForMissingNominals()
        if self.outputDir is None:
            raise RuntimeError("Datacard output path is not set")

        # Save root file #
        shapes = {histName:os.path.join(self.outputDir,f"{histName}.root") for histName in self.content.keys()}
        for histName in self.content.keys():
            # Open root file #
            with TFileOpen(shapes[histName],'w') as F:
                if self.root_subdir is not None:
                    d = F.mkdir(self.root_subdir,self.root_subdir)
                    d.cd()
                for group in self.content[histName].keys():
                    if len(self.content[histName][group]) == 0:
                        continue # We have checked nominals before, can happen that empty because of conditionals
                    # Loop over systematics first to fix and save #
                    if self.use_syst:
                        for systName in self.content[histName][group].keys():
                            if self.pseudodata and group == 'data_real': # avoid writing data when using pseudodata
                                continue
                            if systName ==  "nominal": # will do afterwards
                                continue
                            if systName.endswith('Down'): # wait for the Up and do both at the same time
                                continue

                            if systName.endswith('Up'): 
                                # Get correct name #
                                systName = systName.replace('Up','')
                                CMSName = self.findSystematicName(systName,histName,group)
                                if CMSName is None:
                                    continue
                
                                systNameUp   = systName + "Up"
                                systNameDown = systName + "Down"
                                 
                                if systNameUp not in self.content[histName][group].keys():
                                    raise RuntimeError(f"Could not find syst named {systNameUp} in group {group} for histogram {histName}")
                                if systNameDown not in self.content[histName][group].keys():
                                    raise RuntimeError(f"Could not find syst named {systNameDown} in group {group} for histogram {histName}")

                                # Fix shape in case needed #
                                if group != 'data_obs' and self.fix_histograms:
                                    self.fixHistograms(hnom   = self.content[histName][group]['nominal'],
                                                        hup   = self.content[histName][group][systNameUp],
                                                        hdown = self.content[histName][group][systNameDown])

                                # Write to file #
                                CMSNameUp   = f"{group}__{CMSName}Up"
                                CMSNameDown = f"{group}__{CMSName}Down"

                                self.content[histName][group][systNameUp].SetTitle(CMSNameUp)
                                self.content[histName][group][systNameUp].SetName(CMSNameUp)
                                self.content[histName][group][systNameDown].SetTitle(CMSNameDown)
                                self.content[histName][group][systNameDown].SetName(CMSNameDown)
                
                                self.content[histName][group][systNameUp].Write(CMSNameUp) 
                                self.content[histName][group][systNameDown].Write(CMSNameDown) 
                            else:
                                raise RuntimeError(f"Something wrong happened with {systName} in group {group} for histogram {histName}")

                    # Save nominal (done after because can be correct when fixing systematics) #
                    if 'nominal' not in self.content[histName][group].keys():
                        raise RuntimeError(f"Group {group} nominal histogram {histName} was not found")
                    if group != 'data_obs' and self.fix_histograms:
                        self.fixHistograms(hnom=self.content[histName][group]['nominal'])

                    self.content[histName][group]['nominal'].SetTitle(group)
                    self.content[histName][group]['nominal'].SetName(group)
                    self.content[histName][group]['nominal'].Write(group)
                            
                F.Write()
            logging.info(f"Saved file {shapes[histName]}")

        # Save txt file #
        if self.textfiles is None:
            self.textfiles = '{}.txt'
        if '{}' not in self.textfiles:
            logging.info('Will create a single datacard txt file for all the categories')
            writer = Writer([f'{histName}' for histName in self.content.keys()])
        else:
            logging.info('Will create one datacard txt file per category')
        for histName in self.content.keys():
            binName = f'{histName}'
            if '{}' in self.textfiles:
                writer = Writer(binName)
            # Add processes #
            for group in self.content[histName]:
                if len(self.content[histName][group]) == 0:
                    continue # We have checked nominals before, can happen that empty because of conditionals
                writer.addProcess(binName       = binName,
                                  processName   = group,
                                  rate          = self.content[histName][group]['nominal'].Integral(),
                                  processType   = self.groups[group]['type'])
                # Add shape systematics #
                if self.use_syst:
                    for systName in self.content[histName][group]:
                        if systName == 'nominal':
                            continue
                        if systName.endswith('Up'):
                            systName = systName[:-2]
                            # Check integral of systematics
                            normUp = self.content[histName][group][f'{systName}Up'].Integral()
                            normDown = self.content[histName][group][f'{systName}Down'].Integral()
                            if normUp <= 0. or normDown <=0.:
                                continue

                            # Get convention naming #
                            CMSName = self.findSystematicName(systName,histName,group)
                            if CMSName is None:
                                continue
                            # Record #
                            writer.addShapeSystematic(binName,group,CMSName)
                # Add norm systematics #
                if self.use_syst and self.normSyst is not None:
                    for systName,systList in self.normSyst.items():
                        if not isinstance(systList,list):
                            systList = [systList]
                        if systName == 'footer':
                            continue
                        for systContent in systList:
                            if systContent is None:
                                raise RuntimeError(f"Problem with lnL systematic {systName}")
                            for group in self.groups.keys():
                                if self.checkConditional(systContent,group,histName):
                                    writer.addLnNSystematic(binName,group,systName,systContent['val'])
            # Add potential footer #
            if self.use_syst and self.normSyst is not None and "footer" in self.normSyst.keys():
                for footer in self.normSyst['footer']:
                    if isinstance(footer,str):
                        writer.addFooter(footer)
                    elif isinstance(footer,dict):
                        if "autoMCStats" in footer.keys():
                            conditions = {k:v for k,v in footer.items() if k not in ['autoMCStats','group']}
                            if self.checkConditional(conditions,'',histName):
                                if isinstance(footer['autoMCStats'],bool) and not footer['autoMCStats']:
                                    writer.useAutoMCStat = None
                                elif isinstance(footer['autoMCStats'],list) and not footer['autoMCStats']:
                                    writer.useAutoMCStat = footer['autoMCStats']
                                else:   
                                    raise RuntimeError(f"`autoMCStats` entry type{type(footer['autoMCStats'])} not understood")
                             
                        else:
                            conditions = {k:v for k,v in footer.items() if k not in ['line','group']}
                            if self.checkConditional(conditions,'',histName):
                                writer.addFooter(footer['line'])
                    else:   
                        raise RuntimeError(f'Footer type {type(footer)} not understood : {footer}')
                
            if '{}' in self.textfiles:
                textPath = os.path.join(self.outputDir,self.textfiles.format(f"{histName}"))
                writer.dump(textPath,os.path.basename(shapes[histName]))
                logging.info(f"Saved file {textPath}")
        if '{}' not in self.textfiles:
            textPath = os.path.join(self.outputDir,self.textfiles)
            writer.dump(textPath,[os.path.basename(shape) for shape in shapes.values()])
            logging.info(f"Saved file {textPath}")

        # Save extra data #
        if hasattr(self,'plotLinearizeData'):
            for key,data in self.plotLinearizeData.items():
                path_data = os.path.join(self.outputDir,f'{key}.json')
                with open(path_data,'w') as handle:
                    json.dump(data,handle,indent=4)
                logging.info(f"Saved plot data info in {path_data}")

    @staticmethod
    def fixHistograms(hnom,hdown=None,hup=None):
        #stats = _getStats(hnom)
        #fallback = stats[1]/stats[0] * LAMBDA0
        originalYieldNom  = hnom.Integral()
        originalYieldUp   = hup.Integral() if hup is not None else None
        originalYieldDown = hdown.Integral() if hdown is not None else None
        getVal = lambda y : 1e-5 * min(1., max(1e-10, y))
        #getErr = lambda e : max(
        val = getVal(hnom.Integral())
        valup = getVal(hup.Integral()) if hup is not None else None
        valdown = getVal(hdown.Integral()) if hdown is not None else None
        # Loop over bins #
        for i in range(1,hnom.GetNbinsX()+1):
            # First clip to 0 all negative bin content #
            if hnom.GetBinContent(i) <= 0.:
                hnom.SetBinContent(i,val)
                hnom.SetBinError(i,val) 
            if hup is not None and hup.GetBinContent(i) <= 0.:
                hup.SetBinContent(i,valup)
                hup.SetBinError(i,valup)
            if hdown is not None and hdown.GetBinContent(i) <= 0.:
                hdown.SetBinContent(i,valdown)
                hdown.SetBinError(i,valdown)
            # Second, check the up and down compared to nominal #
            if hnom.GetBinContent(i) > 0 and hdown is not None and hup is not None:
                # Nominal bin not zero #
                # Check if zero bin -> apply nominal**2 / up or down
                if hdown.GetBinContent(i) == 0. and abs(hup.GetBinContent(i)) > 0:
                    hdown.SetBinContent(i, hnom.GetBinContent(i)**2 / hup.GetBinContent(i)) 
                if hup.GetBinContent(i) == 0. and abs(hdown.GetBinContent(i)) > 0:
                    hup.SetBinContent(i, hnom.GetBinContent(i)**2 / hdown.GetBinContent(i))
                # Check if too big, deflate in case #
                if hdown.GetBinContent(i)/hnom.GetBinContent(i) > 100:
                    hdown.SetBinContent(i, 100 * hnom.GetBinContent(i))
                if hup.GetBinContent(i)/hnom.GetBinContent(i) > 100:
                    hup.SetBinContent(i, 100 * hnom.GetBinContent(i))
                # Check if too small, inflate in case #
                if hdown.GetBinContent(i)/hnom.GetBinContent(i) < 1./100:
                    hdown.SetBinContent(i, 1./100 * hnom.GetBinContent(i))
                if hup.GetBinContent(i)/hnom.GetBinContent(i) < 1./100:
                    hup.SetBinContent(i, 1./100 * hnom.GetBinContent(i))
            else:
                # Nominal is == 0 #
                if hup is not None and hdown is not None:
                    if abs(hup.GetBinContent(i)) > 0 or abs(hdown.GetBinContent(i)) > 0:
                        # zero nominal but non zero systematics -> set all at 0.00001 #
                        hnom.SetBinContent(i,val)
                        hup.SetBinContent(i,valup)
                        hdown.SetBinContent(i,valdown)
        # Rescale to keep integral consistent #
        if hnom.Integral() > 0:
            hnom.Scale(originalYieldNom/hnom.Integral())
        if hup is not None and hup.Integral() > 0:
            hup.Scale(originalYieldUp/hup.Integral())
        if hdown is not None and hdown.Integral() > 0:
            hdown.Scale(originalYieldDown/hdown.Integral())

            
    def applyCorrections(self,after_aggregation=True,after_rebinning=False):
        for corrConfig in self.histCorrections:
            if ":" not in corrConfig['module']:
                raise RuntimeError(f"`:` needs to be in the module arg {corrConfig['module']}")
            # If regrouping, order of non closure is important #
            if self.aggregation:
                if 'aggregation' in corrConfig.keys() and after_aggregation:
                    continue # Non-closure corrections that require aggregation are already done
                if not 'aggregation' in corrConfig.keys() and not after_aggregation:
                    continue # Non-closure corrections that do not require aggregation will be done in next call
            # Check rebinning #
            if 'rebinned' in corrConfig.keys() and corrConfig['rebinned']:
                if not after_rebinning:
                    continue
            else:
                if after_rebinning:
                    continue

            # Get the module and class #
            lib, clsName = corrConfig['module'].split(':')
            lib = os.path.join(os.path.abspath(os.path.dirname(__file__)),lib)
            if not os.path.isfile(lib):
                raise RuntimeError(f'File {lib} does not exist')
            spec = importlib.util.spec_from_file_location(clsName,lib)
            mod = spec.loader.load_module()
            logging.info(f"Applying {corrConfig['module']}")
            cls = getattr(mod, clsName, None)(**corrConfig['init'])
            # Loop over categories #
            for cat in self.content.keys():
                cat_no_era = cat.replace(f'_{self.era}','')
                if cat_no_era not in corrConfig['categories'].keys():
                    continue
                catCfg = corrConfig['categories'][cat_no_era]
                logging.info(f'... Histogram {cat:20s} in {cls.group} group')
                # Check if applicable #
                if cls.group is not None:
                    if isinstance(cls.group,str):
                        if cls.group not in self.content[cat].keys():
                            continue
                        groups = [cls.group]
                    elif isinstance(cls.group,list):
                        if len(set(cls.group).intersection(set(self.content[cat].keys()))) == 0:
                            continue
                        groups = cls.group
                    else:
                        raise ValueError(f'Class {clsName} attribute `group` type {type(cls.group)} not understood')
                else:
                    groups = list(self.content[cat].keys())
                # Group loop #
                for group in groups:
                    logging.info(f'\tGroup {group}')
                    # Add potential additional shapes #
                    # To be done before the correction because syst will be corrected twice otherwise
                    additional_syst = {}
                    if hasattr(cls,'additional') and self.use_syst:
                        additional_syst = cls.additional(self.content[cat][group]['nominal'],cat,group,**catCfg)
                    # Correct nominal and all syst hists #
                    if hasattr(cls,'modify'):
                        logging.info(f'\t\t-> applying corrections with key in file {catCfg["key"]}')
                        for hist in self.content[cat][group].values():
                            cls.modify(hist,cat,group,**catCfg)
                    if len(additional_syst) > 0:
                        N = max([len(key) for key in additional_syst.keys()]) + 5 
                        # Add the additional syst to the content #
                        for key,h in additional_syst.items():
                            intChange = 2*(h.Integral()-self.content[cat][group]['nominal'].Integral())/(h.Integral()+self.content[cat][group]['nominal'].Integral()+1e-9)*100
                            logging.info(f'\t\t-> Adding systematic shape {key:{N}s} [{intChange:+8.3f}%]')
                        self.content[cat][group].update(additional_syst)
            logging.info('... done')

    def applyAggregation(self):
        logging.info('Applying aggregation of categories')
        self.checkForMissingNominals()

        # Initialize and checks #
        aggregation_conv = {v:key for key,val in self.aggregation.items() for v in val}
        contentToAggregate = {}
        for histName in self.content.keys():
            if histName in aggregation_conv.keys():
                contentToAggregate[histName] = self.content[histName]
        # Crop categories to regoup from content #
        for histName in contentToAggregate.keys():
            del self.content[histName]
        innerCats = [v for val in self.aggregation.values() for v in val]
        for outCat in self.aggregation.keys():
            logging.info(f"New category {outCat} will aggregate :")
            for inCat in self.aggregation[outCat]:
                logging.info(f'... {inCat}')
        # Need to compensate missing systematics in categories to aggregate 
        # Eg if someone wants to merge electron and muon channels, there will 
        # be different systematics and we must use the nominal hist when one is missing
        list_syst = {}
        for group in self.groups.keys():
            for outCat in self.aggregation.keys():
                inCats = self.aggregation[outCat]
                list_syst = list(set([systName for cat in inCats for systName in contentToAggregate[cat][group].keys()]))
                for cat in inCats:
                    for systName in list_syst:
                        if systName != 'nominal' and systName not in contentToAggregate[cat][group].keys():
                            contentToAggregate[cat][group][systName] = copy.deepcopy(contentToAggregate[cat][group]['nominal'])
                            contentToAggregate[cat][group][systName].SetName(contentToAggregate[cat][group][systName].GetName()+f'_{systName}')
    
        # Aggregate and save as new content #
        for cat in contentToAggregate.keys():
            recat = aggregation_conv[cat] 
            if recat not in self.content.keys():
                self.content[recat] = {}
            for group in contentToAggregate[cat].keys():
                if group not in self.content[recat].keys():
                    self.content[recat][group] = {}
                for systName,h in contentToAggregate[cat][group].items():
                    if systName not in self.content[recat][group].keys():
                        self.content[recat][group][systName] = copy.deepcopy(contentToAggregate[cat][group][systName])
                    else:
                        self.content[recat][group][systName].Add(contentToAggregate[cat][group][systName])

        # Aggregate the yields #
        for outCat in self.aggregation.keys():
            inCats = self.aggregation[outCat]
            self.yields[outCat] = {}
            for cat in inCats:
                for group in self.yields[cat].keys():
                    if group not in self.yields[outCat].keys():
                        self.yields[outCat][group] = [0.,0.]
                    self.yields[outCat][group][0] += self.yields[cat][group][0]
                    self.yields[outCat][group][1] += self.yields[cat][group][1]**2
                del self.yields[cat]
            for group in self.yields[outCat].keys():
                self.yields[outCat][group][1] = math.sqrt(self.yields[outCat][group][1])
                # yield error added quadratically

        # Change back the histConverter #
        for iconf,corrConfig in enumerate(self.histCorrections):
            if 'aggregation' in corrConfig.keys():
                aggregation = corrConfig['aggregation']
                for newCat, oldCats in aggregation.items():
                    if all([oldCat in self.histConverter.keys() for oldCat in oldCats]):
                        # Aggregate #
                        self.histConverter[newCat] = [value for oldCat in oldCats for value in self.histConverter[oldCat]]
                        # Delete old inner categories #
                        for oldCat in oldCats:
                            del self.histConverter[oldCat]
                    

        # Clear for garbage collector #
        del contentToAggregate

    def checkForMissingNominals(self):
        # Check at least one nominal per group #
        missings = []
        for histName in self.content.keys():
            for group in self.content[histName]:
                # Check condition #
                condPassed = False
                for cond in self.condGroup[group]:
                    if self.checkConditional(conditions=cond,group=group,cat=histName):
                        condPassed = True
                        break
                # Check if nominal is there
                if condPassed and 'nominal' not in self.content[histName][group].keys():
                    missings.append([histName,group])
        if len(missings) != 0 :
            error_message = 'Following histograms are missing :'
            for histName,group in missings:
                error_message += f'\n... {group:20s} -> {histName}'
            raise RuntimeError(error_message)


    def applyOverflow(self):
        # Overflow bins #
        for histName in self.content.keys(): 
            for group in self.content[histName].keys():
                for h in self.content[histName][group].values():
                    if h.__class__.__name__.startswith('TH1'):
                        N = h.GetNbinsX()
                        if self.include_overflow:
                            # Include under and overflow bins
                            h.SetBinContent(1,h.GetBinContent(1)+h.GetBinContent(0))
                            h.SetBinError(1,math.sqrt(h.GetBinError(1)**2+h.GetBinError(0)**2))
                            h.SetBinContent(N,h.GetBinContent(N)+h.GetBinContent(N+1))
                            h.SetBinError(N,math.sqrt(h.GetBinError(N)**2+h.GetBinError(N+1)**2))
                        # Set under-over flow bins to zero (to avoid confusion)
                        h.SetBinContent(0,0.)
                        h.SetBinError(0,0.)
                        h.SetBinContent(N+1,0.)
                        h.SetBinError(N+1,0.)
                    elif h.__class__.__name__.startswith('TH2'):
                        Nx = h.GetNbinsX()
                        Ny = h.GetNbinsX()
                        # Loop over right and left egdes outside the TH2 #
                        for y in range(1,Ny+1):
                            if self.include_overflow:
                                h.SetBinContent(1,y,h.GetBinContent(1,y)+h.GetBinContent(0,y))
                                h.SetBinError(1,y,math.sqrt(h.GetBinError(1,y)**2+h.GetBinError(0,y)**2))
                                h.SetBinContent(Nx,y,h.GetBinContent(Nx,y)+h.GetBinContent(Nx+1,y))
                                h.SetBinError(Nx,y,math.sqrt(h.GetBinError(Nx,y)**2+h.GetBinError(Nx+1,y)**2))
                            h.SetBinContent(0,y,0.)
                            h.SetBinError(0,y,0.)
                            h.SetBinContent(Nx+1,y,0.)
                            h.SetBinError(Nx+1,y,0.)
                        # Loop over bottom and top egdes outside the TH2 #
                        for x in range(1,Nx+1):
                            if self.include_overflow:
                                h.SetBinContent(x,1,h.GetBinContent(x,1)+h.GetBinContent(x,0))
                                h.SetBinError(x,1,math.sqrt(h.GetBinError(x,1)**2+h.GetBinError(x,0)**2))
                                h.SetBinContent(x,Ny,h.GetBinContent(x,Ny)+h.GetBinContent(x,Ny+1))
                                h.SetBinError(x,Ny,math.sqrt(h.GetBinError(x,Ny)**2+h.GetBinError(x,Ny+1)**2))
                            h.SetBinContent(x,0,0.)
                            h.SetBinError(x,0,0.)
                            h.SetBinContent(x,Ny+1,0.)
                            h.SetBinError(x,Ny+1,0.)
                        # Four corners of the TH2 #
                        if self.include_overflow:
                            h.SetBinContent(1,1,h.GetBinContent(1,1)+h.GetBinContent(0,0))
                            h.SetBinError(1,1,math.sqrt(h.GetBinError(1,1)**2+h.GetBinError(0,0)**2))
                            h.SetBinContent(1,Ny,h.GetBinContent(1,Ny)+h.GetBinContent(0,Ny+1))
                            h.SetBinError(1,Ny,math.sqrt(h.GetBinError(1,Ny)**2+h.GetBinError(0,Ny+1)**2))
                            h.SetBinContent(Nx,1,h.GetBinContent(Nx,1)+h.GetBinContent(Nx+1,0))
                            h.SetBinError(Nx,1,math.sqrt(h.GetBinError(Nx,1)**2+h.GetBinError(Nx+1,0)**2))
                            h.SetBinContent(Nx,Ny,h.GetBinContent(Nx,Ny)+h.GetBinContent(Nx+1,Ny+1))
                            h.SetBinError(Nx,Ny,math.sqrt(h.GetBinError(Nx,Ny)**2+h.GetBinError(Nx+1,Ny+1)**2))
                        h.SetBinContent(0,0,0.)
                        h.SetBinError(0,0,0.)
                        h.SetBinContent(0,Ny+1,0,0.)
                        h.SetBinError(0,Ny+1,0,0.)
                        h.SetBinContent(Nx+1,0,0.)
                        h.SetBinError(Nx+1,0,0.)
                        h.SetBinContent(Nx+1,Ny+1,0.)
                        h.SetBinError(Nx+1,Ny+1,0.)
                    else:
                        raise NotImplementedError


                
    def applyEditing(self):
        logging.info('Applying editing')
        self.checkForMissingNominals()

        # Apply editing schemes #
        for histName in self.content.keys():
            # Remove era #
            cat = histName.replace(f'_{self.era}','')
            # Check name and type #
            if cat not in self.histEdit.keys():
                continue
            histType = self.content[histName][list(self.groups.keys())[0]]['nominal'].__class__.__name__ 
            if histType.startswith('TH1'):
                valid_args = ['x','i','val']
            elif histType.startswith('TH2'):
                valid_args = ['x','y','i','j','val']
            else:   
                raise RuntimeError(f'Histogram {histName} type not understood : {histType}')

            logging.info(f'Editing histogram {histName}')
            
            # Get the functions with partial #
            editFuncs = []
            for iedit,editCfg in enumerate(self.histEdit[cat]):
                if 'val' not in editCfg.keys():
                    logging.warning(f'Histogram editing for {histName} is missing the `val` key, will assume 0.')
                    editCfg['val'] = 0.
                if any([key not in valid_args for key in editCfg.keys()]):
                    raise RuntimeError('Keys not understood for config {iedit} in histogram {histName} : '+\
                                    ','.join([key not in valid_args for key in editCfg.keys()]))
                if histType.startswith('TH1'):
                    editFuncs.append(partial(self.editTH1,**editCfg))
                if histType.startswith('TH2'):
                    editFuncs.append(partial(self.editTH2,**editCfg))

            # Apply editings #
            for group in self.content[histName].keys():
                for systName,hist in self.content[histName][group].items():
                    if systName == 'nominal':
                        oldInt = hist.Integral()
                    for editFunc in editFuncs:
                        editFunc(hist)
                    if systName == 'nominal':
                        newInt = hist.Integral()
                logging.info(f'... group {group:30s} : Integral = {oldInt:9.3f} -> {newInt:9.3f} [{(oldInt-newInt)/(oldInt+1e-9)*100:5.2f}%]')

    def editTH1(self,h,val,x=None,i=None):
        assert (x is not None and i is None) or (x is None and i is not None)
        if x is not None:
            bins = [h.FindBin(x)]
        if i is not None:
            bins = [i]
        self.editHistByBinNumber(h,bins,val)

    def editTH2(self,h,val,x=None,y=None,i=None,j=None):
        assert ((x is not None or y is not None) and (i is None and j is None)) or \
               ((x is None and y is None) and (i is not None or j is not None))
        yAxis = h.GetYaxis()
        xAxis = h.GetXaxis()

        if x is not None or y is not None:
            if x is not None and y is None:
                initBin = h.FindBin(x,yAxis.GetBinCenter(1))
                bins = [initBin + k*(h.GetNbinsX()+2) for k in range(h.GetNbinsX()-1)]
            elif x is None and y is not None:
                initBin = h.FindBin(xAxis.GetBinCenter(1),y) 
                bins = [initBin + k for k in range(h.GetNbinsX())]
            else:
                bins = [h.FindBin(x,y)]
        if i is not None or j is not None:
            if i is not None and j is None:
                initBin = h.GetBin(i,1)
                bins = [initBin + k*h.GetNbinsX()+1 for k in range(h.GetNbinsX()-1)]
            elif i is None and j is not None:
                initBin = h.GetBin(1,j)
                bins = [initBin + k for k in range(h.GetNbinsX())]
            else:
                bins = [h.GetBin(i,j)]
        self.editHistByBinNumber(h,bins,val)

    @staticmethod
    def editHistByBinNumber(h,bins,val):
        assert isinstance(bins,list)
        for b in bins:
            h.SetBinContent(b,val)

    def applyCut(self):
        logging.info('Applying cut')
        self.checkForMissingNominals()

        # Apply editing schemes #
        for histName in self.content.keys():
            # Remove era #
            cat = histName.replace(f'_{self.era}','')
            # Check name and type #
            if cat not in self.histCut.keys():
                continue
            histType = self.content[histName][list(self.groups.keys())[0]]['nominal'].__class__.__name__ 
            if histType.startswith('TH1'):
                axes = ['x']
            elif histType.startswith('TH2'):
                axes = ['x','y']
            else:   
                raise RuntimeError(f'Histogram {histName} type not understood : {histType}')
            if self.histCut[cat] is None:
                continue

            # Loop through cut configs #
            logging.info(f'Cutting histogram {histName}')
            required_args = set(['axis','direction','value'])
            for icut,cutCfg in enumerate(self.histCut[cat]):
                # Safety checks #
                if len(set(cutCfg.keys()).intersection(required_args)) != len(required_args):
                    raise RuntimeError(f'Required args are {required_args}, you provided {cutCfg.keys()}')
                if cutCfg['axis'] not in axes:
                    raise RuntimeError(f"You provided axis {cutCfg['axis']}, but available values are {axes}")
                if cutCfg['direction'] not in ['>','<','<=','>=']:
                    raise RuntimeError(f"You provided diection {cutCfg['direction']}, but available values are [>,<,<=,>=]")

                # Loop through histograms #
                for group in self.content[histName].keys():
                    oldInt = self.content[histName][group]['nominal'].Integral()
                    for systName,h in self.content[histName][group].items():
                        self.content[histName][group][systName] = self.cutHistogram(h,**cutCfg)
                    newInt = self.content[histName][group]['nominal'].Integral()
                    logging.info(f'... group {group:30s} : Integral = {oldInt:9.3f} -> {newInt:9.3f} [{(oldInt-newInt)/(oldInt+1e-9)*100:5.2f}%]')

    @staticmethod
    def cutHistogram(h,axis,direction,value):
        # Get histogram content #
        nph = NumpyHist.getFromRoot(h)
        # Checks #
        if value < nph.e.min() or value > nph.e.max():
            logging.error(f"Cut at {value} on axis {axis} is outside the axis range [{nph.e.min()},{nph.e.max()}], will not do anything")
            return h
        # Find edge #
        idx_closest_edge = np.argmin(abs(value-nph.e))
        if direction == '>':
            idx_closest_edge += 1
        if direction == '<':
            idx_closest_edge -= 1
        # Split #
        if axis == 'x':
            split_nphs = nph.split(x_edges=[nph.e[idx_closest_edge]])
        if axis == 'y':
            split_nphs = nph.split(y_edges=[nph.e[idx_closest_edge]])
        # Get correct histogram #
        if '>' in direction:
            new_nph = split_nphs[1]
            newh = new_nph.fillHistogram(h.GetName()+'cut')
            newh.SetBinContent(0,
                               newh.GetBinContent(0) + split_nphs[0].sumw)
            newh.SetBinError(0,
                             np.sqrt(newh.GetBinError(0)**2 + np.sqrt(split_nphs[0].sumw2**2)))
        if '<' in direction:
            new_nph = split_nphs[0]
            newh = new_nph.fillHistogram(h.GetName()+'cut')
            newh.SetBinContent(newh.GetNbinsX()+1,
                               newh.GetBinContent(newh.GetNbinsX()+1) + split_nphs[1].sumw)
            newh.SetBinError(newh.GetNbinsX()+1,
                             np.sqrt(newh.GetBinError(newh.GetNbinsX()+1)**2 + np.sqrt(split_nphs[1].sumw2**2)))
        # return ROOT histogram #
        return newh

    def applyRebinning(self):
        assert self.rebin is not None
        logging.info('Applying rebinning')
        self.checkForMissingNominals()

        # Apply rebinning schemes #
        for histName in self.content.keys():
            histType = self.content[histName][list(self.groups.keys())[0]]['nominal'].__class__.__name__ 
            logging.info(f'... {histName:20s} ({histType})')
            if histName.replace(f'_{self.era}','') in self.rebin.keys():
                rebinSchemes = self.rebin[histName.replace(f'_{self.era}','')]
                if not isinstance(rebinSchemes,list):
                    rebinSchemes = [rebinSchemes]
                for rebinScheme in rebinSchemes:
                    if rebinScheme is None:
                        continue
                    method = rebinScheme['method']
                    params = rebinScheme['params']
                    # 1D rebinnings #
                    if method == 'classic':
                        rebinFunc = self.rebinClassic
                    elif method == 'boundary':
                        rebinFunc = self.rebinBoundary
                    elif method == 'quantile':
                        rebinFunc = self.rebinInQuantile
                    elif method == 'threshold':
                        rebinFunc = self.rebinThreshold
                    elif method == 'threshold2':
                        rebinFunc = self.rebinThreshold2
                    # 2D rebinnings #
                    elif method == 'classic2d':
                        rebinFunc = self.rebinClassic2D
                    elif method == 'boundary2d':
                        rebinFunc = self.rebinBoundary2D
                    elif method == 'quantile2d':
                        rebinFunc = self.rebinInQuantile2D
                    elif method == 'threshold2d':
                        rebinFunc = self.rebinThreshold2D
                    # 2D Linearized #
                    elif method == 'linearize2d':
                        rebinFunc = self.rebinLinearize2D
                    elif method == 'rebinlinearize2d':
                        rebinFunc = self.rebinLinearizeSplit
                    # Error #
                    else:
                        raise RuntimeError("Could not understand rebinning method {} for histogram {}".format(method,histName))
                    logging.info(f'\t -> rebinning scheme : {method}')
                    rebinFunc(histName,params)
            else:
                logging.info('\t -> rebinning scheme not requested')

    def rebinGetHists(self,histName,groups_for_binning):
        if not set(groups_for_binning).issubset(set(self.groups.keys())):
            raise RuntimeError('Groups {'+','.join([g for g in groups_for_binning if g not in self.groups.keys()])+'} are not in group dict')
        hists_for_binning = [histDict['nominal'] for group,histDict in self.content[histName].items() if group in groups_for_binning if histDict['nominal'] is not None]
        return hists_for_binning

    def countPerHistName(self,histName):
        N = 0
        for group in self.content[histName].keys():
            N += len(self.content[histName][group])
        return N
        
    def rebinClassic(self,histName,params):
        """
            Using the classic ROOT rebinning (1D)
            histName: name of histogram in keys of self.content
            params : (list) [groups(int)]
        """
        assert isinstance(params,list)
        assert len(params) == 1
        assert isinstance(params[0],int)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group,histDict in self.content[histName].items():
            for systName,hist in histDict.items():
                self.content[histName][group][systName].Rebin(params[0])
                if logging.root.level <= 10:
                    pbar.update()

    def rebinClassic2D(self,histName,params):
        """
            Using the classic ROOT rebinning (2D)
            histName: name of histogram in keys of self.content
            params : (list) [groups(int),groups(int)]
        """
        assert isinstance(params,list)
        assert len(params) == 2
        assert isinstance(params[0],int)
        assert isinstance(params[1],int)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group,histDict in self.content[histName].items():
            for systName,hist in histDict.items():
                self.content[histName][group][systName].Rebin(params[0],params[1])
                if logging.root.level <= 10:
                    pbar.update()


    def rebinBoundary(self,histName,params):
        """
            Using the given hardcoded boundaries (1D)
            histName: name of histogram in keys of self.content
            params : (list) [boundaries(list)]
        """
        from Rebinning import Boundary
        assert isinstance(params,list)
        assert len(params) == 1
        assert isinstance(params[0],list)
        boundObj = Boundary(boundaries=params[0])
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = boundObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()

    def rebinBoundary2D(self,histName,params):
        """
            Using the given hardcoded boundaries (2D)
            histName: name of histogram in keys of self.content
            params : (list) [boundaries(list),boundaries(list)]
        """
        from Rebinning import Boundary2D
        assert isinstance(params,list)
        assert len(params) == 2
        assert isinstance(params[0],list)
        assert isinstance(params[1],list)
        boundObj = Boundary2D(bx=params[0],by=params[1])
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = boundObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()


    def rebinInQuantile(self,histName,params):
        """
            Using the quantile rebinning (1D)
            histName: name of histogram in keys of self.content
            params : (list) [quantiles (list), groups (list)]
        """
        from Rebinning import Quantile
        assert isinstance(params,list)
        assert len(params) == 2
        assert isinstance(params[0],list)
        assert isinstance(params[1],list)
        quantiles = params[0]
        groups_for_binning = params[1]
        hists_for_binning = self.rebinGetHists(histName,groups_for_binning)
        qObj = Quantile(hists_for_binning,quantiles)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = qObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()

    def rebinInQuantile2D(self,histName,params):
        """
            Using the quantile rebinning (2D)
            histName: name of histogram in keys of self.content
            params : (list) [quantiles (list), quantiles (list), groups (list)]
        """
        from Rebinning import Quantile2D
        assert isinstance(params,list)
        assert len(params) == 3
        assert isinstance(params[0],list)
        assert isinstance(params[1],list)
        assert isinstance(params[2],list)
        qx = params[0]
        qy = params[1]
        groups_for_binning = params[2]
        hists_for_binning = self.rebinGetHists(histName,groups_for_binning)
        qObj = Quantile2D(hists_for_binning,qx,qy)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = qObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()


    def rebinThreshold(self,histName,params):
        """
            Using the threshold rebinning (1D)
            histName: name of histogram in keys of self.content
            params : (list) [thresholds (list), extra contributions (list | str), rsut (float), groups (list | str)]
        """
        from Rebinning import Threshold
        assert isinstance(params,list)
        assert len(params) == 4
        assert isinstance(params[0],list) 
        assert isinstance(params[1],list) or isinstance(params[1],str)
        assert isinstance(params[2],float)
        assert isinstance(params[3],list) or isinstance(params[1],str)
        thresholds          = params[0]
        extra               = params[1] # main processes to be kept above threshold
        rsut                = params[2] # relative stat. unc. threshold
        groups_for_binning  = params[3]
        hists_for_binning   = self.rebinGetHists(histName,groups_for_binning)
        hists_extra         = self.rebinGetHists(histName,extra)
        tObj = Threshold(hists_for_binning,thresholds,hists_extra,rsut)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = tObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()

    @staticmethod
    def _getStats(h):
        if h.__class__.__name__.startswith('TH1'):
            stats = np.zeros(4)
        elif h.__class__.__name__.startswith('TH2'):
            stats = np.zeros(7)
        else:
            raise NotImplementedError(f'Not understood histogram type {type(h)}')
        h.GetStats(stats)
        return stats


    def _getHistAndFallBack(self,histName,main):
        fallback            = []
        hists_for_binning   = []
        hists_signal        = None
        for group,groupCfg in self.groups.items():
            if groupCfg['type'] == 'data':
                continue
            h = self.rebinGetHists(histName,[group])[0]
            if group in main:
                stats = self._getStats(h)
                # stats[0] = sumw, stats[1] = sumw2 , rest not important
                fallback.append(stats[1]/stats[0])
                # Add to list of hists for the binning #
                hists_for_binning.append(h)
            elif groupCfg['type'] == 'signal':
                # Force each threshold bin to have at least a signal event #
                #fallback.append(np.inf)
                if hists_signal is None:
                    hists_signal = copy.deepcopy(h)
                else:
                    hists_signal.Add(h)
            else:
                # If non main background, use 0 #
                fallback.append(0.)
                # Add to list of hists for the binning #
                hists_for_binning.append(h)
        # Add the sgnal histogram to hists for binning #
        hists_for_binning.append(hists_signal)
        fallback.append(np.inf) # Force each threshold bin to have at least a signal event 
        return hists_for_binning, fallback


    def rebinThreshold2(self,histName,params):
        """
            Using the threshold rebinning (1D)
            histName: name of histogram in keys of self.content
            params : (list) [thresholds (list), main backgrounds (list)]
        """
        from Rebinning import Threshold2
        assert isinstance(params,list)
        assert len(params) == 2 or len(params) == 3
        assert isinstance(params[0],list) 
        assert isinstance(params[1],list) 
        thresholds          = params[0]
        main                = params[1]
        min_yield_per_bin   = params[2] if len(params) == 3 else 1.
        hists_for_binning, fallback = self._getHistAndFallBack(histName,main)
        # Make rebinning object and apply #
        tObj = Threshold2(hists_for_binning,thresholds,fallback,min_yield_per_bin)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = tObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()

                
    def rebinThreshold2D(self,histName,params):
        """
            Using the threshold rebinning (2D)
            histName: name of histogram in keys of self.content
            params : (list) [thresholds (list), extra contributions (list), number of bins (int), rsut (float), groups (list | str)]
            params : (list) [thresholds x (list), thresholds y (list),extra contributions (list), number of bins (int) rsut (float), groups (list)]
        """
        from Rebinning import Threshold2D
        assert isinstance(params,list)
        assert len(params) == 4
        assert isinstance(params[0],list)
        assert isinstance(params[1],list)
        assert isinstance(params[2],list) or isinstance(params[2],str)
        assert isinstance(params[3],float)
        assert isinstance(params[4],list) or isinstance(params[5],str)
        threshx             = params[0]
        threshy             = params[1]
        extra               = params[2] # main processes to be kept above threshold
        rsut                = params[3] # relative stat. unc. threshold
        groups_for_binning  = params[4]
        hists_for_binning   = self.rebinGetHists(histName,groups_for_binning)
        hists_extra         = self.rebinGetHists(histName,extra)
        tObj = Threshold(hists_for_binning,threshx,threshy,hists_extra,nbins,rsut)
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = tObj(hist) 
                if logging.root.level <= 10:
                    pbar.update()

    def rebinLinearize2D(self,histName,params):
        """
            Using the linearization of 2D hostogram
            histName: name of histogram in keys of self.content
            params : Major axis ('x' or 'y')
        """
        from Rebinning import Linearize2D
        assert isinstance(params,list)
        assert len(params) == 1
        assert params[0] in ['x','y']
        lObj = Linearize2D(major=params[0])
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = lObj(hist)
                if logging.root.level <= 10:
                    pbar.update()
        if not hasattr(self,'plotLinearizeData'):
            self.plotLinearizeData = {histName:lObj.getPlotData()}
        else:
            self.plotLinearizeData[histName] = lObj.getPlotData()
         
    def rebinLinearizeSplit(self,histName,params):
        """
            Using the linearization of 2D hostogram
            histName: name of histogram in keys of self.content
            params : Major axis ('x' or 'y')
        """
        from Rebinning import LinearizeSplit
        assert isinstance(params,list)
        assert len(params) == 6
        assert params[0] in ['x','y']       # Major binning
        assert isinstance(params[1],str)    # Name of class for major ax rebinning
        assert isinstance(params[2],list)   # Params of the major ax rebinning
        assert isinstance(params[3],str)    # Name of class for minor ax rebinning
        assert isinstance(params[4],list)   # Params of the minor ax rebinning
        assert isinstance(params[5],list)   # Groups to be used in the rebinning definition
        groups_for_binning = params[5]
        hists_for_binning  = self.rebinGetHists(histName,groups_for_binning)
        # Major #
        major_class = params[1]
        if major_class == "Boundary":
            major_params = params[2]
        elif major_class == "Quantile":
            major_params = params[2]
        elif major_class == "Threshold":
            major_params = params[2]
            major_params[2][1] = self.rebinGetHists(histName,params[2][1])
        elif major_class == "Threshold2":
            hists_for_binning, fallback = self._getHistAndFallBack(histName,params[2][1])
            major_params = [params[2][1],hists_for_binning, fallback]
            if len(params[2]) == 3:
                minor_params.append(params[2][2])
        else:
            major_class = None
            major_params = None
        # Minor #
        minor_class = params[3]
        if minor_class == "Boundary":
            minor_params = params[4]
        elif minor_class == "Quantile":
            minor_params = params[4]
        elif minor_class == "Threshold":
            minor_params = params[4]
            minor_params[1] = self.rebinGetHists(histName,params[4][1])
        elif minor_class == "Threshold2":
            hists_for_binning, fallback = self._getHistAndFallBack(histName,params[4][1])
            minor_params = [params[4][0],hists_for_binning, fallback]
            if len(params[4]) == 3:
                minor_params.append(params[4][2])
        else:
            minor_class = None
            minor_params = None

        lObj = LinearizeSplit(major        = params[0],
                              major_class  = major_class,
                              major_params = major_params,
                              minor_class  = minor_class,
                              minor_params = minor_params,
                              h            = hists_for_binning)
        
        if logging.root.level <= 10:
            pbar = enlighten.Counter(total=self.countPerHistName(histName), desc='Progress', unit='histograms')
        for group in self.content[histName].keys():
            for systName,hist in self.content[histName][group].items():
                self.content[histName][group][systName] = lObj(hist)
                if logging.root.level <= 10:
                    pbar.update()

        # Save linearized info for plotIt #
        if hasattr(lObj,'getPlotData') and lObj.getPlotData() is not None:
            if not hasattr(self,'plotLinearizeData'):
                self.plotLinearizeData = {histName:lObj.getPlotData()}
            else:
                self.plotLinearizeData[histName] = lObj.getPlotData()
    
    def saveYieldFromDatacard(self):
        missingCats = self.check_datacards()
        # Warning if not all cards #
        if len(missingCats) != 0:
            logging.warning('Missing following categories :')
            for era,cats in missingCats.items():
                logging.info(f'Era : {era}')
                for cat in cats:
                    logging.info(f'... {cat}')
            logging.warning('Will produce yield tables for the present categories, but not the combined ones')
        missingCats = [f'{cat}_{era}' for era,cats in missingCats.items() for cat in cats]

        # Output dir #
        yieldDir = os.path.join(self.outputDir,'yields')
        if not os.path.exists(yieldDir):
            os.makedirs(yieldDir)

        logging.info('Producing yield tables :')
        # Produce one yield table per cateogory #
        txtPaths = self.getTxtFilesPath()
        for cat in sorted(txtPaths.keys()):
            txtPath = txtPaths[cat]
            if cat in missingCats: 
                continue
            path_yield = os.path.join(yieldDir,f'yields_{cat}.txt')
            yield_cmd = f"cd {SETUP_DIR}; env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && yield_table.py {txtPath} --precision 5 > {path_yield}'"
            logging.info(f'... {cat}')
            rc,output = self.run_command(yield_cmd,shell=True, return_output=True)
            if rc != 0: 
                if logging.root.level > 10:
                    for line in output:
                        logging.info(line.strip())
                logging.error(f'Failed to produce yields for category {cat}, see log above')
        
        # Produce inclusive yield table #
        if len(missingCats) == 0:
            # Combine the cards #
            inclusiveTxtPath = os.path.join(yieldDir,f'datacard_{self.era}.txt')
            inclusiveYieldPath = os.path.join(yieldDir,f'yields_inclusive_{self.era}.txt')
            combine_step_success = self.combineCards(txtPaths.values(),inclusiveTxtPath)
            # Produce inclusive yield
            if combine_step_success:
                yield_cmd = f"cd {SETUP_DIR}; env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {SCRIPT_DIR} && yield_table.py {inclusiveTxtPath} > {inclusiveYieldPath}'"
                logging.info(f'Producing inclusive {self.era} yield table')
                rc,output = self.run_command(yield_cmd,shell=True, return_output=True)
                if rc != 0: 
                    if logging.root.level > 10:
                        for line in output:
                            logging.info(line.strip())
                    logging.error(f'Failed to produce yields for inclusive category, see log above')
            else:
                logging.warning('Combining cards failed, inclusive yield not computed')


    def prepare_plotIt(self):
        if self.plotIt is None:
            logging.warning('The entry `plotIt` is absent in the config, will skip the plotIt part')
            return
        # Initialize and get YAML #
        self.initialize()
        self.yaml_dict = self.loadYaml(self.path,self.yamlName)

        # Prepare root files #
        logging.info("Preparing plotIt root files for categories")
        histConverter = {f'{key}_{self.era}': val for key,val in self.histConverter.items()}

        categories = list(histConverter.keys())
        for cat in histConverter.keys():
            logging.info(f"... {cat}")

        # Initialize containers #
        content = {cat:{group:{} for group in self.groups.keys()} for cat in histConverter.keys()}
        systematics = []
        plotLinearizeData = {}

        # Make new directory with root files for plotIt #
        path_plotIt = os.path.join(self.outputDir,'plotit')
        path_rootfiles = os.path.join(path_plotIt,'root')
        if not os.path.exists(path_rootfiles):
            os.makedirs(path_rootfiles)

        # Unitary transformation #
        def unitary_binning(h):
            nh = getattr(ROOT,h.__class__.__name__)(
                    h.GetName()+'unitary',
                    h.GetTitle(),
                    h.GetNbinsX(),
                    0.,
                    h.GetNbinsX())
            bin_widths = []
            for i in range(1,h.GetNbinsX()+1):
                nh.SetBinContent(i,h.GetBinContent(i))
                nh.SetBinError(i,h.GetBinError(i))
                bin_widths.append(h.GetXaxis().GetBinWidth(i))
            return nh,bin_widths

        # Loop over root files to get content #
        datacardRoots = glob.glob(os.path.join(self.outputDir,"*root"))
        bin_widths = {}
        for f in glob.glob(os.path.join(self.outputDir,"*root")):
            category = os.path.basename(f).replace('.root','')
            logging.debug(f"Looking at {f} -> category {category}")
            if category not in content.keys():
                continue
            logging.debug('\t-> will be processed')

            with TFileOpen(f,'r') as F:
                for key in F.GetListOfKeys():
                    name = key.GetName()
                    if '__' in name: # systematic histogram
                        group,systName = name.split('__')
                        systName = systName.replace('Up','up').replace('Down','down')
                        baseSyst = systName.replace('up','').replace('down','')
                    else:   # nominal histogram
                        group = name
                        systName = 'nominal'
                        baseSyst = None
                    if group not in content[category].keys():
                        continue
                    h = F.Get(name)
                    # Rebin the histogram with unitary binning in case needed #
                    if 'unitary_bin_width' in self.plotIt.keys() and self.plotIt['unitary_bin_width']:
                        h,bin_width = unitary_binning(h)
                        if category not in bin_widths.keys():
                            bin_widths[category] = bin_width
                    # Save it in content #
                    content[category][group][systName] = copy.deepcopy(h)
                    if baseSyst is not None and baseSyst not in systematics:
                        systematics.append(baseSyst)
            path_data = f.replace('.root','.json')
            if os.path.exists(path_data):
                with open(path_data,'r') as handle:
                    plotLinearizeData[category] = json.load(handle)

        missingTxt = ""
        for cat in histConverter.keys():
            for group in self.groups.keys():
                if len(content[cat][group]) == 0:
                    missingTxt += f"\nGroup {group} in category {cat} is empty"
        if len(missingTxt) > 0:
            raise RuntimeError('There are missing histograms :'+missingTxt)

        # Write to files #
        for group in self.groups.keys():
            with TFileOpen(os.path.join(path_rootfiles,f'{group}_{self.era}.root'),'u') as F:
                for histName in content.keys():
                    for systName,hist in content[histName][group].items():
                        if hist is None:
                            continue
                        histName = histName.replace(f'_{self.era}','')
                        outName = histName if systName == "nominal" else f'{histName}__{systName}'
                        hist.Write(outName,ROOT.TObject.kOverwrite)

        # Create yaml file #
        lumi = self.yaml_dict["luminosity"][str(self.era)]
        config = {}
        config['configuration'] = {'eras'                     : [self.era],
                                   'experiment'               : 'CMS',
                                   'extra-label'              : 'Preliminary datacard',
                                   'luminosity-label'         : '%1$.2f fb^{-1} (13 TeV)',
                                   'luminosity'               : {self.era:lumi},
                                   'luminosity-error'         : 0.025,
                                   'blinded-range-fill-style' : 4050,
                                   'blinded-range-fill-color' : "#FDFBFB",
                                   'margin-bottom'            : 0.13,
                                   'margin-left'              : 0.15,
                                   'margin-right'             : 0.03,
                                   'margin-top'               : 0.05,
                                   'height'                   : 599,
                                   'width'                    : 800,
                                   'root'                     : 'root',
                                   'show-overflow'            : 'true'}

        # Compute min-max values for plots #
        def getMinNonEmptyBins(h):
            hmin = math.inf
            for i in range(1,h.GetNbinsX()+1):
                if h.GetBinContent(i) > 0:
                    hmin = min(hmin,h.GetBinContent(i))
            return hmin
            
        histMax = {}
        histMin = {}
        for histName in content.keys():
            histMax[histName] = -math.inf
            histMin[histName] = 0.1
            hstack = ROOT.THStack(histName+"stack",histName+"stack")
            for group in content[histName].keys():
                if len(content[histName][group]) == 0:
                    continue
                if self.groups[group]['type'] == 'mc':
                    hstack.Add(content[histName][group]['nominal'])
                if self.groups[group]['type'] == 'signal':
                    if 'hide' in self.groups[group].keys() and self.groups[group]['hide']:
                        continue
                    histMin[histName] = min(histMin[histName],getMinNonEmptyBins(content[histName][group]['nominal']))
                    histMax[histName] = max(histMax[histName],content[histName][group]['nominal'].GetMaximum())
            histMax[histName] = max(hstack.GetStack().Last().GetMaximum(),histMax[histName])
            histMin[histName] = min(getMinNonEmptyBins(hstack.GetStack().Last()),histMin[histName])
                # Last element of stack is the sum
        
        # Files informations #
        config['files'] = {}
        for group,gconfig in self.groups.items():
            outFile = f'{group}_{self.era}.root'
            if 'hide' in gconfig.keys() and gconfig['hide']:
                continue
            config['files'][outFile] = {'cross-section'   : 1./lumi,
                                        'era'             : str(self.era),
                                        'generated-events': 1.,
                                        'type'            : gconfig['type']}
            if gconfig['type'] != 'signal':
                plotGroup = group if 'group' not in gconfig.keys() else gconfig['group']
                config['files'][outFile].update({'group':plotGroup})
            else:
                config['files'][outFile].update({k:v for k,v in gconfig.items() if k not in ['files','type']})

        # Groups informations #
        config['groups'] = {}
        for group,gconfig in self.groups.items():
            if gconfig['type'] != 'signal':
                plotGroup = group if 'group' not in gconfig.keys() else gconfig['group']
                if plotGroup not in config['groups'].keys():
                    config['groups'][plotGroup] = {k:v for k,v in gconfig.items() if k not in ['files','type']}

        # Plots informations #
        config['plots'] = {}
        for h1,h2 in histConverter.items():
            # Check histogram dimension #
            hist = content[h1][list(self.groups.keys())[0]]['nominal']
            if isinstance(hist,ROOT.TH2F) or isinstance(hist,ROOT.TH2D):
                logging.error(f'Histogram {h1} is a TH2 and can therefore not be used in plotIt')
                continue
            # Get basic config in case not found in plots.yml
            baseCfg = {'log-y'              : 'both',
                       'x-axis'             : hist.GetXaxis().GetTitle(),
                       'y-axis'             : 'Events',
                       'y-axis-show-zero'   : True,
                       'ratio-y-axis'       : '#frac{Data}{MC}',
                       'show-ratio'         : True}
                       
            # Check if plots already in yaml #
            h1n = h1.replace(f'_{self.era}','')
            if isinstance(h2,list):
                if h2[0] in self.yaml_dict['plots'].keys():
                    config['plots'][h1n] = self.yaml_dict['plots'][h2[0]]
                else:
                    config['plots'][h1n] = baseCfg
            else:
                if h2 in self.yaml_dict['plots'].keys():
                    config['plots'][h1n] = self.yaml_dict['plots'][h2]
                else:
                    config['plots'][h1n] = baseCfg
            # Overwrite a few options #
            config['plots'][h1n]['sort-by-yields'] = True
            config['plots'][h1n]['show-overflow'] = True
            config['plots'][h1n]['x-axis-range'] = [hist.GetXaxis().GetBinLowEdge(1),hist.GetXaxis().GetBinUpEdge(hist.GetNbinsX())] 
            config['plots'][h1n]['y-axis-format'] = "%1%"
            config['plots'][h1n]['era'] = self.era
                # No Events/ [bin width] (can be misunderstood)

            # Adjust the y axis #
            if histMax[h1] > 0.:
                config['plots'][h1n]['y-axis-range'] = [0.,histMax[h1]*1.5]
                config['plots'][h1n]['log-y-axis-range'] = [max(1e-2,histMin[h1]*0.1),histMax[h1]*1000]
                config['plots'][h1n]['ratio-y-axis-range'] = [0.8,1.2]
            # Add labels #
            if 'labels' in config['plots'][h1n].keys():
                del config['plots'][h1n]['labels']
            # Modify legend #
            if 'legend' in self.plotIt.keys():
                legend = self.plotIt['legend']
                if 'position' in legend.keys():
                    assert len(legend['position']) == 4
                    config['plots'][h1n]['legend-position'] = legend['position']
                if 'columns' in legend.keys():
                    config['plots'][h1n]['legend-columns'] = legend['columns']
            # Additional linearized data #
            if len(plotLinearizeData) > 0:
                if h1 in plotLinearizeData.keys():
                    # Get margins (needed to correct the positions) #
                    margin_left = config['configuration']['margin-left']
                    margin_right = 1-config['configuration']['margin-right']
                    extraItems = plotLinearizeData[h1]
                    # Plot labels #
                    for idx in range(len(extraItems['labels'])):
                        extraItems['labels'][idx]['position'][0] = margin_left + extraItems['labels'][idx]['position'][0] * (margin_right-margin_left)
                    # Plot lines #
                    if 'unitary_bin_width' in self.plotIt.keys() and self.plotIt['unitary_bin_width']:
                        # Unitary bins, need to adapt positions of lines #
                        if h1 in bin_widths.keys():
                            original_bin_positions = np.cumsum(bin_widths[h1])
                            for idx in range(len(extraItems['lines'])):
                                # Find closest x pos in original binning -> use it as new position
                                extraItems['lines'][idx] = int(np.abs(original_bin_positions - extraItems['lines'][idx]).argmin() + 1)
                    # Draw lines #
                    for idx in range(len(extraItems['lines'])):
                        extraItems['lines'][idx] = [[extraItems['lines'][idx],0.],[extraItems['lines'][idx],histMax[h1]]]
                    config['plots'][h1n].update(extraItems)
            # If user put some options in the config, add them #
            if 'plots' in self.plotIt:
                if h1n in self.plotIt['plots']:
                    config['plots'][h1n].update(self.plotIt['plots'][h1n])
                    if 'discard' in config['plots'][h1n].values():
                        keys_to_discard = [key for key,val in config['plots'][h1n].items() if val == 'discard']
                        for key in keys_to_discard:
                            del config['plots'][h1n][key]

        # Add shape systematics
        if len(systematics) > 0 and self.use_syst:
            config['systematics'] = systematics
        # Add lnN systematics #
        if self.use_syst and self.normSyst is not None:
            pass
            # Not really possible to assign lnN to only some processes and/or only on some categories

        return config

    @staticmethod
    def merge_plotIt(configs):
        mainConfig = {'configuration':{}}
        for config in configs:
            for key in ['files','groups','plots']:
                if key not in mainConfig.keys():
                    mainConfig[key] = config[key]
                else:
                    mainConfig[key].update(**config[key])
            for key,values in config['configuration'].items():
                if key not in mainConfig['configuration'].keys():
                    mainConfig['configuration'][key] = values
                elif isinstance(values,list):
                    for val in values:
                        if val not in mainConfig['configuration'][key]:
                            mainConfig['configuration'][key].append(val)
                elif isinstance(values,dict):
                    for k,val in values.items():
                        if k not in mainConfig['configuration'][key].keys():
                            mainConfig['configuration'][key][k] = val
            if 'systematics' in config:
                if 'systematics' not in mainConfig.keys():
                    mainConfig['systematics'] = []
                for syst in config['systematics']:
                    if syst not in mainConfig['systematics']:
                        mainConfig['systematics'].append(syst)
        return mainConfig

    def run_plotIt(self,config,era):
        # Write yaml file #
        path_plotIt = os.path.join(self.outputDir,'plotit')
        path_yaml = os.path.join(path_plotIt,f'plots_{era}.yml')
        with open(path_yaml,'w') as handle:
            yaml.dump(config,handle)
        logging.info("New yaml file for plotIt : %s"%path_yaml)

        # PlotIt command #
        path_pdf = os.path.join(path_plotIt,f'plots_{era}')
        if not os.path.exists(path_pdf):
            os.makedirs(path_pdf)
        logging.info("plotIt command :")
        cmd = f"plotIt -i {path_plotIt} -o {path_pdf} -e {era} {path_yaml}"
        logging.info(cmd)
        if self.which('plotIt') is not None:
            logging.info("Calling plotIt")
            exitCode,output = self.run_command(shlex.split(cmd),return_output=True)
            if exitCode != 0:
                if logging.root.level > 10:
                    for line in output:
                        logging.info(line.strip())
                logging.info('... failure (see log above)')
            else:
                logging.info('... success')
        else:
            logging.warning("plotIt not found")

    def check_datacards(self):
        txtPaths = self.getTxtFilesPath()
        missingCats = {}
        for txtCat,txtPath in txtPaths.items():
            if not os.path.exists(txtPath):
                era = txtCat.split('_')[-1]
                cat = txtCat.replace(f'_{era}','')
                if era not in missingCats:
                    missingCats[era] = [cat]
                else:
                    missingCats[era].append(cat)
        return missingCats

    def run_combine(self,entries,additional_args={},debug=False):
        logging.info(f'Running combine on {self.outputDir}')
        if not isinstance(self.era, list):
            self.era = [self.era]
        if self.combineConfigs is None:
            logging.warning("No combine command in the yaml config file")
            return
        txtPaths = self.getTxtFilesPath()

        if entries is None or (isinstance(entries,list) and len(entries)==0):
            entries = self.combineConfigs.keys()

        # Combine setup #
        combineModes = ['significance','limits','gof','pulls_impacts','prefit','postfit_b','postfit_s']
        slurmIdPerEntry = {}
        # Loop over all the entries inside the combine configuration #
        for entry in entries:
            if entry not in self.combineConfigs.keys():
                raise RuntimeError(f'Combine entry {entry} not in combine config in the yaml file')
            combineCfg = self.combineConfigs[entry]
            combineMode = combineCfg['mode']
            slurmIdPerEntry[entry] = []

            # Make subdirectory #
            logging.info(f'Running entry {entry} with mode {combineMode}')
            if combineMode not in combineModes:
                raise RuntimeError(f'Combine mode {combineMode} not understood')
            subdir = os.path.join(self.outputDir,entry)
            if os.path.exists(subdir) and len(os.listdir(subdir)) != 0:
                logging.warning(f'Subdirectory {subdir} is not empty')
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            if 'command' not in combineCfg.keys():
                logging.warning('No "command" entry for combine mode "{combineMode}"')

            # Select txt files #
            if not 'bins' in combineCfg.keys() \
                    and (('combine_bins' in combineCfg.keys() and combineCfg['combine_bins']) \
                    or   ('split_bins' in combineCfg.keys() and combineCfg['split_bins'])):
                raise RuntimeError(f'You did not specify the bins in the config entry {entry} but used `combine_ins` or `split_bins`')
            if 'bins' in combineCfg.keys() and list(txtPaths.keys()) == ['all']:
                raise RuntimeError(f'You asked for a single datacard text file, but filter on bins for the command mode {combineMode}')

            # Additional datacards from extern #
            if 'extern' in combineCfg.keys():
                externDir = os.path.join(self.outputDir,'extern')
                if not os.path.exists(externDir):
                    os.makedirs(externDir)
                if isinstance(combineCfg['extern']['txtFiles'],dict):
                    externTxtFiles = {}
                    for key,value in combineCfg['extern']['txtFiles'].items():
                        if isinstance(value,str):
                            externTxtFiles[key] = [self.bundleCards(txtFile=value,output=externDir)]
                        elif isinstance(value,list):
                            externTxtFiles[key] = [self.bundleCards(txtFile=val,output=externDir) for val in value]
                elif isinstance(combineCfg['extern']['txtFiles'],list):
                    externTxtFiles = [self.bundleCards(txtFile=item,output=externDir) for item in combineCfg['extern']['txtFiles']]
                elif isinstance(combineCfg['extern']['txtFiles'],str):
                    externTxtFiles = [self.bundleCards(txtFile=combineCfg['extern']['txtFiles'],output=externDir)]
                else:
                    raise RuntimeError(f"Format of `extern` txtFiles {type(combineCfg['extern']['txtFiles'])} not understood")

                if 'split' in combineCfg['extern'].keys():
                    if isinstance(externTxtFiles,dict):
                        for era, txtFiles in externTxtFiles.items(): 
                            for txtFile in txtFiles:
                                cardPaths = self.splitCards(txtFile,suffix=combineCfg['extern']['split']['suffix'],era=era)
                                for binName,cardPath in cardPaths.items():
                                    newName = f"{combineCfg['extern']['split']['bins'][binName]}_{era}"
                                    cardPath = self.renameBin(txtFile=cardPath,oldBinName=binName,newBinName=newName)
                                    cardPath = self.bundleCards(txtFile=cardPath,output=self.outputDir)
                                    txtPaths[newName] = cardPath
                    if isinstance(externTxtFiles,list):
                        for txtFiles in externTxtFiles:
                            cardPaths = self.splitCards(txtFile,suffix=combineCfg['extern']['split']['suffix'])
                            for binName,cardPath in cardPaths.items():
                                binName = combineCfg['extern']['split']['bins'][binName]
                                txtPaths[binName] = self.bundleCards(txtFile=cardPath,output=self.outputDir)

            # Select bins #
            binsToUse = []
            if 'bins' in combineCfg.keys():
                combine_bins = 'combine_bins' in combineCfg.keys() and combineCfg['combine_bins']
                combine_eras = 'combine_eras' in combineCfg.keys() and combineCfg['combine_eras']
                split_bins   = 'split_bins' in combineCfg.keys() and combineCfg['split_bins'] 
                split_eras   = 'split_eras' in combineCfg.keys() and combineCfg['split_eras']
                if len(combineCfg['bins']) > 1 and not combine_bins and not split_bins:
                    raise RuntimeError('Several bins defined, please mark either `combine_bins`, `split_bins`, or both as `True`')
                if len(self.era) > 1 and not combine_eras and not split_eras:
                    raise RuntimeError('Several eras defined, please mark either `combine_eras`, `split_eras`, or both as `True`')

                # Get contributions separately #
                if len(combineCfg['bins']) > 1:
                    categories = []
                    if combine_bins:
                        categories.append(combineCfg['bins'])
                    if split_bins:
                        categories.extend([[b] for b in combineCfg['bins']])
                else:
                    categories = combineCfg['bins']
                #if 'extern' in combineCfg.keys() and len(categories) == 0:
                #    # Full extern run
                #    categories = [[]]
                if len(self.era) > 1:
                    eras = []
                    externs = []
                    if combine_eras:
                        eras.append(self.era)
                    if split_eras:
                        eras.extend([[era] for era in self.era])
                else:
                    eras = self.era

                # Combine them into bins #
                binsToUse = []
                for cat,era in itertools.product(categories,eras):
                    if not isinstance(cat,list):
                        cat = [cat]
                    if not isinstance(era,list):
                        era = [era]
                    bins = []
                    if len(cat) > 0:
                        bins = [f'{c}_{e}' for c in cat for e in era]
                    # Add extern #
                    #if 'extern' in combineCfg.keys():
                    #    if isinstance(externTxtFiles,dict):
                    #        bins.extend([f for e in era for f in externTxtFiles[e]])
                    #    else:
                    #        bins.extend(externTxtFiles)
                    binsToUse.append(bins)
            else:
                binsToUse.append('all')
            

            # Loop over all the bin combinations (can be all of the one and/or one by one) #
            subdirBinPaths = []
            for binNames in binsToUse:
                # Make bin subdir #
                if len(binsToUse) == 1:
                    subdirBin = subdir
                    binSuffix = ''
                    eras_in_bins = None
                    cats_in_bins = None
                else:
                    # Get eras and cats from bin names #
                    eras_in_bins = set()
                    cats_in_bins = set()
                    for binName in binNames:
                        for era in self.era:
                            if str(era) in binName:
                                eras_in_bins.add(str(era))
                                if not os.path.exists(binName): # Not extern datacard
                                    cats_in_bins.add(binName.replace(f'_{era}',''))

                    if len(binNames) == 1:
                        binSuffix = binNames[0]
                        if 'extern' in combineCfg.keys():
                            binSuffix = f'combination_{list(eras_in_bins)[0]}'
                    else:
                        # Check if one cat / several eras 
                        if len(cats_in_bins) == 1 and len(eras_in_bins) > 1:
                            binSuffix = f'combination_{list(cats_in_bins)[0]}'
                        # Check if several cats / one era
                        elif len(cats_in_bins) != 1 and len(eras_in_bins) == 1:
                            binSuffix = f'combination_{list(eras_in_bins)[0]}'
                        # Check if several cats / several eras
                        else:
                            binSuffix = 'combination'

                    subdirBin = os.path.join(subdir,binSuffix)
                    if not os.path.exists(subdirBin):
                        os.makedirs(subdirBin)
                    subdirBinPaths.append(subdirBin)
                
                # Check with argument (either debug or worker mode) #
                if len(binsToUse) > 1 and 'bin' in additional_args.keys():
                    if additional_args['bin'] != binSuffix:
                        continue

                # Check if root output is already in subdir #
                combinedTxtPath = os.path.join(subdirBin,'datacard.txt')
                workspacePath = os.path.join(subdirBin,'workspace.root')
                rootFiles = glob.glob(os.path.join(subdirBin,'*.root'))
                txtPathsForCombination = []
                if binNames == 'all':
                    txtPathsForCombination = list(txtPaths.values())
                else:
                    for binName in binNames:
                        if os.path.exists(binName): # Extern datacard 
                            txtPathsForCombination.append(binName)
                        elif binName not in txtPaths.keys():
                            raise RuntimeError(f'{binName} not in hist content')
                        else:
                            txtPathsForCombination.append(txtPaths[binName])
                logging.info('Running on txt files :')
                for txtPathComb in txtPathsForCombination:
                    logging.info(f'\t- {txtPathComb}')

                # Combine txt files into one #
                if not os.path.exists(combinedTxtPath):
                    logging.info('Producing combine datacard')
                    for txtPathComb in txtPathsForCombination:
                        logging.debug(f'... {txtPathComb}')
                    logging.debug(f'-> {combinedTxtPath}')
                    if os.path.exists(combinedTxtPath):
                        os.remove(combinedTxtPath)
                    self.combineCards(txtPathsForCombination,combinedTxtPath)
                    logging.info('... done')
                else:
                    logging.info(f'Found combined datacard : {combinedTxtPath}')
                # Create workspace #
                if not os.path.exists(workspacePath):
                    logging.info('Producing workspace')
                    workspaceCmd = f"cd {SETUP_DIR}; "
                    text2wpCmd = f"text2workspace.py {combinedTxtPath} -o {workspacePath}"
                    if 'physics' in combineCfg.keys():
                        assert 'model' in combineCfg['physics'].keys()
                        text2wpCmd += f" --physics-model {combineCfg['physics']['model']}"
                        if 'options' in combineCfg['physics'].keys():
                            options = combineCfg['physics']['options']
                            if not isinstance(options,list):
                                options = [options]
                            for option in options:
                                text2wpCmd += f" {option}"
                    workspaceCmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {subdirBin} && {text2wpCmd} '"
                    logging.debug(f'Running `{workspaceCmd}`')
                    exitCode,output = self.run_command(workspaceCmd,return_output=True,shell=True)
                    if exitCode != 0:
                        if logging.root.level > 10:
                            for line in output:
                                logging.info(line.strip())
                        raise RuntimeError('Could not produce the workspace, see log above')
                    logging.info('... done')
                else:
                    logging.info(f'Found workspace : {workspacePath}')

                # List systematics for the pulls and impacts #
                if combineMode == 'pulls_impacts':
                    workspaceJson = workspacePath.replace('.root','.json')
                    if not os.path.exists(workspaceJson):
                        worspace_cmd = f"cd {SETUP_DIR}; "
                        worspace_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {SCRIPT_DIR} && python helperWorkspace.py {workspacePath} {workspaceJson}'"
                        rc,output = self.run_command(worspace_cmd,shell=True,return_output=True)
                        if rc != 0: 
                            if logging.root.level > 10:
                                for line in output:
                                    logging.info(line.strip())
                            raise RuntimeError(f'Could not convert {workspacePath} into {workspaceJson}, see log above')
                        else:
                            logging.info(f'Produced {workspaceJson}')
                    else:
                        logging.info(f'Loaded {workspaceJson}')

                    with open(workspaceJson,'r') as handle:
                        workspaceParams = json.load(handle)

                    # Get systematic names for the jobs #
                    if 'mc_stats' in combineCfg.keys() and combineCfg['mc_stats']:
                        systNames = [key for key in workspaceParams.keys() if key != 'r']
                        # Move mc stats to end of list #
                        systNames.sort(key=lambda x: 'prop_bin' in x) 
                    else:
                        systNames = [key for key in workspaceParams.keys() if key != 'r' and 'prop_bin' not in key]

                        
                if self.worker or len(rootFiles) == 0 or rootFiles == [workspacePath]:
                    logging.info(f'No root file found in {subdirBin} or worker mode, will run the command for mode {combineMode}')
                    # Submit mode #
                    if 'submit' in combineCfg.keys() and not self.worker:
                        params = combineCfg['submit']
                        args = {'combine':entry}
                        arrayIds = None
                        # Create slurm directories
                        slurmDir  = os.path.join(subdirBin,'batch')
                        logDir    = os.path.join(slurmDir,'logs')
                        outputDir = os.path.join(slurmDir,'output')
                        scriptDir = os.path.join(slurmDir,'scripts')
                        for directory in [slurmDir,logDir,outputDir]:
                            if not os.path.exists(directory):
                                os.makedirs(directory)

                        n_jobs = 1
                        if combineMode == 'gof':
                            toys = combineCfg['toys-per-job']
                            n_jobs = math.ceil(combineCfg['toys']/toys) + 1 # 1 : data, >1: toys (to match array ids)
                            idxs = [_ for _ in range(1,n_jobs+1)]
                            def fileChecker(f,idx):
                                with TFileOpen(f) as F:
                                    valid = True
                                    if not F.GetListOfKeys().FindObject('limit'):
                                        valid = False
                                        logging.debug(f'Tree limit not found in {f}')
                                    else:
                                        tree = F.Get('limit')
                                        if idx == 1 and tree.GetEntries() != 1:
                                            logging.debug(f'Tree limit in {f} has {tree.GetEntries()} entries instead of 1')
                                            valid = False
                                        if idx > 1 and tree.GetEntries() != toys:
                                            logging.debug(f'Tree limit in {f} has {tree.GetEntries()} entries instead of {toys}')
                                            valid = False
                                return valid
                        elif combineMode == 'pulls_impacts':
                            n_jobs = len(systNames) + 1 # 1 = initial fit, >1: all the systematics
                            idxs = [_ for _ in range(1,n_jobs+1)]
                            if 'use_snapshot' in combineCfg.keys() and combineCfg['use_snapshot']:
                                idxs.insert(0,0)
                            def fileChecker(f,idx):
                                with TFileOpen(f) as F:
                                    valid = True
                                    if not F.GetListOfKeys().FindObject('limit'):
                                        valid = False
                                        logging.debug(f'Tree limit not found in {f}')
                                    else:
                                        tree = F.Get('limit')
                                        if tree.GetEntries() != 3 and idx>0: # 0 is for snapshot
                                            logging.debug(f'Tree limit in {f} has {tree.GetEntries()} entries instead of 3')
                                            valid = False
                                return valid
                        else:
                            n_jobs = 1
                            idxs = [1]
                            def fileChecker(f,idx):
                                return True

                        subScript = os.path.join(scriptDir,'slurmSubmission.sh')

                        if not os.path.exists(subScript):
                            logging.info(f'{entry} : submitting {n_jobs} jobs')
                            # Base args #
                            jobArrayArgs = []
                            args['worker'] = ''
                            args['yaml'] = self.configPath
                            args['era'] = self.era
                            # Mode args #
                            kwargs = {}
                            if combineMode == 'pulls_impacts':
                                kwargs['output_starts_at_zero'] = True
                            # Custom args #
                            if self.custom_args is not None:
                                args['custom'] = self.custom_args
                            # Single or multiple jobs bash script creation #
                            if n_jobs == 1:
                                if binSuffix != '':
                                    args['combine_args'] = f'bin={binSuffix}'
                                subScript = self.writeSbatchScript(slurmDir,params=params,args=args,stageoutFiles=["*root","*out"],**kwargs)
                            else:
                                for idx in idxs:
                                    subsubdir = os.path.join(outputDir,str(idx))
                                    if not os.path.exists(subsubdir):
                                        os.makedirs(subsubdir)
                                    jobArgs = {**args,'combine_args':f'idx={idx}'} 
                                    if binSuffix != '':
                                        jobArgs['combine_args'] += f' bin={binSuffix}'
                                    jobArrayArgs.append(jobArgs)
                                subScript = self.writeSbatchScript(slurmDir,params=params,args=jobArrayArgs,stageoutFiles=["*root","*out"],**kwargs)
                        else:
                            logging.info(f'{entry}: found batch script, will look for unfinished jobs')
                            arrayIds = []
                            for idx in idxs:
                                if n_jobs > 1:
                                    subsubdir = os.path.join(outputDir,str(idx))
                                else:
                                    subsubdir = outputDir
                                rootfile = glob.glob(os.path.join(subsubdir,'higgs*.root')) 
                                if len(rootfile) == 0:
                                    logging.debug(f'Root output not found in {subsubdir}')
                                    arrayIds.append(str(idx))
                                else:
                                    if not fileChecker(rootfile[0],idx):
                                        #os.remove(rootfile[0])
                                        arrayIds.append(str(idx))
                        if arrayIds is None:
                            slurmCmd = f'sbatch {subScript}'
                        else:
                            if len(arrayIds) > 0:
                                logging.info('... will resubmit the following array ids : '+','.join(arrayIds))
                                if combineMode == 'pulls_impacts':
                                    for arrayId in arrayIds:
                                        if int(arrayId) == 0: 
                                            logging.info(f'\t{arrayId:4s} -> snapshot')
                                        elif int(arrayId) == 1: 
                                            logging.info(f'\t{arrayId:4s} -> initial fit')
                                        else:
                                            logging.info(f'\t{arrayId:4s} -> {systNames[int(arrayId)-2]}')
                                slurmCmd = f"sbatch --array={','.join(arrayIds)} {subScript}"
                            else:
                                logging.info('... all jobs have succeeded')
                                slurmCmd = ''
                        if slurmCmd != '' and not debug:
                            logging.debug(f'Submitting {subScript}')
                            rc,output = self.run_command(slurmCmd,return_output=True,shell=True)
                            slurm_id = None
                            for line in output:
                                if logging.root.level > 10:
                                    logging.info(line.strip())
                                numbers = re.findall(r'\d+',line.strip())
                                if len(numbers) == 1:
                                    slurm_id = numbers[0]
                                    if len(slurm_id) != 8:
                                        slurm_id = None
                            if slurm_id is None:
                                logging.error('Slurm job id could not be found')
                            else:
                                slurmIdPerEntry[entry].append(slurm_id)
                            if arrayIds is None or len(arrayIds) == len(idxs): 
                                # For single job we want to wait until something has finished before going to finalize
                                # For array job, as soon at least one job has run, produce finalize with intermediate results
                                continue

                    # Produce command #
                    combineCmd = combineCfg['command']

                    if 'submit' in combineCfg.keys() and not self.worker: # pass to finalize part
                        combineCmd = ''
                    else:
                        if 'idx' in additional_args.keys():
                            subdirBin = os.path.join(subdirBin,'batch','output',additional_args['idx'])
                            if '--seed' not in combineCmd:
                                combineCmd += f" --seed {additional_args['idx']}"
                            if combineMode == 'gof':
                                if int(additional_args['idx']) > 1: # Toys case
                                    combineCmd += f" --toys {combineCfg['toys-per-job']}"
                            if combineMode == 'pulls_impacts':
                                if 'use_snapshot' in combineCfg.keys() and combineCfg['use_snapshot'] and int(additional_args['idx']) == 0: 
                                    # Save snapshot #
                                    if '--saveWorkspace' not in combineCmd:
                                        combineCmd += ' --saveWorkspace'
                                    if '--saveNLL' not in combineCmd:
                                        combineCmd += ' --saveNLL'
                                elif int(additional_args['idx']) == 1: # initial fit 
                                    combineCmd += " --algo singles --saveFitResult"
                                else: # One job per systematic
                                    systName = systNames[int(additional_args['idx'])-2] # Offset 0->1 (for array id) + #1 is for initial fit
                                    combineCmd += f" --algo impact -P {systName} --saveFitResult --floatOtherPOIs 1 --saveInactivePOI 1"
                                    
                                if 'use_snapshot' in combineCfg.keys() and combineCfg['use_snapshot'] and int(additional_args['idx']) > 0:
                                    # Find snapshot, or wait #
                                    attempts = 0
                                    while attempts < 60: # wait until 30 min then fails
                                        path_snapshot = glob.glob(os.path.join(os.path.dirname(subdirBin),'0','*MultiDimFit*root'))
                                        if len(path_snapshot) > 0:
                                            path_snapshot = path_snapshot[0]
                                            time.sleep(3) # If jobs start at the exact same time, maybe the file is not fully created yet
                                            # Check if workspace contained in snapshot 
                                            found_workspace = False
                                            if os.path.exists(path_snapshot):
                                                with TFileOpen(path_snapshot) as F:
                                                    if F.GetListOfKeys().FindObject("w"):
                                                        found_workspace = True
                                            if found_workspace: 
                                                break
                                        time.sleep(30) # Wait 30 seconds to see if first job has succeeded 
                                        logging.info(f'Attempt {attempts} failed to find the snapshot in {os.path.join(os.path.dirname(subdirBin),"0")} with the workspace, will wait another 30s')
                                        attempts += 1

                                    if isinstance(path_snapshot,list) or not os.path.exists(path_snapshot):
                                        raise RuntimeError(f'Problem with finding {path_snapshot}')
                                    logging.info(f'Found snapshot at {path_snapshot}')
                                    combineCmd += f' -d {path_snapshot} --snapshotName "MultiDimFit"'

                                        
                        # Unblind #
                        if 'unblind' in combineCfg.keys() and combineCfg['unblind']:
                            if '--toys' in combineCmd or '-t' in combineCmd:
                                raise RuntimeError(f'You want to unblind entry {entry} but used `--toys` in the combine command, these are exclusive')
                        else:
                            if not '--toys' in combineCmd and not '-t' in combineCmd:
                                combineCmd += " --toys -1"

                        # Add the path to workspace #
                        if '-d' not in combineCmd:
                            combineCmd += f" -d {workspacePath}"

                        # Add default args #
                        for arg in COMBINE_DEFAULT_ARGS:
                            if arg not in combineCmd:
                                combineCmd += f" {arg}"


                    if combineCmd != '':
                        fullCombineCmd  = f"cd {SETUP_DIR}; "
                        fullCombineCmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} "
                        if 'scratch' in str(subprocess.check_output('echo $LOCALSCRATCH',shell=True)) and self.worker: 
                            # If running on computing node, leave the script running on scratch area, CP3SlurmUtils will make sure to transfer
                            fullCombineCmd += f" && cd {os.getcwd()} "
                        else:
                            # Else, go work at correct output place
                            fullCombineCmd += f" && cd {subdirBin} "
                        fullCombineCmd += f" && {combineCmd}'"

                        logging.info('Extensive command is below')
                        logging.info(fullCombineCmd)

                        # Run command #
                        exitCode,output = self.run_command(fullCombineCmd,return_output=True,shell=True)
                        path_log = os.path.join(subdirBin,'log_combine.out')
                        with open(path_log,'w') as handle:
                            for line in output:
                                handle.write(line)

                        if exitCode != 0:
                            raise RuntimeError(f'Something went wrong in combine, see log in {path_log}') 
                        else:
                            logging.info('... done')
                        logging.info(f'Output directory : {subdirBin}')
                else:
                    logging.warning(f"Already a root file in subdirectory {subdirBin}, will not run the combine command again (remove and run again if needed)")

                if self.worker:
                    continue

                # Per bin mode finalize #
                # Get campaign era #
                campaign = list(eras_in_bins) if eras_in_bins is not None else None
                if campaign  is None:
                    pass
                elif len(campaign) == 1:
                    campaign = str(campaign[0])
                else:
                    campaign = 'run2'

                # Producing significance file #
                if combineMode == 'significance':
                    # Find the root file #
                    sigFile = None
                    for rootFile in rootFiles:
                        if not os.path.basename(rootFile).startswith('workspace'):
                            sigFile = rootFile
                    if sigFile is not None:
                        # Extact significance #
                        with TFileOpen(sigFile,'r') as F:
                            tree = F.Get('limit')
                            for event in tree:
                                sig = event.limit
                        path_sig = os.path.join(self.outputDir,subdirBin,'significance.json')
                        with open(path_sig,'w') as handle:
                            json.dump(sig,handle,indent=4)
                        logging.info(f'Saved significance as {path_sig}') 

                # Producing limits #
                if combineMode == 'limits':
                    # Saving as json #
                    limits = {}
                    log_file = os.path.join(subdirBin,'log_combine.out')
                    if os.path.exists(log_file):
                        with open(log_file,'r') as output:
                            for line in output:
                                l_str = None
                                r_str = None
                                if line.startswith('Observed'):
                                    logging.info("\t"+line.strip())
                                    r_str = line.split(':')[1]
                                elif line.startswith('Expected'):
                                    logging.info("\t"+line.strip())
                                    l_str,r_str = line.split(':')
                                else:
                                    continue
                                if l_str is None: 
                                    level = -1.
                                else:   
                                    level = float(re.findall("\d+.\d+",l_str)[0])
                                limits[level] = float(re.findall("\d+.\d+",r_str)[0])
                                
                        path_limits = os.path.join(self.outputDir,subdirBin,'limits.json')
                        with open(path_limits,'w') as handle:
                            json.dump(limits,handle,indent=4)
                        logging.info(f'Saved limits as {path_limits}') 

                        # Produce plots #
                        if len(limits) > 0:
                            path_plots = [os.path.join(subdirBin,'limits.pdf'),os.path.join(subdirBin,'limits.png')]
                            data = {"expected" : (limits[50.0],limits[84.0],limits[16.0],limits[97.5],limits[2.5]),
                                    "name"     : binSuffix}
                            if 'unblind' in combineCfg.keys() and combineCfg['unblind']:
                                data["observed"] = limits[-1.0],
                            content = {'paths'      : path_plots,
                                       'poi'        : 'r',
                                       'data'       : [data],
                                       'campaign'   : campaign}
                            if 'plotting' in combineCfg.keys():
                                content.update(combineCfg['plotting'])
                            path_json = os.path.join(subdirBin,'limits_plot.json')
                            logging.info(f'Saved {os.path.basename(subdirBin)} data in {path_json}')
                            with open(path_json,'w') as handle:
                                json.dump(content,handle,indent=4)

                            limit_cmd = f"cd {SETUP_DIR}; "
                            limit_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {SCRIPT_DIR} && python helperInference.py dhi.plots.limits.plot_limit_points {path_json}'"
                            rc,output = self.run_command(limit_cmd,shell=True, return_output=True)
                            if rc != 0: 
                                if logging.root.level > 10:
                                    for line in output:
                                        logging.info(line.strip())
                                logging.error('Failed to produce limit plot, see log above')
                            else:
                                for path_plot in path_plots:
                                    logging.info(f'Produced {path_plot}')
                    else:
                        logging.warning(f'File {log_file} does not exist, maybe something went wrong in the limit computation, remove the root file and run again')

                # Producing GoF #
                if combineMode == 'gof':
                    if len(additional_args) > 0:
                        continue

                    # Inspect root files to get data and toys test statistics #
                    def getLimitValues(f):
                        with TFileOpen(f) as F:
                            if F.GetListOfKeys().FindObject('limit'):
                                t = F.Get('limit')
                                limits = [event.limit for event in t]
                            else:
                                limits = []
                        return limits

                    # Hadding in case the file is not there
                    toys_file = os.path.join(subdirBin,'higgsCombine.GoodnessOfFit.toys.root')
                    data_file = os.path.join(subdirBin,'higgsCombine.GoodnessOfFit.data.root')

                    # If file is there, check if content is with expected number #
                    hadd_toys = False
                    hadd_data = False
                    if os.path.exists(toys_file):
                        limit_toys = getLimitValues(toys_file)
                        if len(limit_toys) < combineCfg['toys']:
                            hadd_toys = True
                            logging.warning(f'The gof toys tree contains {len(limit_toys)} < {combineCfg["toys"]} that you requested, will hadd again the root files')
                    else:
                        hadd_toys = True
                        limit_toys = []
                    if os.path.exists(data_file):
                        limit_data = getLimitValues(data_file)
                        if len(limit_data) != 1:
                            hadd_data = True
                            logging.warning(f'The gof data tree contains {len(limit_data)} != 1, will hadd again the root files')
                    else:
                        hadd_data = True
                        limit_data = []

                    if hadd_toys: # Either not the file, or not enough toys (maybe not all jobs had converged the first time)
                        hadd_cmd = ['hadd','-f',toys_file]
                        for rootfile in glob.glob(os.path.join(subdirBin,'batch','output','*','higgs*root')):
                            if os.path.join(subdirBin,'batch','output','1','higgs') not in rootfile: # avoid taking data measurement in toys
                                hadd_cmd.append(rootfile)
                        if len(hadd_cmd) > 3:
                            rc = self.run_command(hadd_cmd)
                            if rc != 0:
                                raise RuntimeError(f"Hadd command `{' '.join(hadd_cmd)}` failed")
                            logging.debug(f'Created {toys_file}')
                        limit_toys = getLimitValues(toys_file)
                    if hadd_data:
                        rootfile = glob.glob(os.path.join(subdirBin,'batch','output','1','*root'))
                        if len(rootfile) != 1:
                            logging.error(f'Cannot find file {rootfile.__repr__()}')
                            continue
                        shutil.copy(rootfile[0],data_file)
                        logging.debug(f'Created {data_file}')
                        limit_data = getLimitValues(data_file)

                    # Find algorithm #
                    algorithm = None
                    for i,arg in enumerate(combineCfg['command'].split()):
                        if arg == '--algo': # Used --algo ...
                            algorithm = combineCmd.split()[i+1]
                        if '--algo' in arg and arg != '--algo': # Used --algo=...
                            algorithm = arg.replace('--algo','').replace('=','')
                    if algorithm is None:
                        raise RuntimeError('Could not understand algorithm in combine command, did you use `--algo=...` or `--algo ...` ?')

                    # Produce plots #
                    if len(limit_toys) > 0:
                        path_plots = [os.path.join(subdirBin,'gof.pdf'),os.path.join(subdirBin,'gof.png')]
                        content = {'paths'      : path_plots,
                                   'data'       : limit_data[0],
                                   'toys'       : limit_toys,
                                   'algorithm'  : algorithm,
                                   'campaign'   : campaign}
                        if 'plotting' in combineCfg.keys():
                            content.update(combineCfg['plotting'])
                        path_json = os.path.join(subdirBin,'gof.json')
                        logging.info(f'Saved {os.path.basename(subdirBin)} data in {path_json}')
                        with open(path_json,'w') as handle:
                            json.dump(content,handle,indent=4)

                        gof_cmd = f"cd {SETUP_DIR}; "
                        gof_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {SCRIPT_DIR} && python helperInference.py dhi.plots.gof.plot_gof_distribution {path_json}'"
                        rc,output = self.run_command(gof_cmd,shell=True, return_output=True)
                        if rc != 0: 
                            if logging.root.level > 10:
                                for line in output:
                                    logging.info(line.strip())
                            logging.error('Failed to produce gof plot, see log above')
                        else:
                            for path_plot in path_plots:
                                logging.info(f'Produced {path_plot}')

                # Pulls and impacts #
                if combineMode == 'pulls_impacts':
                    data = {"params": []}
                    # Inspect root files to get poi values #
                    for idx in range(1,len(systNames)+2):
                        rootfiles = glob.glob(os.path.join(subdirBin,'batch','output',str(idx),'higgs*root'))
                        values = {'r':[-1,-1,-1]}
                        if idx != 1:
                            systName = systNames[idx-2]
                            values[systName] = [-1,-1,-1]
                        if len(rootfiles) != 0:
                            with TFileOpen(rootfiles[0]) as F:
                                if 'limit' in [key.GetName() for key in F.GetListOfKeys()]:
                                    tree = F.Get('limit')
                                    values['r'] = [event.r for event in tree]
                                    if idx != 1:
                                        values[systName] = [getattr(event,systName) for event in tree]
                                    if len(values['r']) != 3:
                                        values['r'] = [-9999.,-9999.,-9999.]
                                        values[systName] = [-9999.,-9999.,-9999.]
                                    
                        if idx == 1:
                            data['POIs'] = [{"name": 'r', "fit": [values['r'][1],values['r'][0],values['r'][2]]}]
                        else:
                            d = {}
                            params = workspaceParams[systName]
                            d['name']     = systName
                            d["type"]     = params['type']
                            d["groups"]   = params['groups']
                            d["prefit"]   = params['prefit']
                            d["fit"]      = [values[systName][1],values[systName][0],values[systName][2]]
                            d["r"]        = [values['r'][1],values['r'][0],values['r'][2]]
                            d["impacts"]  = {
                                'r' : [
                                        d['r'][1] - d['r'][0],
                                        d['r'][2] - d['r'][1],
                                ]
                            }
                            d["impact_r"] = max(map(abs, d["impacts"]['r']))

                            data['params'].append(d)

                    path_plots = [os.path.join(subdirBin,'pulls_impacts.pdf'), os.path.join(subdirBin,'pulls_impacts.png')]
                    content = {
                        'paths'     : path_plots,
                        'data'      : data,
                        'poi'       : 'r',
                        'campaign'  : campaign,
                    }
                    if 'plotting' in combineCfg.keys():
                        content.update(combineCfg['plotting'])
                    path_json = os.path.join(subdirBin,'pulls_impacts.json')
                    with open(path_json,'w') as handle:
                        json.dump(content,handle,indent=4)

                    pull_cmd = f"cd {SETUP_DIR}; "
                    pull_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {SCRIPT_DIR} && python helperInference.py dhi.plots.pulls_impacts.plot_pulls_impacts {path_json}'"
                    rc,output = self.run_command(pull_cmd,shell=True, return_output=True)
                    if rc != 0: 
                        if logging.root.level > 10:
                            for line in output:
                                logging.info(line.strip())
                        logging.error('Failed to produce pulls and impact plot, see log above')
                    else:
                        for path_plot in path_plots:
                            logging.info(f'Produced {path_plot}')

                # Producing prefit and postfit plots #
                if 'prefit' in combineMode or 'postfit' in combineMode:
                    fitdiagFile = glob.glob(os.path.join(subdirBin,'batch','output','fitDiagnostic*root'))
                    if len(fitdiagFile) == 0:
                        logging.error("Could not find any fitdiag file in subdir")
                        continue
                    fitdiagFile = fitdiagFile[0].replace('/auto','') 
                    # Generate dat config file #
                    def getProcesses(samples):
                        processes = {}
                        for sample, sampleCfg in samples.items():
                            if 'hide' in sampleCfg and sampleCfg['hide']:
                                continue
                            if 'group' in sampleCfg.keys():
                                group = sampleCfg['group']
                            else:
                                group = sample
                            if sampleCfg['type'] == 'data':
                                continue
                            processes[sample] = {'label'     : sampleCfg['legend'], 
                                                 'group'     : group,
                                                 'type'      : sampleCfg['type']}
                            if 'fill-color' in sampleCfg.keys():
                                processes[sample]['color'] = sampleCfg['fill-color']
                            elif 'line-color' in sampleCfg.keys():
                                processes[sample]['color'] = sampleCfg['line-color']
                            else:
                                if sampleCfg['type'] != 'data':
                                    raise RuntimeError(f'Process {sample} does not have a color, is that normal ?')
                            if 'scale' in sampleCfg.keys():
                                processes[sample]['scale'] = sampleCfg['scale']
                            if 'fill-type' in sampleCfg.keys():
                                processes[sample]['fill_style'] = sampleCfg['fill-type']
                        return processes

                    
                    if len(set(self.groups.keys()).intersection(set(self.era))) > 0:
                        # Groups are era specific 
                        processes = {}
                        set_keys = set()
                        for era, groupCfg in self.groups.items():
                            processes[era] = getProcesses(groupCfg)
                            set_keys = set_keys.union(set(processes[era].keys()))
                        for era in processes.keys():
                            if len(set_keys) != len(processes[era]):
                                raise RuntimeError(f'Era {era} has {len(processes[era])} processes, but combination has {len(set_keys)}')
                        processes = processes[self.era[0]]
                    else:
                        processes = getProcesses(self.groups)
                
                    eras_plots = [list(eras_in_bins)] if split_eras else eras
                    for plotCfg in combineCfg['plots']:
                        for era in eras_plots:
                            plotCfg.update({'fit_diagnostics_path'  : fitdiagFile,
                                            'output_path'           : subdirBin,
                                            'processes'             : processes,
                                            'eras'                  : era,
                                            'fit_type'              : combineMode})
                            # Test if categories of the plot are in the file #
                            binsInFile = set()
                            for binName in binNames:
                                for e in era:
                                    binName = binName.replace(f'_{e}','')
                                binsInFile.add(binName)
                            catsForPlot = set(plotCfg['categories'])
                            if len(binsInFile.intersection(set(plotCfg['categories']))) == 0:
                                logging.info(f"Categories for plots {plotCfg['categories']} do not match any content, will skip")
                                continue
                            # Bin width #
                            if 'keep_bin_width' in plotCfg.keys():
                                if plotCfg['keep_bin_width']:
                                    # Need to open datacard root files to get the binning #
                                    bin_edges = {}
                                    if isinstance(era,str):
                                        era = copy.deepcopy([era])
                                    if isinstance(era,list):
                                        era = copy.deepcopy(era)
                                    for e in era:
                                        bin_edges[e] = {}
                                        for cat in plotCfg['categories']:
                                            with TFileOpen(os.path.join(self.outputDir,f'{cat}_{e}.root'),'r') as F:
                                                hName = list(F.GetListOfKeys())[0].GetName() # Take the first histogram
                                                h = F.Get(hName)
                                                bin_edges[e][cat] = [round(h.GetXaxis().GetBinLowEdge(i),6) for i in range(1,h.GetNbinsX()+2)]
                                    # Take the first era bin edges #
                                    plotCfg['bin_edges'] = list(bin_edges[era[0]].values())
                                    if len(era) > 1:
                                        # Still check that bin edges are the same #
                                        for cat in plotCfg['categories']:
                                            bin_eras = [bin_edges[e][cat] for e in era]
                                            if not all ([bin_eras[0] == subbin for subbin in bin_eras[1:]]):
                                                logging.warning(f'In category {cat}, bin edges vary -> this might produce weird plots')
                                                for e,bin_era in zip(era,bin_eras):
                                                    logging.warning(f'... {era} -> bin edges = {bin_era}')
                                del plotCfg['keep_bin_width']
                            if logging.root.level <= 10:
                                plotCfg['verbose'] = True
                            if 'logy' in plotCfg['plot_options'].keys() and plotCfg['plot_options']['logy'] == 'both':
                                plotCfg['plot_options']['logy'] = False
                                PostfitPlots(**plotCfg)
                                plotCfg['plot_options']['logy'] = True
                                PostfitPlots(**plotCfg)
                            else:
                                PostfitPlots(**plotCfg)


                    # Nuisances likelihoods #
                    if 'postfit' in combineMode and ('nuisance_likelihood_scan' not in combineCfg.keys() or combineCfg['nuisance_likelihood_scan']):
                        fit_name = None
                        if combineMode == 'postfit_b':
                            fit_name = 'fit_b'
                        if combineMode == 'postfit_s':
                            fit_name = 'fit_s'
                        path_plots = [os.path.join(subdirBin,'nuisances.pdf'), os.path.join(subdirBin,'nuisances.png')]
                        resultFile = glob.glob(os.path.join(subdirBin,'higgs*root'))
                        if len(resultFile) == 0:
                            raise RuntimeError(f'Could not find result file in {subdirBin} for nuisance likelihood scan')

                        content = {
                            'paths'                 : path_plots,
                            'poi'                   : 'r',
                            'fit_name'              : fit_name,
                            'workspace'             : resultFile[0],
                            'dataset'               : resultFile[0],
                            'fit_diagnostics_path'  : fitdiagFile,
                            'show_derivatives'      : True,
                            'y_min'                 : 0.,
                            'y_max'                 : 5.,
                        }
                        path_json = os.path.join(subdirBin,'nuisances.json')
                        with open(path_json,'w') as handle:
                            json.dump(content,handle,indent=4)
                        getter = {
                            'workspace'     : {'type':'ROOT','name':'w'},
                            'dataset'       : {'type':'ROOT','name':'toys/toy_asimov'},
                        }
                        path_getter = os.path.join(subdirBin,'nuisances_getter.json')
                        with open(path_getter,'w') as handle:
                            json.dump(getter,handle,indent=4)

                        nuisance_cmd = f"cd {SETUP_DIR}; "
                        nuisance_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && cd {SCRIPT_DIR} && python helperInference.py dhi.plots.likelihoods.plot_nuisance_likelihood_scans {path_json} {path_getter}'"
                        logging.info("Running nuisance likelihood plots")
                        rc,output = self.run_command(nuisance_cmd,shell=True, return_output=True)
                        if rc != 0: 
                            if logging.root.level > 10:
                                for line in output:
                                    logging.info(line.strip())
                            logging.error('Failed to produce nuisance plots, see log above')
                        else:
                            for path_plot in path_plots:
                                logging.info(f'Produced {path_plot}')

            # Combined finalize mode for all bins (in case there was) #
            if len(subdirBinPaths) > 1 and not self.worker:
                # Significance : combine into one file #
                if combineMode == 'significance':
                    content = {}
                    for subdirBinPath in subdirBinPaths:
                        path_json = os.path.join(subdirBinPath,'significance.json')
                        if not os.path.exists(path_json):
                            logging.warning(f'Could not load {path_json}, will continue')
                            continue
                        logging.info(f'Loading {path_json}')
                        with open(path_json,'r') as handle:
                            subCont = json.load(handle)
                            content[os.path.basename(subdirBinPath)] = subCont
                    # Save combined content #
                    with open(os.path.join(subdir,'significance.json'),'w') as handle:
                        json.dump(content,handle)
                    logging.info(f"Saved significance summary in {os.path.join(subdir,'significance.json')}")
                # Limits : combine into one plot #
                if combineMode == 'limits':
                    # Produce plots #
                    path_plots = [os.path.join(subdir,'limits.pdf'),os.path.join(subdir,'limits.png')]

                    content = {'paths'      : path_plots,
                               'poi'        : 'r',
                               'data'       : [],
                               'campaign'   : 'run2'}
                    for subdirBinPath in subdirBinPaths:
                        path_json = os.path.join(subdirBinPath,'limits_plot.json')
                        if not os.path.exists(path_json):
                            logging.warning(f'Could not load {path_json}, will continue')
                            continue
                        logging.info(f'Loading {path_json}')
                        with open(path_json,'r') as handle:
                            subCont = json.load(handle)
                            content['data'].extend(subCont['data'])
                    if 'plotting' in combineCfg.keys():
                        content.update(combineCfg['plotting'])
                    path_json = os.path.join(subdir,'limits_plot.json')
                    logging.info(f'Saved {os.path.basename(subdirBin)} data in {path_json}')
                    with open(path_json,'w') as handle:
                        json.dump(content,handle,indent=4)
                    limit_cmd = f"cd {SETUP_DIR}; "
                    limit_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {ULIMIT} && cd {SCRIPT_DIR} && python helperInference.py dhi.plots.limits.plot_limit_points {path_json}'"
                    rc,output = self.run_command(limit_cmd,shell=True, return_output=True)
                    if rc != 0: 
                        if logging.root.level > 10:
                            for line in output:
                                logging.info(line.strip())
                        logging.error('Failed to produce limit plot, see log above')
                    else:
                        for path_plot in path_plots:
                            logging.info(f'Produced {path_plot}')

                # Goodness of fit : combine into one plot
                if combineMode == 'gof':
                    logging.info('GoF combination :')
                    path_plots = [os.path.join(subdir,'gof.pdf'),os.path.join(subdir,'gof.png')]
                    content = {'paths'      : path_plots,
                               'n_bins'     : 32,
                               'data'       : [],
                               'algorithm'  : 'saturated',
                               'campaign'   : 'run2'}
                    for subdirBinPath in subdirBinPaths:
                        path_json = os.path.join(subdirBinPath,'gof.json')
                        if not os.path.exists(path_json):
                            logging.warning(f'Could not load {path_json}, will continue')
                            continue
                        logging.info(f'Loading {path_json}')
                        with open(path_json,'r') as handle:
                            subCont = json.load(handle)
                        if len(subCont['toys']) > 0:
                            content['data'].append({'data' : subCont['data'],
                                                    'toys' : subCont['toys'], 
                                                    'name' : os.path.basename(subdirBinPath)})
                    
                    path_json = os.path.join(subdir,'gof.json')
                    logging.info(f'Saved combination data in {path_json}')
                    with open(path_json,'w') as handle:
                        subCont = json.dump(content,handle,indent=4)

                    if len(content['data']) > 0:
                        gof_cmd = f"cd {SETUP_DIR}; "
                        gof_cmd += f"env -i bash -c 'source {SETUP_SCRIPT} && cd {SCRIPT_DIR} && python helperInference.py dhi.plots.gof.plot_gofs {path_json}'"
                        rc,output = self.run_command(gof_cmd,shell=True, return_output=True)
                        if rc != 0: 
                            if logging.root.level > 10:
                                for line in output:
                                    logging.info(line.strip())
                            logging.error('Failed to produce gof plot, see log above')
                        else:
                            for path_plot in path_plots:
                                logging.info(f'Produced {path_plot}')

        # Slurm ID printout #
        if len([v for key,val in slurmIdPerEntry.items() for v in val]) > 0: # If any slurm Id has been registered
            logging.info('Following slurm job ids were submitted')
            for entry,slurm_ids in slurmIdPerEntry.items():
                logging.info(f'... Entry {entry} : '+' '.join(slurm_ids))

        return []

    def getTxtFilesPath(self):
        initTextFiles = {}
        if not isinstance(self.era,list) and not isinstance(self.era,tuple):
            eras = [self.era]
        else:
            eras = self.era

        for era in eras:
            keys = [str(key) for key in self.histConverter.keys()]
            if len(set(self.era).intersection(set(keys)))>0: # keys are eras
                histConverter = self.histConverter[list(self.histConverter.keys())[keys.index(era)]]
            else: # keys are not eras
                histConverter = self.histConverter
            if self.textfiles is None:
                initTextFiles.update(**{f'{histName}_{era}':os.path.join(self.outputDir,f'{histName}_{era}.txt') 
                                        for histName in histConverter.keys()})
            elif isinstance(self.textfiles,str):
                if '{}' in self.textfiles:
                    initTextFiles.update(**{f'{histName}_{era}':os.path.join(self.outputDir,
                                                        self.textfiles.format(f'{histName}_{era}.txt'))
                                                for histName in self.histConverter.keys()})
                else:
                    initTextFiles.update(**{f'all_{era}',os.path.join(self.outputDir,self.textfiles)})
            else:
                raise RuntimeError('Format of "textfiles" entry not understood')
        for key,val in initTextFiles.items(): # Crrect in case the user put the `.txt` in the textFiles
            if val.count('.txt') > 1:
                initTextFiles[key] = val.replace('.txt','') + '.txt'

        return initTextFiles

    def combineCards(self,initTextPaths,outputTextPath):
        for initTextPath in initTextPaths:
            if not os.path.exists(initTextPath):
                raise RuntimeError(f'File {initTextPath} not found')
        if len(initTextPaths) == 0:
            raise RuntimeError('No initial txt paths')
        # combine the datacards #
        combineCmd = "combineCards.py "
        for initTextPath in initTextPaths:
            if 'auto' in initTextPath:
                initTextPath = initTextPath.replace('/auto','')
            binName = os.path.basename(initTextPath).replace('.txt','')
            combineCmd += f" {binName}={os.path.relpath(initTextPath,os.path.dirname(outputTextPath))}"
                # we use relative path so it can be later exported to other filesystem
        fullCombineCmd  = f"cd {SETUP_DIR}; "
        fullCombineCmd += f"env -i bash -c 'source {SETUP_SCRIPT} && cd {os.path.dirname(outputTextPath)} && {combineCmd}'"
        logging.debug(f'Running command `{fullCombineCmd}`')
        rc, output = self.run_command(fullCombineCmd,return_output=True,shell=True)
        if rc != 0:
            logging.error(f'combineCards failed with {outputTextPath}')
        with open(outputTextPath,'w') as handle:
            for line in output:
                handle.write(line)
        return rc == 0

    def bundleCards(self,txtFile,output,**kwargs):
        if not os.path.exists(txtFile):
            raise RuntimeError(f'File {txtFile} not found')
        basename = os.path.basename(txtFile)
        if os.path.exists(os.path.join(output,basename)):
            logging.info(f'\t{basename} is already bundled into {output}')
            return os.path.join(output,basename)
        bundleCmd = f"bundle_datacard.py {txtFile} {output}"
        for argName,argVal in kwargs.items():
            bundleCmd += f' --{argName}={argVal}'
        fullCombineCmd  = f"cd {SETUP_DIR}; "
        fullCombineCmd += f"env -i bash -c 'source {SETUP_SCRIPT} && {bundleCmd}'"
        logging.debug(f'Running command `{fullCombineCmd}`')
        logging.info(f'\tBundling card {basename} into {output}')
        rc, output = self.run_command(fullCombineCmd,return_output=True,shell=True)
        if rc != 0:
            logging.error(f'bundle_datacard.py failed, log below')
        for line in output:
            if rc != 0:
                logging.info(line.strip())
            if 'bundled datacard' in line:
                return line.split()[-1] 

    def splitCards(self,txtFile,suffix,era=None,**kwargs):
        if not os.path.exists(txtFile):
            raise RuntimeError(f'File {txtFile} not found')
        basename = os.path.basename(txtFile)
        dirname = os.path.dirname(txtFile)
        # Make pattern #
        if era is None:
            pattern = f'{suffix}_{{}}.txt'
            cardPaths = glob.glob(os.path.join(dirname,f'{suffix}*txt'))
        else:
            pattern = f'{suffix}_{{}}_{era}.txt'
            cardPaths = glob.glob(os.path.join(dirname,f'{suffix}*{era}*txt'))
        # Get number of split files in directory and expected from txt file #
        N_bins = 0
        with open(txtFile,'r') as handle:
            for line in handle:
                if line.startswith('imax'):
                    N_bins = int(line.split()[1])
                    break
        if N_bins <= 0:
            raise RuntimeError(f'N bins = {N_bins} <= 0 in file {txtFile}')
        # If all split files already there -> get their names #
        if len(cardPaths) == N_bins:
            logging.info(f'\t{txtFile} is already split')
        # If not -> run the splitting #
        else:
            # Do the splitting #
            splitCmd = f"split_datacard_by_bins.py {os.path.basename(txtFile)} --pattern {pattern}"
            for argName,argVal in kwargs.items():
                splitCmd += f' --{argName}={argVal}'
            fullCombineCmd  = f"cd {SETUP_DIR}; "
            fullCombineCmd += f"env -i bash -c 'source {SETUP_SCRIPT} && cd {dirname} && {splitCmd}'"
            logging.debug(f'Running command `{fullCombineCmd}`')
            logging.info(f'\tSplitting card {basename}')
            rc, output = self.run_command(fullCombineCmd,return_output=True,shell=True)
            if rc != 0:
                logging.error(f'split_datacard_by_bins.py failed, log below')
            cardPaths = []
            for line in output:
                if rc != 0:
                    logging.info(line.strip())
                if 'split_datacard_by_bins - removing all bins' in line:
                    cardPaths.append(line.split()[-1])
        # Recover card paths #
        return {parse.parse(pattern,os.path.basename(cardPath))[0]:cardPath for cardPath in cardPaths}
                

    def renameBin(self,txtFile,oldBinName,newBinName):
        # Change file name #
        newFile = os.path.join(os.path.dirname(txtFile),f'{newBinName}.txt')
        if os.path.exists(newFile):
            logging.info(f'\tRenamed card {newFile} already exists')
            return newFile
        # Load content #
        logging.info(f'\tRenaming card {txtFile} into {os.path.basename(newFile)}')
        with open(txtFile,'r') as handle:
            content = handle.read()
        # Edit content and write #
        with open(newFile,'w') as handle:
            for line in content.split('\n'):
                if line.startswith('shapes'):
                    line = line.split()
                    line[2] = line[2].replace(oldBinName,newBinName) # replace only the binname and not the rest
                    line = '  '.join(line)
                if line.startswith('bin') or 'autoMCStats' in line:
                    line = line.replace(oldBinName,newBinName)
                handle.write(line+'\n') 
        # return #
        return newFile

    @staticmethod
    def writeSbatchScript(mainDir,params={},args={},stageoutFiles=[],output_in_main=False,output_starts_at_zero=False):
        # Make arguments easier to handle #
        if isinstance(args,dict):
            args = [args]
        # Make default slurm config #
        config = Configuration()
        config.scratchDir = "${LOCALSCRATCH}"
        config.sbatch_partition = 'cp3'
        config.sbatch_qos = 'cp3'
        config.sbatch_chdir = os.path.dirname(os.path.abspath(__file__))
        config.inputSandboxContent = ["*.h"]
        config.sbatch_time = '0-02:00:00'
        config.sbatch_memPerCPU = '2000'
        config.sbatch_additionalOptions = ["--export=ALL"]
        config.useJobArray = True 
        config.inputParamsNames = []
        config.inputParams = []
        config.stageout = True
        config.stageoutFiles = stageoutFiles
        #config.writeLogsOnWN = False

        # Make paths #
        config.inputSandboxDir = mainDir
        config.batchScriptsDir = os.path.join(mainDir,'scripts')
        if output_in_main:
            config.stageoutDir = mainDir
        elif len(args) > 1:
            if output_starts_at_zero:
                config.stageoutDir = os.path.join(mainDir,'output','$((${SLURM_ARRAY_TASK_ID}-1))')
            else:
                config.stageoutDir = os.path.join(mainDir,'output','${SLURM_ARRAY_TASK_ID}')
        else:
            config.stageoutDir = os.path.join(mainDir,'output')
        config.stageoutLogsDir       = os.path.join(mainDir, 'logs')   
        config.batchScriptsFilename  = 'slurmSubmission.sh'

        # Edit slurm parameters #
        for k,v in params.items():
            if 'time' in k:
                config.sbatch_time = str(v)
            elif 'mem' in k:
                config.sbatch_memPerCPU = str(v)
            else:
                config.sbatch_additionalOptions.append(f'--{k}={v}')
            
        # Make payload and parameters #
        config.inputParamsNames = ['payload']

        scriptName = os.path.abspath(__file__).replace('/auto','')
        config.payload = "${payload}"

        for arg in args:
            payload = f"python3 {scriptName}" 
            for argName,argVal in arg.items():
                if argVal is None:
                    continue
                elif isinstance(argVal,bool) and not argVal:
                    continue
                elif isinstance(argVal,str) and os.path.exists(argVal):
                    argVal = os.path.abspath(argVal)
                payload += f" --{argName} "
                if isinstance(argVal,list) or isinstance(argVal,tuple):
                    for v in argVal:
                        if isinstance(v,str) and os.path.exists(v):
                            v = os.path.abspath(v).replace('/auto','')
                        payload += f'{v} '
                elif isinstance(argVal,bool):
                    pass
                else:
                    payload += str(argVal)
            config.inputParams.append([payload])
    
        f = io.StringIO()
        with redirect_stdout(f): # Catch verbose output
            # Initialize submission #
            submitWorker = SubmitWorker(config, submit=False, debug=True, quiet=True)

            # Run creation of bash script #
            submitWorker()

        out = f.getvalue()
        # Save verbose output to file #
        with open(os.path.join(config.batchScriptsDir, 'submission_log.txt'),'w') as handle:
            handle.write(out)

        return os.path.join(config.batchScriptsDir, config.batchScriptsFilename)

 
    @staticmethod
    def run_command(command,return_output=False,**kwargs):
        process = subprocess.Popen(command,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,**kwargs)
        # Poll process for new output until finished #
        output = []
        while True:
            try:
                nextline = process.stdout.readline()
            except UnicodeDecodeError:
                continue
            if nextline == '' and process.poll() is not None:
                break
            logging.debug(nextline.strip())
            if return_output:
                output.append(nextline)
        process.communicate()
        exitCode = process.returncode
        if return_output:
            return exitCode,output
        else:
            return exitCode


    @staticmethod
    def which(program):
        import os
        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    
        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
    
        return None


def parseYaml(yamlPath,custom=None):
    if custom is not None:
        # Custom command line argument for parsing #
        formatting = {}
        for arg in custom:
            if '=' in arg:
                formatting[arg.split('=')[0]] = arg.split('=')[1]
            else:
                logging.warning(f'`--custom {arg}` will be ignored because no `=`')
        config = yaml.load({'filename':yamlPath,'formatting':formatting},
                      Loader=YMLIncludeLoader)
    else:
        # Classic parse #
        with open(yamlPath,'r') as handle:
            config = yaml.load(handle,Loader=YMLIncludeLoader)
    return config
 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Produce datacards')
    parser.add_argument('--yaml', action='store', required=True, type=str,
                        help='Yaml containing parameters')
    parser.add_argument('--era', action='store', required=False, type=str, default=None, nargs='*',
                        help='List of eras to restrict production to')
    parser.add_argument('--pseudodata', action='store_true', required=False, default=False,
                        help='Whether to use pseudo data (data = sum of MC) [default = False]')
    parser.add_argument('--plotIt', action='store_true', required=False, default=False,
                        help='Browse datacard files and produce plots via plotIt \
                              (note : done by default when datacards are produced, except in submit mode)') 
    parser.add_argument('--yields', action='store_true', required=False, default=False,
                        help='Browse datacard files and produce yields from the txt datacards')
    parser.add_argument('--split', action='store', required=False, default=None, nargs='*',
                        help='If used without argument, will process all categories in serie, \
                              otherwise will process the categories given in argument')
    parser.add_argument('-j,','--jobs', action='store', required=False, default=None, type=int,
                        help='Number of parallel processes (only useful with `--split`) [-1 = number of processes]')
    parser.add_argument('--submit', action='store', required=False, default=None, nargs='*',
                        help="""Whether to submit the production to the cluster, \n
                                 combined with `--split` will submit one job per category otherwise will submit a single job, \n
                                 combined with `--j` will submit a multithreaded job, \n
                                 used `--split time=... and/or `--split mem-per-cpu=... to modify the default values (or any sbatch arg)""")
    parser.add_argument('--combine', action='store', required=False, default=None, nargs='*',
                        help='Run combine on the txt datacard only')
    parser.add_argument('--combine_args', action='store', required=False, default=[], nargs='*',
                        help='Additional args for the combine commands (used in jobs, keep for debug)')
    parser.add_argument('--custom', action='store', required=False, default=None, nargs='*',
                        help='Format the yaml file')
    parser.add_argument('--interpolation', action='store', required=False, default=None, nargs='*',
                        help='Config for the matching interpolation')
    parser.add_argument('--interpolation_config1', action='store', required=False, default=None, nargs='*',
                        help='First config for the interpolation (to be used with `--interpolation`)')
    parser.add_argument('--interpolation_config2', action='store', required=False, default=None, nargs='*',
                        help='Second config for the interpolation (to be used with `--interpolation`)')
    parser.add_argument('--interpolation_era', action='store', required=False, default=None, type=str, nargs='*',
                        help='Era for the interpolation [default : done for each era in the initial config] (to be used with `--interpolation`)')
    parser.add_argument('-v','--verbose', action='store_true', required=False, default=False,
                        help='Verbose mode')
    parser.add_argument('--worker', action='store_true', required=False, default=False,
                        help='Force working locally')
    parser.add_argument('--debug', action='store_true', required=False, default=False,
                        help='Do not send jobs')
    args = parser.parse_args()

    logging.basicConfig(level   = logging.DEBUG,
                        format  = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S')

    # Verbose level #
    if not args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # File checking #
    if args.yaml is None:
        raise RuntimeError("Must provide the YAML file")
    if not os.path.isfile(args.yaml):
        raise RuntimeError("YAML file {} is not a valid file".format(args.yaml))

    # Yaml parsing #
    configMain = parseYaml(args.yaml,args.custom)

    # Content checks #
    required_items = ['path','outputDir','yamlName','histConverter','groups','era']
    if any(item not in configMain.keys() for item in required_items): 
        raise RuntimeError('Your configMain is missing the following items : '+ \
                ','.join([item for item in required_items if item not in configMain.keys()]))

    # Create output directory #
    outputDir = configMain['outputDir']
    if not os.path.isabs(outputDir):
        outputDir = os.path.join(os.path.abspath(os.path.dirname(__file__)),outputDir)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    configMain['outputDir'] = outputDir
    plotIt_path = os.path.join(outputDir,'plotit','root')
    logging.info(f"Output path : {outputDir}")

    # Eras #
    eras = configMain['era']
    if isinstance(configMain['era'],list):
        eras = [str(era) for era in configMain['era']]
    else:
        eras = [str(configMain['era'])]
    if args.era is not None:
        if len(set(args.era) - set(eras)) > 0:
            raise RuntimeError(f'You asked for eras {",".join(args.era)} while the eras in the configs are {",".join(eras)}')
        eras = args.era
        configMain['era'] = eras
        

    # Interpolation #
    if args.interpolation is not None:
        # Check for the other args #
        if args.interpolation_config1 is None:
            raise RuntimeError("You need to also provide `--interpolation_config1`")
        if args.interpolation_config2 is None:
            raise RuntimeError("You need to also provide `--interpolation_config2`")
        # Check for formatting in the arg #
        if len(args.interpolation) > 1:
            configInt = parseYaml(args.interpolation[0],custom=args.interpolation[1:])
        else:
            configInt = parseYaml(args.interpolation[0])
        if len(args.interpolation_config1) > 1:
            config1 = parseYaml(args.interpolation_config1[0],custom=args.interpolation_config1[1:])
        else:
            config1 = parseYaml(args.interpolation_config1[0])
        if len(args.interpolation_config2) > 1:
            config2 = parseYaml(args.interpolation_config2[0],custom=args.interpolation_config2[1:])
        else:
            config2 = parseYaml(args.interpolation_config2[0])

        configPartial = copy.deepcopy(configMain) # What is to be used as main for interpolation
        # Remove steps not necessary for config1 and config2 #
        # (will be done only for the interpolated one)
        to_remove = ['rebin']
        for key in to_remove:
            for conf in [config1,config2,configPartial]:
                if key in conf.keys():
                    del conf[key]
        # Cross checks #
        # (make sure content is common to both main and the configs)
        to_check = []
        for key in to_check:
            if set(config1[key]) != set(config2[key]) or set(config1[key]) != set(configPartial[key]) or set(config2[key]) != set(configPartial[key]):
                raise RuntimeError(f'Need same content key {key} between the main config and the two configs for the interpolation')
        # Equalization #
        # (set some content to two configs from the main)
        to_equalize  = ['outputDir','era']
        for key in to_equalize:
            for conf in [config1,config2,configPartial]:
                conf[key] = configMain[key] 
        # Avoid saving datacards for the two config #
        # (only done on the fly and used for interpolation) #
        config1['save_datacard'] = False
        config2['save_datacard'] = False
        configPartial['save_datacard'] = False
        if 'plotIt' in config1:
            del config1['plotIt'] 
        if 'plotIt' in config2:
            del config2['plotIt']
        if 'plotIt' in configPartial:
            del configPartial['plotIt']
        # Use the correct groups for interpolation #
        def editGroups(config,groups_to_keep=[],groups_to_remove=[],eras=None):
            groups = Datacard.includeEntry(config['groups'],'groups',eras)
            if eras is not None and len(set(groups.keys())-set(eras))==0:
                groupsNew = dict()  
                for era,groupEntries in groups.items():
                    groupsNew[era] = dict()
                    for group,groupCfg in groupEntries.items():
                        if (len(groups_to_keep)>0 and group in groups_to_keep) \
                                or (len(groups_to_remove)>0 and group not in groups_to_remove):
                            groupsNew[era][group] = groupCfg
                    if len(groupsNew[era]) == 0:
                        raise RuntimeError(f'Empty groups in era {era} -> to keep = ['+','.join(groups_to_keep)+'], to remove = ['+','.join(groups_to_remove)+'], but initial ['+','.join([str(group) for group in groups[era].keys()])+']')
            else:
                groupsNew = dict()
                for group,groupCfg in groups.items():
                    if (len(groups_to_keep)>0 and group in groups_to_keep) \
                            or (len(groups_to_remove)>0 and group not in groups_to_remove):
                        groupsNew[group] = groupCfg
                if len(groupsNew) == 0:
                    logging.warning('Empty groups -> to keep = ['+','.join(groups_to_keep)+'], to remove = ['+','.join(groups_to_remove)+'], but initial ['+','.join([str(group) for group in groups.keys()])+']')
            config['groups'] = groupsNew

        groups1_to_keep = []            
        groups2_to_keep = []            
        groupsInt_to_remove = []            
        for groupInt, groupPair in configInt['matchingGroup'].items():
            groups1_to_keep.append(groupPair[configInt['param1']])
            groups2_to_keep.append(groupPair[configInt['param2']])
            groupsInt_to_remove.append(groupInt)
        editGroups(config1,groups_to_keep=groups1_to_keep,eras=eras)
        editGroups(config2,groups_to_keep=groups2_to_keep,eras=eras)
        editGroups(configPartial,groups_to_remove=groupsInt_to_remove,eras=eras)

        configs = [configPartial,config1,config2,configMain]
    else:
        configs = [configMain]


    # Producing the instances #
    instances = []
    categoriesToSubmit  = []
    for era in eras:
        for iconf,config in enumerate(configs):
            if args.interpolation is not None:
                if args.interpolation_era is not None and str(era) not in args.interpolation_era:
                    if iconf <= 2:
                        continue
                else:
                    if iconf > 2:
                        continue
            config_era = {}
            for itemName,itemCfg in config.items():
                if isinstance(itemCfg,dict): # Dict -> check if common or per era
                    keys = [str(key) for key in itemCfg.keys()]
                    if len(keys) == 0:
                        config_era[itemName] = dict()
                    elif len(set(keys).intersection(set(eras))) > 0: # keys are eras -> era specific  
                        config_era[itemName] = itemCfg[list(itemCfg)[keys.index(era)]]
                    else:
                        config_era[itemName] = itemCfg
                else:   # Common item for all eras
                    config_era[itemName] = itemCfg
            
            config_era['era'] = era

            # Splitting #
            global_histConverter = copy.deepcopy(config_era['histConverter']) # Run all at once
            categoriesToRun = [config_era['histConverter'].keys()]
                
            if args.split is not None:
                if len(args.split) == 0:  # run all sequentially
                    categoriesToRun = [[cat] for cat in categoriesToRun[0]]
                else: # Run only requested
                    if args.interpolation is None:
                        splitCats = args.split
                    else:
                        if iconf == 0 or iconf == 3:
                            splitCats = args.split
                        elif iconf == 1:
                            splitCats = [configInt['matchingHist'][cat][configInt['param1']] for cat in args.split]
                        elif iconf == 2:
                            splitCats = [configInt['matchingHist'][cat][configInt['param2']] for cat in args.split]
                        else:
                            raise ValueError
                    if not all([cat in config_era['histConverter'].keys() for cat in splitCats]):
                        error_message = 'Category(ies) requested in split not found : '
                        error_message += ','.join([cat for cat in splitCats if cat not in config_era['histConverter'].keys()])
                        error_message += '\nAvailable categories :\n' + '\n'.join([f'... {key}' for key in config_era['histConverter'].keys()])
                        raise RuntimeError(error_message)
                    categoriesToRun = [splitCats]

            if iconf == 0:
                categoriesToSubmit.extend([cat for cat in categoriesToRun if cat not in categoriesToSubmit])

            # Instantiate #
            if args.custom is not None:
                custom_args = args.custom
            else:
                custom_args = None
            for icat,categories in enumerate(categoriesToRun):
                config_era['histConverter'] = {cat:global_histConverter[cat] for cat in categories}
                ilog = icat + iconf * len(categories) 
                instance = Datacard(configPath      = os.path.abspath(args.yaml),
                                    worker          = args.worker,
                                    pseudodata      = args.pseudodata,
                                    logName         = f'log_{ilog}.log',
                                    custom_args     = custom_args,
                                    **config_era)
                instances.append(instance)

    # Run #
    def run_instance(instance,methods):
        for method in methods:
            assert hasattr(instance,method)
            getattr(instance,method)()
        return instance 

    def run_interpolation(instance1,instance2,instanceInt,configMain,configInt):
        contentInt = InterpolateContent(content1        = instance1.content,
                                        content2        = instance2.content,
                                        era             = instance1.era,
                                        param1          = configInt['param1'],
                                        param2          = configInt['param2'],
                                        paramInt        = configInt['paramInt'],
                                        matchingHist    = configInt['matchingHist'],
                                        matchingGroup   = configInt['matchingGroup'])

        contentMain = instanceInt.content
        for cat in contentInt.keys():
            if cat not in contentMain.keys():
                contentMain[cat] = {}
            for group, histCfg in contentInt[cat].items():
                contentMain[cat][group] = histCfg
                    
        instanceInt.rebin       = configMain['rebin'] if 'rebin' in configMain.keys() else None
        instanceInt.groups      = configMain['groups']
        instanceInt.initialize()
        instanceInt.content     = contentMain
        if instanceInt.rebin is not None:
            instanceInt.applyRebinning()
        instanceInt.saveDatacard()

        return instanceInt

    def run_all(instances):
        if not args.worker:
            if os.path.exists(plotIt_path):
                shutil.rmtree(plotIt_path)
            os.makedirs(plotIt_path)
        if not args.plotIt and not args.yields:
            make_prod = True
            make_plots = True
            make_yields = True
        else:
            make_prod = False
            make_plots = args.plotIt
            make_yields = args.yields
        if args.worker:
            make_plots = False
            make_yields = False
        # Need to be done prior to avoid thread concurrences
        plotIt_configs = collections.defaultdict(list)
        # Serial processing #
        if args.jobs is None:
            for instance in instances:
                if make_prod:
                    instance.run_production()
        # Parallel processing #
        else:
            if args.jobs == -1 or args.jobs > len(instances):
                args.jobs = len(instances)
            methods = ['run_production']
            if make_prod:
                with mp.Pool(processes=args.jobs) as pool:
                    instances = pool.starmap(run_instance,[(instance,methods) for instance in instances])
        # Plots and yields #
        for instance in instances:
            if make_plots:
                plotIt_config = instance.prepare_plotIt()
                if plotIt_config is not None:
                    plotIt_configs[instance.era].append(plotIt_config)
            if make_yields:
                instance.saveYieldFromDatacard()

        # Interpolation #
        if args.interpolation is not None:
            # Split by era #
            instancesPerEra = {}
            instanceMatches = []
            for instance in instances:
                if instance.era not in instancesPerEra.keys():
                    instancesPerEra[instance.era] = [instance]
                else:
                    instancesPerEra[instance.era].append(instance)
            # Interpolate based on era #
            for era,instancesEra in instancesPerEra.items():
                if args.interpolation_era is not None and str(era) not in args.interpolation_era:
                    continue
                assert len(instancesEra) % 3 == 0
                N = len(instancesEra)
                instancesInt = instancesEra[:N//3]
                instances1 = instancesEra[N//3:2*N//3]
                instances2 = instancesEra[2*N//3:] # by construction
                # Build tuples of instances with histograms to be used for interpolation #
                paramInt = configInt['paramInt']
                param1 = configInt['param1']
                param2 = configInt['param2']
                for instanceInt in instancesInt:
                    if instanceInt.era != era:
                        continue
                    keysInt = list(instanceInt.histConverter.keys())
                    # Find equivalent keys in instances1 & instances2
                    keys1 = []
                    keys2 = []
                    for keyInt in configInt['matchingHist'].keys():
                        key1 = configInt['matchingHist'][keyInt][param1]
                        key2 = configInt['matchingHist'][keyInt][param2]
                        if keyInt in keysInt:
                            keys1.append(key1)
                            keys2.append(key2)
                    # Build the triplets #
                    found_match = False
                    for instance1 in instances1:
                        if instance1.era == era and set(instance1.histConverter.keys()) == set(keys1):
                            for instance2 in instances2:
                                if instance2.era == era and set(instance2.histConverter.keys()) == set(keys2):
                                    instanceMatches.append((instance1,instance2,instanceInt))
                                    found_match = True
                                    break
                            if found_match:
                                break
                    if not found_match:
                        raise RuntimeError(f'Could not find match to instance Int with hist conv keys in era {era} : '+','.join(keys1)+' (config 1) and '+','.join(keys2)+' (config 2)')

            # Serial processing #
            if args.jobs is None:
                for instance1,instance2,instanceInt in instanceMatches:
                    if make_prod:
                        instanceInt = run_interpolation(instance1,instance2,instanceInt,configMain,configInt)
            # Parallel processing #
            else:
                if args.jobs == -1 or args.jobs > len(instancesInt):
                    args.jobs = len(instancesInt)
                if make_prod:
                    with mp.Pool(processes=args.jobs) as pool:
                        instancesInt = pool.starmap(run_interpolation,[(*instanceMatch,configMain,configInt) for instanceMatch in instanceMatches])

            # plotit and yields #
            for instance in instancesInt:
                instance.plotIt = None
                instanceInt.groups.update(configInt['plotIt'])
                if make_plots:
                    plotIt_config = instance.prepare_plotIt()
                    if plotIt_config is not None:
                        plotIt_configs[instanceInt.era].append(plotIt_config)
                if make_yields:
                    instanceInt.saveYieldFromDatacard()

        if len(plotIt_configs) > 0:
            # Merge and run plotit (single thread anyway) #
            for era in plotIt_configs.keys():
                plotIt_config = Datacard.merge_plotIt(plotIt_configs[era])
                instances[0].run_plotIt(plotIt_config,era)

    # instance for combine #
    combine_instance = Datacard(configPath      = os.path.abspath(args.yaml),
                                worker          = args.worker,
                                pseudodata      = args.pseudodata,
                                logName         = f'log_{icat}.log',
                                custom_args     = custom_args,
                                **configMain) 
    combine_instance.initialize()

    ### SUBMIT ###
    if args.submit is not None:
        argsToSubmit = []
        argsToDelete = ['combine','submit']
        missingCats = combine_instance.check_datacards()
        missingIdx = []
        if len(missingCats) == 0:
            logging.info('All datacards have been produced, will not resubmit')
        else:
            for icat,cats in enumerate(categoriesToSubmit):
                argEntry = {k:v for k,v in args.__dict__.items() if k not in argsToDelete}
                # Add categories on which to run #
                argEntry['split'] = cats
                argEntry['worker'] = ''
                if args.split is None:
                    argEntry['era'] = eras
                    argsToSubmit.append(argEntry)
                    missingIdx.append(str(icat+1))
                else:
                    for iera,era in enumerate(eras):
                        argsToSubmit.append({**argEntry,'era':era})
                        # Check if category is done #
                        if era in missingCats.keys() and len(set(missingCats[era]).intersection(set(cats))) > 0:
                            missingIdx.append(str(icat*len(eras)+iera+1))
            # Get slurm parameters #
            paramsToSubmit = {}
            if args.jobs is not None:
                paramsToSubmit['cpus-per-task'] = args.jobs
            for submitArg in args.submit:
                if not '=' in submitArg:
                    logging.warning(f'Argument of `--submit` {submitArg} does not contain `=`  sign, will ignore')
                    continue
                paramsToSubmit[submitArg.split('=')[0]] = submitArg.split('=')[1]
            # Write sbatch script #
            slurmScript = os.path.join(outputDir,'slurmSubmission.sh')
            new_script = False
            if not os.path.exists(slurmScript):
                slurmScript = Datacard.writeSbatchScript(outputDir,params=paramsToSubmit,args=argsToSubmit,output_in_main=True)
                new_script = True

            #Submit #
            if not args.debug:
                if len(missingIdx) == 0:
                    logging.info('All datacard files are present, will not resubmit')
                else:
                    if len(missingIdx) < len(argsToSubmit):
                        logging.info('Some array ids did not succeed, resubmitting them : '+','.join(missingIdx))
                        slurmCmd = f'sbatch --array={",".join(missingIdx)} '
                    else:
                        slurmCmd = f'sbatch '
                    if not new_script:
                        for key,val in paramsToSubmit.items():
                            slurmCmd += f"--{key}={val} "
                    slurmCmd += slurmScript
                    logging.info(f'Sbatch command : {slurmCmd}')
                    logging.debug(f'Submitting {slurmScript}')
                    rc,output = Datacard.run_command(slurmCmd,return_output=True,shell=True)
                    for line in output:
                        logging.info(line.strip())

        ### PLOTIT ###
        if len(missingCats) == 0 and (args.plotIt or args.yields):
            run_all(instances)

    ### COMBINE ###
    if args.combine is not None:
        # Find missing jobs and if none missing, start combine #
        if args.submit is not None and len(missingIdx) > 0:
            logging.info('In submit mode and not all jobs have succeeded, will stop here')
        else:
            combine_args = {}
            for arg in args.combine_args:
                if '=' in arg:
                    combine_args[arg.split('=')[0]] = arg.split('=')[1]
                else:
                    logging.warning(f'`{arg}` will be ignored because no `=`')
            while True:
                missingCats = combine_instance.check_datacards()
                if len(missingCats) == 0:
                    combine_instance.run_combine(args.combine,combine_args,debug=args.debug) 
                    break
                else:
                    logging.info('Missing following categories in txt files :')
                    for era,cats in missingCats.items():
                        add_cats = []
                        for cat in cats:
                            logging.info(f'... {cat} [{era}]')
                            if args.interpolation is not None: # add the other two configs cats
                                add_cats.extend(configInt['matchingHist'][cat].values())
                        missingCats[era].extend(add_cats)
                        
                    logging.info('Some categories are missing, will produce them')
                    # Find associated instances #
                    remainingInstances = []
                    for instance in instances:
                        if str(instance.era) in missingCats.keys() and len(set(missingCats[str(instance.era)]).intersection(set(instance.histConverter.keys()))):
                            remainingInstances.append(instance)
                                # if any category in the instance is in the missing ones, reproduce
                    run_all(remainingInstances)
    ### RUN ###
    else:
        if args.submit is None:
            run_all(instances)

