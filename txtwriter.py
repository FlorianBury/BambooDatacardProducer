import os
import sys
import logging
from datetime import datetime

from collections.abc import MutableMapping

from IPython import embed

sep = os.linesep
dashes = 80 * "-"

threshold = 0.05 # to remove some processes systematics

class Process():
    def __init__(self,name,rate,index):
        self.name = name
        self.rate = rate
        self.index = index

        self.systematics = {}

    def addSystematic(self,systName,systType,systVal):
        self.systematics[systName] = {'type':systType,'val':systVal}

    def __repr__(self):
        string = f"Process : {self.name}{sep}\tRate  : {self.rate}{sep}\tIndex : {self.index}{sep}\tSystematics :{sep}"
        for name,syst in self.systematics.items():
            systType = syst['type']
            systVal  = syst['val']
            string += f'\t\t- {name}{sep}\t\t\tType  : {systType}{sep}\t\t\tValue : {systVal}'
        return string

    def __eq__(self,other):
        raise NotImplemented

    def __lt__(self,other):
        return self.index < other.index
        
    def __gt__(self,other):
        return self.index > other.index


class Processes(MutableMapping):
    def __init__(self,binName):
        self.store = dict()
        self.binName = binName

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self,key,val):
        self.store[key] = val

    def addProcess(self, name, **kwargs):
        self.__setitem__(name,Process(name,**kwargs))

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        for key,val in sorted(self.store.items(), key = lambda x : x[1]):
            yield key

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        string = ""
        for key in self.__iter__():
            string += self.__getitem__(key).__repr__() + sep
        return string 

    def getIndices(self):
        return [process.index for process in self.store.values()]
        
    def addSystematic(self,name,**kwargs):
        self[name].addSystematic(**kwargs)
        
    @property
    def spaces(self):
        return max(*[len(key) for key in self.__iter__()],
                   *[len(systName) for systName in self.getAllSystNamesAndTypes().keys()],
                   len(self.binName),
                   len('observation')) + 5 
                   
    def getAllSystNamesAndTypes(self):
        systNames = {}
        for process in self.values():
            for systName in process.systematics.keys():
                if systName not in systNames.keys():
                    systNames[systName] = process.systematics[systName]['type']
        return systNames

    def applyThreshold(self):
        for processName,process in self.items():
            if process.rate < threshold and process.index > 0: 
                logging.debug(f"Bin {self.binName} - Process {processName} : yield {process.rate:0.5f} < threshold ({threshold})")
                del self[processName]

    def getSystValuesPerProcess(self,systName):
        systs = []
        for process in self.values():
            if systName in process.systematics.keys():
                systs.append(process.systematics[systName]['val'])
            else:
                systs.append('-')
        return systs


class Writer:
    def __init__(self,binNames):
        if isinstance(binNames,list) or isinstance(binNames,tuple):
            self.binNames = binNames
        else:
            self.binNames = [binNames]
        self.processes = {binName:Processes(binName) for binName in self.binNames}
        self.observations = {binName:-1 for binName in self.binNames}
        self.systNames = {}
        self.footer = []

    def build(self):
        for processes in self.processes.values():
            processes.applyThreshold()
            for systName,systType in processes.getAllSystNamesAndTypes().items():
                if systName not in self.systNames.keys():
                    self.systNames[systName] = systType

    def dump(self, txt, shapePaths):
        self.build()

        if not isinstance(shapePaths,list) and not isinstance(shapePaths,tuple):
            shapePaths = [shapePaths]

        spaces = max([processes.spaces for processes in self.processes.values()])
        nprocesses = len(set([processName for processes in self.processes.values() for processName in processes.keys()]))

        card = open(txt, "w")
        # Header #
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        card.write(f"### Date : {date}{sep}")
        card.write(f"imax {len(self.processes)} number of bins{sep}")
        card.write(f"jmax {nprocesses-1} number of processes minus 1{sep}")
        card.write(f"kmax {len(self.systNames)} number of nuisance parameters{sep}")
        card.write(f"{dashes}{sep}")

        # Shapes #
        for binName,shapes in zip(self.binNames,shapePaths):
            card.write(f"shapes * {binName} {shapes} $PROCESS $PROCESS__$SYSTEMATIC{sep}")
        card.write(f"{dashes}{sep}")

        # Bins #
        card.write("bin".ljust(spaces)+"".join([binName.ljust(spaces) for binName in self.binNames])+sep)
        card.write("observation".ljust(spaces)+"".join([str(self.observations[binName]).ljust(spaces) for binName in self.binNames])+sep)
        card.write(f"{dashes}{sep}")

        # Processes #
        card.write("bin".ljust(spaces*2) \
            + "".join([binName.ljust(spaces)  for binName,processes in self.processes.items() for _ in processes.values()]) \
            + sep)
        card.write("process".ljust(spaces*2) \
            + "".join([process.name.ljust(spaces) for processes in self.processes.values() for process in processes.values()]) \
            + sep)
        card.write("process".ljust(spaces*2) \
            + "".join([str(process.index).ljust(spaces) for processes in self.processes.values() for process in processes.values()]) \
            + sep)
        card.write("rate".ljust(spaces*2) \
            + "".join(["{:0.10f}".format(process.rate).ljust(spaces) for processes in self.processes.values() for process in processes.values()]) \
            + sep)
        card.write(f"{dashes}{sep}")

        # Systematics #
        for systName,systType in sorted(self.systNames.items(), key = lambda x : x[0]):
            card.write(systName.ljust(spaces) \
                     + systType.ljust(spaces) \
                     + "".join([str(val).ljust(spaces) for processes in self.processes.values() for val in processes.getSystValuesPerProcess(systName)]) \
                     + sep)
            

        # Footer #
        for line in self.footer:
            card.write(f"{line}{sep}")

        # AutoStats #
        for binName in self.binNames:
            card.write(f"{binName}".ljust(spaces)+"autoMCStats 10 0 1"+sep)


        card.close()

    def addProcess(self,binName,processName,rate,processType):
        """
            processName [str] : name of the process
            rate [float] : rate of the process
            processType[str] : signal | mc (background)| data
        """
        if processType not in ['signal','mc','data']:
            raise RuntimeError(f"Process {processName} has unknow type {processType}")

        if processType == 'data':
            self.observations[binName] = rate
            return

        indices = self.processes[binName].getIndices()
        if len(indices) == 0:
            if processType == 'signal':
                index = 0
            else:
                index = 1
        else:
            amin = min(self.processes[binName].getIndices())
            amax = max(self.processes[binName].getIndices())
            if processType == 'signal':
                index = min(indices)-1
            else:
                index = max(indices)+1
            
        self.processes[binName].addProcess(name=processName,rate=rate,index=index)

    def _addSystematic(self,binName,processName,**kwargs):
        if processName in self.processes[binName].keys():
            self.processes[binName].addSystematic(processName,**kwargs)
        else:
            logging.debug(f'Process {processName} not found for systematic {kwargs["systName"]}')

    def addShapeSystematic(self,binName,processName,shapeName):
        self._addSystematic(binName,processName,systName=shapeName,systType='shape',systVal='1')

    def addLnNSystematic(self,binName,processName,systName,value):
        if isinstance(value,list) or isinstance(value,tuple):
            assert len(value) == 2
            var = '{}/{}'.format(*value)
        elif isinstance(value,float):
            var = str(value)
        else:
            raise RuntimeError("lnN systematic value not understood")
        self._addSystematic(binName,processName,systName=systName, systType='lnN',systVal=var)

    def addFooter(self,line):
        self.footer.append(line) 

        

if __name__ == '__main__':
    writer = Writer('HH')
    writer.addProcess("TT",254.1,"mc")
    writer.addProcess("DY",354.1,"mc")
    writer.addProcess("data_obs",25.1,"data")
    writer.addProcess("ST",0.01,"mc")
    writer.addProcess("HH1",0.1,"signal")
    writer.addProcess("HH2",0.1,"signal")
    
    writer.addShapeSystematic("TT",shapeName='pt_reweight')
    writer.addShapeSystematic("ST",shapeName='testLowStat')
    writer.addLnNSystematic("DY",systName='nonclosure',value=0.15)
    
    print (writer.processes)
    
    writer.dump()
        
