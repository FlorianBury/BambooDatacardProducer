# BambooDatacardProducer

The purpose of this set of scripts is to take the output from bamboo, produce datacards with several modifications and run combine on them.
Following operations are implemented
- Aggregation of several histograms
- Rebinning (several algorithms are implemented)
- Renaming of the histograms and systematics
- Corrections (eg non-closure)
- Production of text datacards
- Running combine : limits, goodness of fit, pulls & impacts, prefit & postfit plots

## produceDataCards.py ##

This script is the one that runs on a config (to provide all the configuration options) and can be run like this 
```
python produceDataCards.py --yaml config.yml [options]
```
An exampe of config can be found [here](config/datacard_example.yml)

The following options can be added to the command : 
- `-v`/`--verbose` : print more debugging information
- `--era` : to only run on a specific era that is in the `era` entry of the config 
   [Note : this only matters if some entries of the config are specified per era]
- `--pseudodata` : when using this flag the data in the `groups` entry of the config is ignored, and the sum of mc samples is used instead
- `--plotIt` : To only run the plotIt plotting part (the datacards must have been produced already)
- `--yields` : To only run the yield table extraction (the datacards must have been produced already)
- `--split` : Instead of running on all categories (per era) at once, will run once per category in serial mode. Useful when there are too many histograms and/or systematics and memory errors appear [plotIt will be run at the end and also split per category]
- `-j`/`--jobs` : Will run the code in parallel mode, with as many threads as given in the agrument. [Note : plotIt and yields will still be run sequentially at the end]
- `--submit` : Instead of running locally, this will create a sbatch script and submit the jobs to the cluster using the default parameters and overriding what is given in the argument (example : `--submit time=...  mem-per-cpu=... [...]`). If used with `--split` will submit one job per category, if not will send a single job that runs sequentially. When used with `-j` it will request a multithreaded node.
- `--debug` : will run without submitting the jobs, useful for debugging, useless if not used with `--submit`
- `--custom` : To apply some variable string to the config files (example : `--custom mass=500` if `{mass}` is somewhere in the config) [Note : see below for more explanation]
- `--combine` : List (separated by spaces) of combine modes defined in the `combineConfigs` entry of the config (see below for more explanation)
- `--worker` : Force jobs to be run locally (used for debugging and when the jobs run on the cluster)
- `--interpolation*` : arguments used when trying to use the interpolation method between paramaters [TODO : explain further]

### Few tricks for writing configs 

Configs can become quite long and repetitive when you try to do different actions or when you are doing parameter scans, etc.

A few tricks have been implemented so make your life easier !

#### Include sub-configs 
You can put an entire config entry (or subpart of it) in a separate file using `!include <name of sub-config>` (they have to be in the same directory though). During the yaml importation it will be filled by the sub-config content.

Can be extremely useful when some content has to be shared between multiple configs, to avoid copy pasting and keeping track of the differences.
Example : 
```yaml
groups:
  - !include <groups1>.yml
  - !include <groups2>.yml
```
Note : in this case the groups wil consist of a list of two dicts, but in the script it will make it a single dict.

#### Using variables in the config
When you try to do a parameter scan, you need one datacard per parameter, but you do not want to have a config per parameter because keeping track of changes is a nightmare...

To overcome that you can put a string representing this parameter in the config (say `{mass}`) and call it in the command this way `--custom mass=500`. It will effectively look at the yaml config, overwrite every `{mass}` string it sees in the content by `500` and then load it.

Note that this works recursively : you can use a variable parameter in a sub-config and it will work the same !

Example : 
```yaml
[...] 
outputDir = path_to_my_datacards_M_{mass}
[...]
groups:
  - !include backgrounds.yml
  - !include signal_{mass}.yml
```


### Combine 




