import os
import sys
import json
from copy import deepcopy
import importlib

from dhi.util import import_ROOT

ROOT = import_ROOT() 

def get_func(func_id):
    """ Obtain function by name """
    if "." not in func_id:
        raise ValueError("invalid func_id format: {}".format(func_id))
    module_id, name = func_id.rsplit(".", 1)

    try:                                                                                         
        mod = importlib.import_module(module_id)
    except ImportError as e:
        raise ImportError(
            "cannot import plot function {} from module {}: {}".format(name, module_id, e)
        )

    func = getattr(mod, name, None)
    if func is None:
        raise Exception("module {} does not contain plot function {}".format(module_id, name))

    return func
 
def byteify(input):
    """ Transform any unicode item in a standard object into strings """
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


assert len(sys.argv) == 3 or len(sys.argv) == 4

func_id = sys.argv[1]
path_json = sys.argv[2]

func = get_func(func_id)

if not os.path.exists(path_json):
    raise RuntimeError('File %s does not exist'%path_json)

with open(path_json,'r') as handle:
    content = byteify(json.load(handle))

if len(sys.argv) == 4:
    path_getter = sys.argv[3]
    root_files = [] # Needed to avoid garbage collection on root attribyes attached to a file
    if not os.path.exists(path_getter):
        raise RuntimeError('File %s does not exist'%path_getter)
    with open(path_getter,'r') as handle:
        getter = byteify(json.load(handle))
    for key,config in getter.items():
        if key not in content.keys():
            raise RuntimeError('Getter key %s not in content keys'%key)
        if config['type'] == 'ROOT':
            F = ROOT.TFile(content[key])
            content[key] = F.Get(str(config['name']))
            root_files.append(F)
        else:
            raise RuntimeError('Getter key %s type %s not understood'%(key,config['type']))

func(**content)

