import os
import sys
import json

from dhi.datacard_tools import get_workspace_parameters

assert len(sys.argv) == 3

path_workspace = sys.argv[1]
path_json = sys.argv[2]

if not os.path.exists(path_workspace):
    raise RuntimeError('File %s does not exist'%path_workspace)

params = get_workspace_parameters(path_workspace)

with open(path_json,'w') as handle:
    content = json.dump(params,handle,indent=4)

