import io
import os
import yaml 
import logging

from IPython import embed

class YMLIncludeLoader(yaml.SafeLoader): 
    """Custom yaml loading to support including config files. Use `!include (file)` to insert content of `file` at that position."""
    
    def __init__(self, stream):
        """
            stream can be : 
            - IO object -> passed as in to the Pyyaml logic
            - dict : hack to pass a string containing the yaml content 
                -> 'filename' : filename including ist relative path
                -> 'formatting' : how to format the content
        """
        self._root = None
        self._formatting = None
        if isinstance(stream,io.TextIOWrapper):
            self._root = os.path.dirname(stream.name)
            super(YMLIncludeLoader, self).__init__(stream)
        elif isinstance(stream,dict):
            self._root = os.path.dirname(stream['filename'])
            with open(stream['filename'],'r') as f:
                content = ''.join([line for line in f])
            if 'formatting' in stream.keys() and isinstance(stream['formatting'],dict):
                self._formatting = stream['formatting']
                for key,val in self._formatting.items():
                    if f'{{{key}}}' not in content:
                        logging.debug(f"Custom format {{{key}}} not present in {stream['filename']}, will have zero effect")
                    else:
                        content = content.replace(f'{{{key}}}',val)
            super(YMLIncludeLoader, self).__init__(content)
        else:
            raise RuntimeError(f'Format {type(stream)} not implemented')
            

    def include(self, node):
        if self._root is None:
            raise RuntimeError('Directory not set')
        filename = os.path.join(self._root, self.construct_scalar(node))
        if not os.path.isfile(filename):
            raise RuntimeError(f'{filename} is not a valid file')
        if self._formatting is None:
            with open(filename, 'r') as f:
                try:
                    return yaml.load(f, YMLIncludeLoader)
                except yaml.parser.ParserError as err: 
                    logging.error(f'Parser error when loading file {filename}, see log below')
                    raise err
        else:
            try:
                return yaml.load({'filename':filename,'formatting':self._formatting}, 
                                 Loader=YMLIncludeLoader) 
            except yaml.parser.ParserError as err:
                logging.error(f'Parser error when loading file {filename}, see log below')
                raise err

YMLIncludeLoader.add_constructor('!include', YMLIncludeLoader.include)


