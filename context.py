import ROOT

class TFileOpen:
    def __init__(self, file_name, mode='r'):
        if mode == 'r':
            self.file_obj = ROOT.TFile(file_name,'READ')
        elif mode == 'w':
            self.file_obj = ROOT.TFile(file_name,'RECREATE')
        elif mode == 'u':
            self.file_obj = ROOT.TFile(file_name,'UPDATE')
        else:
            raise ValueError(f'Unknown mode `{mode}`')
    def __enter__(self):
        return self.file_obj
    def __exit__(self, exception_type, exception_value, traceback):
        self.file_obj.Close()

