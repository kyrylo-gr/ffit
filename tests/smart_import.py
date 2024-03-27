print("Kernel is started")

import os, sys

def add_parent_dir_to_path():
    SCRIPT_DIR = os.path.join(os.path.dirname( __file__ ), os.pardir)
    sys.path.append(os.path.abspath(SCRIPT_DIR))
    
try:
    import scifit
except ModuleNotFoundError:
    add_parent_dir_to_path()

import scifit