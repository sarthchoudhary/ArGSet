# counting all events
# import sys
import numpy as np
from pyreco.manager.manager import Manager
from tqdm import tqdm

event_cntr = 0

filename = '/work/sarthak/argset/data/2024_Mar_27/midas/run00061.mid.lz4'
outfile = 'pyR00061' #100223 events
confile = 'argset.ini'
# tmin,tmax=0,1750

cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
m = Manager( midas=True, cmdline_args=cmdline_args)
# m = Manager( midas=True) #, cmdline_args=cmdline_args)
baseline_samples = m.config('reco', 'bl_to', 'int')

for nev, event in enumerate(m.midas): # what gets the next event?
    event_cntr += 1

print(event_cntr)