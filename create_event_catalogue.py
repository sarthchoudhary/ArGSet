import sys
sys.settrace
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from pyreco.manager.manager import Manager
from time import perf_counter
import h5py

filename = '/work/sarthak/ArgSet/2024_Mar_27/midas/run00061.mid.lz4' #TODO dynmic

outfile = 'temPyR00061'
confile = 'argset.ini'
cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)

pbar = tqdm(total = 110002, colour='blue')

event_counter_ls = []
# wf_ls = []
wf_stack_ch0 = []
wf_stack_ch1 = []
wf_stack_ch2 = []

file_format = 'h5' #'pkl' #'npz' # 'h5' #TODO dynmic

start_count = perf_counter()
for event_index, event in enumerate(pyreco_manager.midas):
    if event.event_counter > 0:
        # event_index_ls.append(event_index)
        event_counter_ls.append(event.event_counter)
        # wf_ls.append(event.adc_data - np.vstack(event.adc_baseline))
        event_ch0, event_ch1, event_ch2 = event.adc_data - np.vstack(event.adc_baseline)
        wf_stack_ch0.append(event_ch0)
        wf_stack_ch1.append(event_ch1)
        wf_stack_ch2.append(event_ch2)

    # if event_index > 10:
    #     break
    pbar.update(1)
pbar.close()

if file_format == 'pkl':
    event_dict = {
        # 'event_index': event_index_ls,
        'event_counter': event_counter_ls,
        # 'wf': wf_ls,
        'wf_ch0': wf_stack_ch0,
        'wf_ch1': wf_stack_ch1,
        'wf_ch2': wf_stack_ch2,
    }
    event_catalogue = pd.DataFrame.from_dict(event_dict)

    print(colored(f'preparing pickle.'))
    event_catalogue.to_pickle('data/event_catalogue_run0061.pkl') #TODO: dynamic

print('time taken:', perf_counter() - start_count)

start_count = perf_counter()
if file_format == 'h5':
    print(f'creating h5 container')
    with h5py.File('data/event_catalogue_run0061.h5', 'w') as event_catalogue: #TODO: dynamic 
        # event_catalogue.create_dataset('wf', data = wf_ls, compression="gzip")
        event_catalogue.create_dataset('event_counter', data = event_counter_ls, compression="gzip")
        event_catalogue.create_dataset('wf_ch0', data = wf_stack_ch0, compression="gzip")
        event_catalogue.create_dataset('wf_ch1', data = wf_stack_ch1, compression="gzip")
        event_catalogue.create_dataset('wf_ch2', data = wf_stack_ch2, compression="gzip")
print('time taken:', perf_counter() - start_count)