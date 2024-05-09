import sys
sys.settrace
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from pyreco.manager.manager import Manager



filename = '/work/sarthak/ArgSet/2024_Mar_27/midas/run00061.mid.lz4'
outfile = 'temPyR00061'
confile = 'argset.ini'
cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)

pbar = tqdm(total = 100002, colour='blue')

event_index_ls = []
event_counter_ls = []
wf_ls = []

for event_index, event in enumerate(pyreco_manager.midas):
    if event.event_counter > 0:
        event_index_ls.append(event_index)
        event_counter_ls.append(event.event_counter)
        wf_ls.append(event.adc_data - np.vstack(event.adc_baseline))

    pbar.update(1)
    # if event_index > 3550:
    #     break
pbar.close()

event_dict = {
    'event_index': event_index_ls,
    'event_counter': event_counter_ls,
    'wf': wf_ls,
}
event_catalogue = pd.DataFrame.from_dict(event_dict)

print(colored(f'Making pickle.'))
event_catalogue.to_pickle('temp_folder/event_catalogue_run0061.pkl')