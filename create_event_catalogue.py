import sys
sys.settrace
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
from pyreco.manager.manager import Manager
from time import perf_counter
import h5py
from os import path


def create_event_catalogue(dir_name:str, filename:str, output_dir:str) -> None:

    file_path = path.join(dir_name, filename)
    file_basename = filename.split(sep='.')[0]
    outfile = 'temPyR00061'
    confile = 'argset.ini'
    cmdline_args = f'--config {confile} -o {outfile} -i {file_path}'
    pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)

    pbar = tqdm(total = 110002, colour='blue')

    event_counter_ls = []
    wf_ls = []
    # wf_stack_ch0 = []
    # wf_stack_ch1 = []
    # wf_stack_ch2 = []

    file_format = 'pkl' #'npz' # 'h5' #TODO dynmic

    start_count = perf_counter()
    for event_index, event in enumerate(pyreco_manager.midas):
        if event.event_counter > 0:
            # event_index_ls.append(event_index)
            event_counter_ls.append(event.event_counter)
            wf_ls.append(event.adc_data - np.vstack(event.adc_baseline))
            # event_ch0, event_ch1, event_ch2 = event.adc_data - np.vstack(event.adc_baseline)
            # wf_stack_ch0.append(event_ch0)
            # wf_stack_ch1.append(event_ch1)
            # wf_stack_ch2.append(event_ch2)

        # if event_index > 10: # diag
        #     break
        pbar.update(1)
    pbar.close()

    if file_format == 'pkl':
        event_dict = {
            # 'event_index': event_index_ls,
            'event_counter': event_counter_ls,
            'wf': wf_ls,
            # 'wf_ch0': wf_stack_ch0,
            # 'wf_ch1': wf_stack_ch1,
            # 'wf_ch2': wf_stack_ch2,
        }
        event_catalogue = pd.DataFrame.from_dict(event_dict)

        print(colored(f'preparing pickle...'))
        output_filename = f"event_catalogue_{file_basename}.pkl"
        output_path = path.join(output_dir, output_filename)
        # event_catalogue.to_pickle(f"data/event_catalogue_{file_basename}.pkl") #TODO: dynamic
        event_catalogue.to_pickle(output_path)
    print(f"pickle written to {output_path}")
    print('time taken:', perf_counter() - start_count)

    start_count = perf_counter()
    if file_format == 'h5':
        print(f'creating h5 container')
        with h5py.File('data/event_catalogue_{file_basename}.h5', 'w') as event_catalogue: #TODO: dynamic 
            # event_catalogue.create_dataset('wf', data = wf_ls, compression="gzip")
            event_catalogue.create_dataset('event_counter', data = event_counter_ls, compression="gzip")
            event_catalogue.create_dataset('wf_ch0', data = wf_stack_ch0, compression="gzip")
            event_catalogue.create_dataset('wf_ch1', data = wf_stack_ch1, compression="gzip")
            event_catalogue.create_dataset('wf_ch2', data = wf_stack_ch2, compression="gzip")
    print('time taken:', perf_counter() - start_count)

def main() -> None:
    dir_name = '/work/sarthak/ArgSet/2024_Mar_27/midas' #TODO dynmic
    # filename = 'run00061.mid.lz4'
    # filename = 'run00052.mid.lz4'
    # create_event_catalogue(dir_name, filename, output_dir)
    output_dir = '/home/sarthak/my_projects/argset/data'
    files_ls = ['run00052.mid.lz4', 'run00053.mid.lz4', 'run00054.mid.lz4', \
        'run00061.mid.lz4', 'run00062.mid.lz4', 'run00063.mid.lz4']
    for filename in files_ls:
        create_event_catalogue(dir_name, filename, output_dir)

if __name__ == "__main__":
    main()