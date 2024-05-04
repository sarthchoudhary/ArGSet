import numpy as np
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pyreco.manager.manager import Manager
from pyreco.reco.filtering import WFFilter
from scipy.signal import find_peaks
import pandas as pd
from os import path
from tqdm import tqdm

def find_clean_wfs( pyreco_manager, \
                     clean_target:int, n_channel:int = 2) -> pd.DataFrame:
    '''
    Steps:
    - read a midas file 
    - search clean events 
    - tmp: save clean wfs as ndarray/pickle 
    - return a dataframe: {data filename, wf index, fit parameter values} 
   '''
    progress_bar = tqdm(total = clean_target, colour='blue')
    mfilter = WFFilter(pyreco_manager.config)
    clean_cntr = 0
    event_index = 0 
    clean_catalogue = pd.DataFrame(columns= [
                    # 'filename',
                    'event_index',
                    # 'fit_param',
                    'wf'
    ]
    )
    ## skip first two events in the file
    # nev_max = pyreco_manager.config('base', 'nevents', 'int')
    # nev_max = 2
    for nev, event in enumerate(pyreco_manager.midas):
        if nev < 2: continue
        if nev > 1: break
    # skip block ends

    while clean_cntr < clean_target: #TODO change to 100k
        og_wf = read_wfs(pyreco_manager)
        flt = np.reshape(mfilter.numba_fast_filter(og_wf), newshape=og_wf.shape)
        mas = pyreco_manager.algos.running_mean(flt, gate=60)
        flt_proc = np.copy(flt[n_channel])
        flt_proc = flt[n_channel] - mas[n_channel]
        flt_proc = np.where(flt_proc>0,flt_proc, 0)
        rms = pyreco_manager.algos.get_rms(flt_proc)
        flt_above_3rms = np.where(flt_proc > 3*rms, flt_proc, 0)        
        if len(find_peaks(flt_above_3rms)[0]) == 1:
            clean_cntr += 1
            clean_catalogue = clean_catalogue._append(
                {
                    # 'filename':,
                    'event_index': event_index,
                    # 'fit_param':,
                    'wf': og_wf
                }, ignore_index=True
            )
            # print(f"Clean found {clean_cntr}.") # diag
 
        progress_bar.update(0.05) #TODO fix progress bar
        event_index += 1
        # if event_index%100 == 0: # diag
        #     print(f'proceed to the Next event. {event_index}') # diag
    progress_bar.close()
    return clean_catalogue

def read_wfs(pyreco_manager:Manager) -> np.ndarray:
    '''
    Steps:
    - reads the next midas event
    - substract baseline value
    - return wfs as ndarray for single event 
    '''
    #TODO failsafe against empty events
    nev_lim = 0
    for nev, event in enumerate(pyreco_manager.midas):
        if nev > nev_lim: # diag
        # if nev > 0:
            break
        if event is None: # does this even do anything?
            nev_lim += 1 # diag
            continue
        if event.nchannels == 0:
            nev_lim += 1 # diag
            continue        
        wfs = event.adc_data
        for i,wf in enumerate(event.adc_data):
            wfs[i] = wf-event.adc_baseline[i]
    return wfs

def main(n_channel:int, save_file:bool, \
        clean_target:int=10, plots_target:int=10) ->None:
    '''
    Steps:
    - run read_wfs
    - run find_clean_wfs
    - get clean_catalogue as a DataFrame
    - plot waveforms and save to pdf
    - save DataFrame as a pickle file 
    '''
    filename = '/work/sarthak/ArgSet/2024_Mar_27/midas/run00061.mid.lz4'
    outfile = 'temp_folder/temp_pyR00061'
    confile = 'argset.ini'

    cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
    pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)

    clean_catalogue_df = find_clean_wfs(pyreco_manager, clean_target, n_channel)

    if save_file:
        output_path = path.join("temp_folder", "argset_wfs_catalogue.pkl")
        clean_catalogue_df.to_pickle(output_path)
        for i in range(plots_target): # temp diag
            wf = clean_catalogue_df.iloc[i]['wf'] # channel specific
            plt.figure(i, figsize=(4,3))
            plt.title('Test waveform plot')
            plt.plot(wf[n_channel], label=f'Channel {n_channel}')
            plt.legend()
            plt.savefig(f'temp_folder/midas_wf_{i}.pdf')

if __name__ == "__main__":
    main(2, True, clean_target=100)