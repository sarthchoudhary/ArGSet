import numpy as np
import matplotlib.pyplot as plt
from pyreco.manager.manager import Manager
from pyreco.reco.filtering import WFFilter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
from os import path
from tqdm import tqdm
from termcolor import colored

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
    print(colored(f"Finding clean waveforms", 'green', attrs = ['blink', 'bold']) )
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
                    'wf': og_wf,
                    'peak_loc': np.ceil(np.mean(np.where(flt_above_3rms != 0))),
                    # prefers to take ceiling value over floor
                }, ignore_index=True
            )
            # print(f"Clean found {clean_cntr}.") # diag
 
        progress_bar.update(0.05) #TODO improve progress bar
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

def arma_template(t, t0, sigma, tau, scale, baseline, K) -> np.ndarray:
    ''' 
    ARMA template.
    t0 - offset
    K - scalar multiplication factor for matching data
    '''
    return baseline + K*((1-scale)/(sigma*np.sqrt(2*np.pi))*np.exp(-((t-t0)/(sigma))**2/2) + scale*np.heaviside(t-t0,0)/tau*np.exp(-(t-t0)/tau))

def transform_shift_wfs(og_wfs:np.ndarray) -> tuple:
    ''' Transforms array. There should be a vectorized method for this.''' #TODO: vectorize
    wfs = np.copy(og_wfs)
    # shift_values = {}
    for _c in range(wfs.shape[0]):
        a = np.min(og_wfs[_c])
        if a < 0:
            wfs[_c] = og_wfs[_c] + np.abs(a)
        #     shift_values[str(_c)] = a
        # else:
        #     shift_values[str(_c)] = 0
    # return (wfs, shift_values)
    return wfs

def red_chisq(f_obs: np.ndarray, f_exp: np.ndarray, fittedparameters: np.ndarray) -> float:
    ''' calculates reduced chisquare '''
    chisqr = np.sum((f_obs - f_exp)**2 / f_exp)
    ndf = f_obs.shape[0]
    return chisqr/(ndf -fittedparameters.shape[0])

def fit_template(clean_catalogue:pd.DataFrame, \
                n_channel:int = 2) -> pd.DataFrame:
    '''
    fits template function to waveform.
    - t0 is selected dynamically.
    - add reduced chisqr to clean_catalogue
    - multiple processing can be performed in parallel
    '''
    fit_begin = 0
    fit_param_df = pd.DataFrame(columns = ['fit_param_ch2']) #'fit_param_ch0', 'fit_param_ch1', 
    # for clean_index in tqdm(range(10)): # temp
    print(colored(f"Commence fitting", 'green', attrs = ['blink', 'bold']) )
    
    for clean_index in tqdm(range(clean_catalogue.shape[0])): #TODO we can do multiprocessing
        wf = clean_catalogue.iloc[clean_index]['wf'] #[n_channel]
        wf = transform_shift_wfs(wf)
        peak_loc = clean_catalogue.iloc[clean_index]['peak_loc']
        fit_end = wf[n_channel].shape[0]
        x_values = np.arange(0,fit_end)
        # mse_700_800 = np.std(wf[n_channel][700:800])/np.sqrt(wf[n_channel][700:800].shape[0]) # do this for each channel
        p0_input_ch2 = [peak_loc,   2.5,  80.0,   0.95, 0.0, 10000.0]
        fittedparameters_ch2, _pcov = curve_fit(arma_template, x_values[fit_begin:fit_end], \
                                    wf[n_channel][fit_begin:fit_end], p0 = p0_input_ch2, \
                                    # sigma=mse_700_800*np.ones(wfs[2].shape), \
                                    bounds = ([0, 0, 0, 0, -np.inf, 0], \
                                              [np.inf, 10, np.inf, 1, np.inf, np.inf])
                                    )
        # fit_param_df.iloc[clean_index]['fit_param_ch2'] = fittedparameters_ch2
        fit_param_df = fit_param_df._append({'fit_param_ch2': fittedparameters_ch2,
                                             'chisqr_ch2': red_chisq(wf[n_channel], arma_template(x_values, *fittedparameters_ch2), \
                                                                                   fittedparameters_ch2)
                                             }, ignore_index=True) #TODO: repeat all channels
        #TODO add chisqr values to fit_param_df
    clean_catalogue = clean_catalogue.join(fit_param_df)
    return clean_catalogue

def main(n_channel:int, save_plot:bool=False, \
        clean_target:int=10, plots_target:int=10) ->None:
    '''
    Steps:
    - call read_wfs
    - call find_clean_wfs
    - call fit_template
    - plots waveforms and saves to pdf
    - saves DataFrame as pickle file 
    - finally need another function for creating all histogram requested by Marcin 
    - make fit optional
    - turn plotting into a function as well
    '''
    filename = '/work/sarthak/ArgSet/2024_Mar_27/midas/run00061.mid.lz4' #TODO: dynamic path
    outfile = 'temp_folder/temp_pyR00061'
    confile = 'argset.ini'

    cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
    pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)

    clean_catalogue_df = find_clean_wfs(pyreco_manager, clean_target, n_channel)
    clean_catalogue_df = fit_template(clean_catalogue_df)

    output_path = path.join("temp_folder", "argset_wfs_catalogue.pkl")
    clean_catalogue_df.to_pickle(output_path)

    if save_plot:
        print(colored(f"Commence plotting", 'green', attrs = ['blink', 'bold']))
        x_values = np.arange(0, 1750)
        for i in tqdm(range(plots_target)):
            wf = clean_catalogue_df.iloc[i]['wf'] # channel specific
            plt.figure(i, figsize=(8,6))
            plt.title('fit vs data')
            plt.plot(wf[n_channel] + np.abs(np.min(wf[n_channel])), '.--', color='black', \
                     label=f'data Ch: {n_channel}')
            
            fittedparameters_ch2 = clean_catalogue_df.iloc[i]['fit_param_ch2']
            
            plt.plot(arma_template(x_values, *fittedparameters_ch2), \
        '-', color='red', label='template fit on Ch_2')
            plt.legend()
            plt.savefig(f'temp_folder/midas_wf_{i}.pdf')

if __name__ == "__main__":
    main(2, save_plot=True, clean_target=10, plots_target=100)