### on chuck do this:
### srun --mem=16G -A bejger-grp -p dgx --pty bash
import numpy as np
import matplotlib.pyplot as plt
from pyreco.manager.manager import Manager # TODO: Can WFfilter work w/o it
from pyreco.reco.filtering import WFFilter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
from os import path
from tqdm import trange
from termcolor import colored

def find_clean_wfs( pyreco_manager, catalogue_filename:str, \
                    n_channel:int = 2) -> pd.DataFrame:
    '''
    Uses ARMA filter to say whether a wf is clean or not!
    Steps:
    - read a midas file 
    - search clean events 
    - tmp: save clean wfs as ndarray/pickle 
    - return a dataframe: {data filename, wf index, fit parameter values} 
   '''
    if catalogue_filename.endswith('.npz'):
        event_catalogue = np.load(catalogue_filename, allow_pickle=True)
        n_events = event_catalogue['event_counter'].shape[0]
    if catalogue_filename.endswith('.pkl'):
        event_catalogue = pd.read_pickle(catalogue_filename)
        n_events = event_catalogue.shape[0]
    
    wf = event_catalogue['wf']
    mfilter = WFFilter(pyreco_manager.config)
    clean_catalogue = pd.DataFrame(columns= [
                    # 'filename',
                    'event_index',
                    'wf'
    ]
    )

    print(colored(f"Finding clean waveforms", 'green', attrs = ['blink', 'bold']) )

    for event_index in trange(n_events):
        og_wf = wf[event_index]
        flt = np.reshape(mfilter.numba_fast_filter(og_wf), newshape=og_wf.shape)
        mas = pyreco_manager.algos.running_mean(flt, gate=60)
        flt_proc = np.copy(flt[n_channel])                  # TODO: all channels
        flt_proc = flt[n_channel] - mas[n_channel]
        flt_proc = np.where(flt_proc>0,flt_proc, 0)
        rms = pyreco_manager.algos.get_rms(flt_proc)
        flt_above_3rms = np.where(flt_proc > 3*rms, flt_proc, 0)
        # flt_above_3rms = np.where(flt_proc > 3.05*rms, flt_proc, 0)        # diag
        if len(find_peaks(flt_above_3rms)[0]) == 1:
            # clean_cntr += 1
            clean_catalogue = clean_catalogue._append(
                {
                    # 'filename':,
                    'event_index': event_index,
                    'wf': og_wf,
                    'peak_loc': np.ceil(np.mean(np.where(flt_above_3rms != 0))),
                    # prefers to take ceiling value over floor
                }, ignore_index=True
            ) # type: ignore
 
    return clean_catalogue

def pulse_template(t, t0, sigma, tau, scale, baseline, K) -> np.ndarray:
    ''' 
    ARMA template.
    t0 - offset
    K - scalar multiplication factor for matching data
    '''
    return baseline + K*((1-scale)/(sigma*np.sqrt(2*np.pi))*np.exp(-((t-t0)/(sigma))**2/2) + scale*np.heaviside(t-t0,0)/tau*np.exp(-(t-t0)/tau))

def transform_shift_wfs(og_wfs:np.ndarray) -> tuple:
    ''' Transforms array. There should be a vectorized method for this.''' #TODO: vectorize
    wfs = np.copy(og_wfs)
    for _c in range(wfs.shape[0]):
        a = np.min(og_wfs[_c])
        if a < 0:
            wfs[_c] = og_wfs[_c] + np.abs(a)
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
    fit_param_df = pd.DataFrame(columns = ['fit_param_ch2']) #TODO 'fit_param_ch0', 'fit_param_ch1', 

    print(colored(f"Commence fitting", 'green', attrs = ['blink', 'bold']) )
    
    for clean_index in trange(clean_catalogue.shape[0]): #TODO we can do split processing on file
        # wf = clean_catalogue.iloc[clean_index]['wf'] #[n_channel]
        wf = clean_catalogue.iloc[clean_index]['wf'] #[n_channel] # development
        wf = transform_shift_wfs(wf)
        peak_loc = clean_catalogue.iloc[clean_index]['peak_loc']
        fit_end = wf[n_channel].shape[0]
        x_values = np.arange(fit_begin,fit_end)
        # mse_700_800 = np.std(wf[n_channel][700:800])/np.sqrt(wf[n_channel][700:800].shape[0]) # do this for each channel
        p0_input_ch2 = [peak_loc,   2.5,  80.0,   0.95, 0.0, 10000.0]
        fittedparameters_ch2, _pcov = curve_fit(pulse_template, x_values[fit_begin:fit_end], \
                                    wf[n_channel][fit_begin:fit_end], p0 = p0_input_ch2, \
                                    # sigma=mse_700_800*np.ones(wfs[2].shape), \
                                    bounds = ([0, 0, 0, 0, -np.inf, 0], \
                                              [np.inf, 10, np.inf, 1, np.inf, np.inf])
                                    )
        fit_param_df = fit_param_df._append({'fit_param_ch2': fittedparameters_ch2,
                                             'chisqr_ch2': red_chisq(wf[n_channel], pulse_template(x_values, *fittedparameters_ch2), \
                                                                                   fittedparameters_ch2)
                                             }, ignore_index=True) # type: ignore #TODO: repeat all channels
    clean_catalogue = clean_catalogue.join(fit_param_df)
    return clean_catalogue

def main(n_channel:int, save_plot:bool=False, \
        plots_target:int=10) ->None:
    '''
    Steps:
    - call read_wfsco
    - call find_clean_wfs
    - call fit_template
    - plots waveforms and saves to pdf
    - saves DataFrame as pickle file 
    - finally need another function for creating all histogram requested by Marcin 
    - make fit optional
    - turn plotting into a function as well
    '''
    filename = '/work/sarthak/ArgSet/2024_Mar_27/midas/run00061.mid.lz4' #TODO: dynamic path
    # event_catalogue_filename = f'/home/sarthak/my_projects/argset/temp_folder/event_catalogue_run0061.pkl'
    event_catalogue_filename = f'/home/sarthak/my_projects/argset/data/event_catalogue_run00061.npz'
    outfile = 'temp_folder/temp_pyR00061_from_pickle'
    confile = 'argset.ini'

    cmdline_args = f'--config {confile} -o {outfile} -i {filename}'
    pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)

    clean_catalogue_df = find_clean_wfs(pyreco_manager, event_catalogue_filename, \
                                        n_channel)
    clean_catalogue_df = fit_template(clean_catalogue_df)

    output_path = path.join("temp_folder", "argset_wfs_catalogue_100_1000.pkl")
    clean_catalogue_df.to_pickle(output_path)

    if save_plot:
        print(colored(f"Commence plotting", 'green', attrs = ['blink', 'bold']))
        x_values = np.arange(0, 1750)
        for i in trange(plots_target):
            wf = clean_catalogue_df.iloc[i]['wf'] #TODO: uniform use of WF
            # x_values = np.arange(0, wf.shape[0]) #TODO: uniform use of WF
            plt.figure(i, figsize=(8,6))
            plt.title('fit vs data')
            plt.plot(wf[n_channel] + np.abs(np.min(wf[n_channel])), '.--', color='black', \
                     label=f'data Ch: {n_channel}')
            
            fittedparameters_ch2 = clean_catalogue_df.iloc[i]['fit_param_ch2']
            
            plt.plot(pulse_template(x_values, *fittedparameters_ch2), \
        '-', color='red', label='template fit on Ch_2')
            plt.legend()
            plt.savefig(f'temp_folder/midas_wf_{i}.pdf')
            plt.close()

if __name__ == "__main__":
    main(2, save_plot=True, plots_target=100)