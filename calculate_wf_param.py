### resources intensive script: needs 16GB minimum to run
### for interactive shell do this on chuck:
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
import pickle


def find_clean_wfs( pyreco_manager, catalogue_filename:str) -> dict[str, pd.DataFrame]: #TODO: work with different input format 
    '''
    Uses ARMA filter to say whether a wf is clean or not! Works on all channels.
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

    print(colored('loading events...', 'magenta'))
    wf = event_catalogue['wf']
    
    mfilter = WFFilter(pyreco_manager.config)

    wf_str_ls = ['wf_ch0', 'wf_ch1', 'wf_ch2']
    peak_str_ls = ['peak_ch0', 'peak_ch1', 'peak_ch2']
    event_df_ch0 = pd.DataFrame( columns = ['event_counter', 'wf_ch0', 'peak_ch0'])
    event_df_ch1 = pd.DataFrame( columns = ['event_counter', 'wf_ch1', 'peak_ch1'])
    event_df_ch2 = pd.DataFrame( columns = ['event_counter', 'wf_ch2', 'peak_ch2'])
    # event_df_ls = [event_df_ch0, event_df_ch1, event_df_ch2]
    print(colored(f"Finding clean waveforms", 'green', attrs = ['blink', 'bold']) )

    # for event_index in trange(n_events, colour='blue'): #TODO: uncomment
    for event_index in trange(100, colour='blue'): # diag
        og_wf = wf[event_index]
        flt = np.reshape(mfilter.numba_fast_filter(og_wf), newshape=og_wf.shape) # TODO: variable names
        mas = pyreco_manager.algos.running_mean(flt, gate=60)
        flt_proc = np.copy(flt)
        flt_proc = flt - mas
        flt_proc = np.where(flt_proc>0,flt_proc, 0)
        rms = pyreco_manager.algos.get_rms(flt_proc)
        flt_above_3rms = np.where(flt_proc > 3*rms, flt_proc, 0)
        ## flt_above_3rms = np.where(flt_proc > 3.05*rms, flt_proc, 0)        # diag
        for ch in range(3): # problematic
            if len(find_peaks(flt_above_3rms[ch])[0]) == 1: # single peak selector
                event_dict = {
                    'event_counter': event_index+1,
                    wf_str_ls[ch]: og_wf[ch],
                    peak_str_ls[ch]: np.ceil(np.mean(np.where(flt_above_3rms[ch] != 0))),
                    # prefers to take ceiling value over floor
                }

                if ch == 0: #TODO: replace with match-case https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
                    event_df_ch0 = event_df_ch0._append(event_dict, ignore_index=True) # type: ignore
                if ch == 1:
                    event_df_ch1 = event_df_ch1._append(event_dict, ignore_index=True) # type: ignore
                if ch == 2:
                    event_df_ch2 = event_df_ch2._append(event_dict, ignore_index=True) # type: ignore    

    clean_catalogue_dict = {
        # 'filename': catalogue_filename, #TODO
        'ch0': event_df_ch0,
        'ch1': event_df_ch1,
        'ch2': event_df_ch2,       
    }
    ### save clean dict of df as pickle
    print(colored("saving clean catalogue dict as pickle to disk", color = 'blue') )
    try:
        clean_dict_path = path.join("temp_folder", "clean_catalogue_dict.pkl") ##TODO: dynamic
        with open(clean_dict_path, 'wb') as clean_dict_file:
            pickle.dump(clean_catalogue_dict, clean_dict_file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(colored('Error saving clean dict of df as pickle', color='red'))
        print(colored(f'>>> {e}', color='red'))
    return clean_catalogue_dict

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

def fit_template(clean_catalogue:pd.DataFrame, n_channel:int) -> pd.DataFrame:
    '''
    fits template function to waveform.
    - t0 is selected dynamically.
    - add reduced chisqr to fit_catalogue
    - multiple processing can be performed in parallel
    - saves fit catalogue to disk. One catalogue per channel.
    '''
    ch_ls = ['ch0', 'ch1', 'ch2']
    wf_str_ls = ['wf_ch0', 'wf_ch1', 'wf_ch2']
    peak_str_ls = ['peak_ch0', 'peak_ch1', 'peak_ch2']
    ch_str = ch_ls[n_channel]
    wf_ch = wf_str_ls[n_channel]
    peak_ch = peak_str_ls[n_channel]
    fit_begin = 0
    # fit_catalogue = pd.DataFrame(columns = ['fit_param_ch2'] ) #TODO 'fit_param_ch0', 'fit_param_ch1', 
    fit_catalogue = pd.DataFrame()
    print(colored(f"Commence fitting {ch_str}", 'green', attrs = ['blink', 'bold']) )
    
    for clean_index in trange(clean_catalogue.shape[0], colour='blue'): #Optional: we can do split processing on file
        wf = clean_catalogue.iloc[clean_index][wf_ch] # channel specific
        wf = transform_shift_wfs(wf)
        peak_loc = clean_catalogue.iloc[clean_index][peak_ch]
        fit_end = wf.shape[0] # type: ignore
        x_values = np.arange(fit_begin,fit_end)
        # mse_700_800 = np.std(wf[n_channel][700:800])/np.sqrt(wf[n_channel][700:800].shape[0]) # do this for each channel
        p0_dict = {                                                         #TODO: tuning for channels 0 & 1
                    'ch0': [peak_loc,   2.5,  80.0,   0.95, 0.0, 10000.0],
                    'ch1': [peak_loc,   2.5,  80.0,   0.95, 0.0, 10000.0],
                    'ch2': [peak_loc,   2.5,  80.0,   0.95, 0.0, 10000.0],
        }
        p0_input = p0_dict[ch_str]
        try:
            fittedparameters, _pcov = curve_fit(pulse_template, x_values[fit_begin:fit_end], \
                                        wf[fit_begin:fit_end], p0 = p0_input, \
                                        # sigma=mse_700_800*np.ones(wfs[2].shape), \
                                        bounds = ([0, 0, 0, 0, -np.inf, 0], \
                                                [np.inf, 10, np.inf, 1, np.inf, np.inf])
                                        )
            fit_catalogue = fit_catalogue._append({
                                            'event_counter': clean_catalogue.iloc[clean_index]['event_counter'],
                                            'fit_param': fittedparameters,
                                            'chisqr': red_chisq(wf, pulse_template(x_values, *fittedparameters), \
                                                                                fittedparameters)
                                                }, ignore_index=True) # type: ignore #TODO: repeat all channels
        except RuntimeError as e:
            fit_catalogue = fit_catalogue._append({   'fit_param': None,
                                                    'chisqr': None,
            }, ignore_index = True) # type: ignore
            print(colored(f'RuntimeError occured while cueve fitting on {ch_str} for clean index {clean_index}', color='red'))
            print(colored(f'>>> {e}', color='red'))
    
    return fit_catalogue

def fit_all_channels(clean_catalogue_dict: dict, \
                     ch_number_ls:list[int]=[0,1,2]) -> dict:
    '''Runs the fitter over all channels. Saves fit catalogue to disk.
    ch_number_ls -> list of channels numbers to be processed.'''

    ch_name_dict ={0:'ch0', 1:'ch1', 2:'ch2'}
    fit_catalogue_dict = {}
    for ch_i in ch_number_ls:
        ch = ch_name_dict[ch_i]
        fit_catalogue_dict[ch] = fit_template(clean_catalogue_dict[ch], ch_i)
    
    ### save clean dict of df as pickle
    print(colored("saving fit catalogue dict as pickle to disk", color = 'blue') )
    try:
        fit_dict_path = path.join("temp_folder", "fit_catalogue_dict.pkl") ##TODO: dynamic
        with open(fit_dict_path, 'wb') as fit_dict_file:
            pickle.dump(fit_catalogue_dict, fit_dict_file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(colored('Error saving fit dict of df as pickle', color='red'))
        print(colored(f'>>> {e}', color='red'))
    return fit_catalogue_dict

def main(ch_number_ls:list[int], save_plot:bool=False, \
        plots_target:int=10) ->None:
    '''
    Steps:
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

    # clean_catalogue_df = find_clean_wfs(pyreco_manager, event_catalogue_filename, \
    #                                     n_channel)
    # clean_catalogue_df = fit_template(clean_catalogue_df)
    # clean_catalogue_df.to_pickle(clean_catalogue_path)
    
    clean_catalogue_dict = find_clean_wfs(pyreco_manager, event_catalogue_filename)
    
    fit_catalogue_dict = fit_all_channels(clean_catalogue_dict, ch_number_ls)
    # fit_catalogue_dict = fit_all_channels(clean_catalogue_dict, ch_number_ls = [1, 2]) #ch_number_ls should be specified in main
    
    # # save fit df individually as pickle #TODO: remove
    # try:
    #     for ch_i, ch in enumerate([['ch0', 'ch1', 'ch2']]):
    #         fit_catalogue_path = path.join("temp_folder", f"argset_fit_catalogue_ch_{ch}.pkl") #TODO: dynamic
    #         fit_catalogue_dict[ch].to_pickle(fit_catalogue_path)
    # except:
    #     print(f"Error saving fit df individually as pickle")

    if save_plot: #TODO: plotting has to change as well because df is now a dict of df
        print(colored(f"Commence plotting", 'green', attrs = ['blink', 'bold'])) #TODO: plot function
        x_values = np.arange(0, 1750) # TODO: dynamic
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
    # main(2, save_plot=True, plots_target=100)
    # main(2, False)
    # main([1, 2], False)
    main([2], False)