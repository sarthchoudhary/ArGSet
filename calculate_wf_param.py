### resources intensive script: needs 16GB minimum to run
### for interactive shell do this on chuck:
### srun --mem=16G -A bejger-grp -p dgx --pty bash
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pyreco.manager.manager import Manager # TODO: Can WFfilter work w/o it?
from pyreco.reco.filtering import WFFilter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
import os
from os import path
from tqdm import trange, tqdm
from termcolor import colored
import pickle
import yaml

def find_clean_wfs( pyreco_manager, catalogue_filename:str, \
                   file_config: dict, name_dict:dict) -> dict[str, pd.DataFrame]: 
    '''
    Uses ARMA filter to detect number of true SiPM pulse peaks
    - reads an event catalogue (either pkl or npz)
    - search clean events 
    - creates a DataFrame for each channel: {event index, wf array, peak location} 
    - returns dictionary of DataFrames
    - writes dictionary to pickle file.
   '''
    print(colored(f'loading events from {catalogue_filename}...', 'magenta'))

    output_folder = file_config['output_folder']   
    file_basename = name_dict['file_basename']
    catalogue_filename = path.join(file_config['data_folder'], catalogue_filename)

    if catalogue_filename.endswith('.npz'):
        event_catalogue = np.load(catalogue_filename, allow_pickle=True)
        n_events = event_catalogue['event_counter'].shape[0]
    if catalogue_filename.endswith('.pkl'):
        event_catalogue = pd.read_pickle(catalogue_filename)
        n_events = event_catalogue.shape[0]

    wf = event_catalogue['wf']
    
    mfilter = WFFilter(pyreco_manager.config)

    wf_str_ls = ['wf_ch0', 'wf_ch1', 'wf_ch2']
    peak_str_ls = ['peak_ch0', 'peak_ch1', 'peak_ch2']
    event_df_ch0 = pd.DataFrame( columns = ['event_counter', 'wf_ch0', 'peak_ch0'])
    event_df_ch1 = pd.DataFrame( columns = ['event_counter', 'wf_ch1', 'peak_ch1'])
    event_df_ch2 = pd.DataFrame( columns = ['event_counter', 'wf_ch2', 'peak_ch2'])
    # event_df_ls = [event_df_ch0, event_df_ch1, event_df_ch2] #TODO: future
    print(colored(f"Finding clean waveforms", 'green', attrs = ['blink', 'bold']) )

    for event_index in trange(n_events, colour='blue'): #TODO: uncomment
    # for event_index in trange(100, colour='blue'): # diag
        og_wf = wf[event_index]
        flt = np.reshape(mfilter.numba_fast_filter(og_wf), newshape=og_wf.shape) # TODO: variable names
        mas = pyreco_manager.algos.running_mean(flt, gate=60)
        flt_proc = np.copy(flt)
        flt_proc = flt - mas
        flt_proc = np.where(flt_proc>0,flt_proc, 0)
        rms = pyreco_manager.algos.get_rms(flt_proc)
        flt_above_3rms = np.where(flt_proc > 3*rms, flt_proc, 0)
        ## flt_above_3rms = np.where(flt_proc > 3.05*rms, flt_proc, 0)        # TODO: rms tuning
        for ch in range(3): ## we always search in all channels.
            if len(find_peaks(flt_above_3rms[ch])[0]) == 1: # single peak selector
                event_dict = {
                    'event_counter': event_index+1,
                    wf_str_ls[ch]: og_wf[ch],
                    peak_str_ls[ch]: np.ceil(np.mean(np.where(flt_above_3rms[ch] != 0))),
                    # I prefer to take ceiling value over floor
                }

                if ch == 0: #TODO: replace with match-case https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
                    event_df_ch0 = event_df_ch0._append(event_dict, ignore_index=True) # type: ignore
                if ch == 1:
                    event_df_ch1 = event_df_ch1._append(event_dict, ignore_index=True) # type: ignore
                if ch == 2:
                    event_df_ch2 = event_df_ch2._append(event_dict, ignore_index=True) # type: ignore    

    clean_catalogue_dict = {
        'filename': file_basename,
        'ch0': event_df_ch0,
        'ch1': event_df_ch1,
        'ch2': event_df_ch2,       
    }
    ### save dict of df of clean event to pickle
    print(colored("saving clean catalogue dict as pickle to disk", color = 'blue') )
    ## TODO: compression for pickle files. https://stackoverflow.com/questions/57983431/whats-the-most-space-efficient-way-to-compress-serialized-python-data
    
    try:
        output_filename = f"clean_catalogue_custom_{file_basename}.pkl"
        clean_dict_path = path.join(output_folder, output_filename)
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

def transform_shift_wf(og_wfs:np.ndarray) -> tuple[np.ndarray, float]:
    ''' Transforms array.'''
    wf = np.copy(og_wfs)
    wf_min = np.min(wf)
    if wf_min < 0:
        wf = wf + np.abs(wf_min)
    else:
        wf_min = 0
    return wf, wf_min

def red_chisq(f_obs: np.ndarray, f_exp: np.ndarray, fittedparameters: np.ndarray) -> float:
    ''' calculates reduced chisquare '''
    chisqr = np.sum((f_obs - f_exp)**2 / f_exp)
    ndf = f_obs.shape[0]
    return chisqr/(ndf -fittedparameters.shape[0]) # type: ignore

def fit_template(clean_catalogue:pd.DataFrame, n_channel:int) -> pd.DataFrame:
    '''
    fits pulse template function to pulses:
    - t0 is taken from peak location in input clean catalogue
    - [NotImplemented] multiple processing can be performed in parallel
    - returns DataFrame with fit parameters and reduced chisqr values
    - saves fit catalogue to disk
    '''
    ch_ls = ['ch0', 'ch1', 'ch2']
    wf_str_ls = ['wf_ch0', 'wf_ch1', 'wf_ch2']
    peak_str_ls = ['peak_ch0', 'peak_ch1', 'peak_ch2']
    ch_str = ch_ls[n_channel]
    wf_ch = wf_str_ls[n_channel]
    peak_ch = peak_str_ls[n_channel]
    fit_begin = 0
    fit_catalogue = pd.DataFrame()
    print(colored(f"Commence fitting {ch_str}", 'green', attrs = ['blink', 'bold']) )
    
    for clean_index in trange(clean_catalogue.shape[0], colour='blue'): #Optional: we can do split processing on file
        wf = clean_catalogue.iloc[clean_index][wf_ch] # channel specific
        wf, wf_min = transform_shift_wf(wf)
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
            
            red_chisqr_value = red_chisq(wf, pulse_template(x_values, *fittedparameters), \
                                        fittedparameters)
            
            fittedparameters[4] = fittedparameters[4] + wf_min # undo baseline shift
            fit_catalogue = fit_catalogue._append({
                                            'event_counter': clean_catalogue.iloc[clean_index]['event_counter'],
                                            wf_ch : wf,
                                            'wf_raw': wf + wf_min,
                                            'fit_param': fittedparameters,
                                            'chisqr': red_chisqr_value,
                                            }, ignore_index=True) # type: ignore
        except RuntimeError as e:
            fit_catalogue = fit_catalogue._append({   'fit_param': None,
                                                    'chisqr': None,
            }, ignore_index = True) # type: ignore
            print(colored(f"RuntimeError occured while curve fitting on {ch_str} for \
                          event counter {clean_catalogue.iloc[clean_index]['event_counter']}", color='red'))
            print(colored(f'>>> {e}', color='red'))
    
    return fit_catalogue

def fit_all_channels(clean_catalogue_dict: dict, file_config: dict, name_dict: dict, \
                     ch_number_ls:list[int]=[0,1,2]) -> dict:
    '''
    Runs the fitter over all channels. Saves fit catalogue to disk.
    ch_number_ls -> list of channel numbers to be processed.
    '''
    output_folder = file_config['output_folder']
    file_basename = name_dict['file_basename']

    ch_name_dict ={0:'ch0', 1:'ch1', 2:'ch2'}
    fit_catalogue_dict = {}
    for ch_i in ch_number_ls:
        ch = ch_name_dict[ch_i]
        fit_catalogue_dict[ch] = fit_template(clean_catalogue_dict[ch], ch_i)
    
    ### save dict of df of fit results to pickle
    print(colored("saving fit catalogue dict as pickle to disk", color = 'blue') )
    if not path.isdir(output_folder):
        os.mkdir(output_folder)
    try:
        output_filename = f"fit_catalogue_custom_{file_basename}.pkl"
        fit_dict_path = path.join(output_folder, output_filename)
        with open(fit_dict_path, 'wb') as fit_dict_file:
            pickle.dump(fit_catalogue_dict, fit_dict_file, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(colored('Error saving fit dict of df as pickle', color='red'))
        print(colored(f'>>> {e}', color='red'))
    return fit_catalogue_dict

def plotter_all(fit_catalogue_dict: dict, ch_number_ls: list[int], \
                file_config:dict, name_dict:dict, plots_target:int = 10) -> None:
    ''' Loops plotting over ch_number_ls '''

    def plotter_ch(fit_catalogue:pd.DataFrame, channel_number:int, plots_target:int) -> None:
        ''' Plot waveform and fit function for individual channel. Quits once plots_target is met.'''

        fit_catalogue = fit_catalogue.dropna()
        
        output_folder = file_config['output_folder']
        file_basename = name_dict['file_basename']
        ch_ls = ['ch0', 'ch1', 'ch2']
        wf_str_ls = ['wf_ch0', 'wf_ch1', 'wf_ch2']
        ch_str = ch_ls[channel_number]
        wf_ch = wf_str_ls[channel_number]
        
        output_folder = output_folder.split(sep = os.sep)[:-1]
        output_folder.append('plots')
        output_folder = f'{os.sep}'.join(output_folder)

        if not path.isdir(output_folder):
            os.mkdir(output_folder)
        
        print(colored(f"Commence plotting: {ch_str}...", 'green', attrs = ['blink', 'bold']))
        plots_target = min(plots_target, fit_catalogue.shape[0])
        
        plot_counter  = 0
        with tqdm(total = plots_target, colour = 'blue') as pbar:
            for plot_index in fit_catalogue.index.values:
                if plot_counter < plots_target:
                    event_counter = fit_catalogue.loc[plot_index]['event_counter']
                    wf = fit_catalogue.loc[plot_index]['wf_raw']
                    x_values = np.arange(0, wf.shape[0])
                    plt.figure(plot_index, figsize=(8,6))
                    plt.title(f"{ch_str} data vs fit") 
                    plt.plot(wf, '.--', color='black', \
                                label=f'data : {ch_str}')
                    fittedparameters = fit_catalogue.loc[plot_index]['fit_param']
                    plt.plot(pulse_template(x_values, *fittedparameters), \
                '-', color='red', label=f'template fit on {ch_str}')
                    text_in_box = AnchoredText(f"reduced chisqr = {fit_catalogue.loc[plot_index]['chisqr']:.2f}", \
                                            loc='upper left')
                    ax = plt.gca()
                    ax.add_artist(text_in_box)
                    plt.legend()
                    
                    output_filename = f"midas_{file_basename}_{ch_str}_{int(event_counter)}.pdf"
                    output_filename = path.join(output_folder, output_filename)
                    plt.savefig(output_filename)
                    plt.close()
                    plot_counter +=1
                    pbar.update(1)

    ch_ls = ['ch0', 'ch1', 'ch2']

    for ch_number in ch_number_ls:
        ch_str = ch_ls[ch_number]
        plotter_ch(fit_catalogue_dict[ch_str], ch_number, plots_target)

def main(file_config: dict, ch_number_ls:list[int], plots_target:int, save_plots:bool=True) ->None:
    '''
    Steps:
    - call find_clean_wfs
    - call fit_template
    - plots waveforms and saves to pdf
    '''
    get_basename = lambda filename: filename.replace('_', '.').split(sep='.')[-2]
    file_config['file_basename'] = list(map(get_basename, file_config['run_catalogue']))

    midas_data_folder = file_config['midas_data_folder']
    get_dataname = lambda filename: path.join(midas_data_folder, f'{filename}.mid.lz4')
    file_config['midas_data_filename'] = list(map(get_dataname, file_config['file_basename']))

    # confile = 'argset.ini'
    confile = 'argset_custom.ini'

    for file_index, event_catalogue_filename in enumerate(file_config['run_catalogue']):
        name_dict = {}
        name_dict['file_basename'] = file_config['file_basename'][file_index]
        midas_data_filename = file_config['midas_data_filename'][file_index]
        
        temp_folder = file_config['temp_folder']
        if not path.isdir(temp_folder):
            os.mkdir(temp_folder)
        outfile = f"{temp_folder}/{name_dict['file_basename']}_from_pickle"

        cmdline_args = f'--config {confile} -o {outfile} -i {midas_data_filename}'
        pyreco_manager = Manager( midas=True, cmdline_args=cmdline_args)
        
        clean_catalogue_dict = find_clean_wfs(pyreco_manager, event_catalogue_filename, file_config, name_dict)
        fit_catalogue_dict = fit_all_channels(clean_catalogue_dict, file_config, name_dict, ch_number_ls)
        
        if save_plots:
            plotter_all(fit_catalogue_dict, ch_number_ls, file_config, name_dict, plots_target)

if __name__ == "__main__":

    analysis_config_file = '/home/sarthak/my_projects/argset/argset_analysis_config.yaml'

    with open(analysis_config_file) as handle:
        try:
            file_config = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            print(exc)    
    
    # main(file_config, ch_number_ls = [0, 1, 2], plots_target=1)
    # main(file_config, ch_number_ls = [0], plots_target=40) # diag
    main(file_config, ch_number_ls = [0, 1, 2], plots_target=10)
