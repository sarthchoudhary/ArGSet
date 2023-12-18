import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')
from CAENReader import DataFile 
import argparse
# import sys # alternative to argparse

# @jit(nopython=True)
def process_over_entire_run(data_path:str, run_folder_name:str, 
                            channel_list:list, segment_index_begin:int=0, 
                            segment_index_end:int=4, t0:int=2) -> dict:
    '''
    Do baseline subtraction and integrate over the pulse window. 
    Return a dictionary of the format {'tr0': ndarray of integration values for channel 0}
    '''
    
    # t0 = 2 # how many bins to skip

    pretrigger_begin = t0
    pretrigger_end = 500

    pulse_integration_window_begin = 500
    pulse_integration_window_end = 700
    
    # integration_values_array_tr0 = np.array([])
    # integration_values_array_tr1 = np.array([])
    # integration_values_array_tr2 = np.array([])
    
    # cntr = 0 # diag
    # not_4070_cntr = 0 # diag
    
    # segment_index_begin = 0 # index of first binary segment in the run folder
    # segment_index_end = 4 # index of the last binary segment

    integration_vector_dict = { 
                                'tr0': np.array([]),
                                'tr1': np.array([]),
                                'tr2': np.array([]),
                                }
    
    for i in range(segment_index_begin, segment_index_end+1):
        binary_segment_name = f"{data_path}/{run_folder_name}/run0_raw_b0_seg{i}.bin"
        open_file = DataFile(binary_segment_name)
        while True:
            trigger_r = open_file.getNextTrigger()
            if trigger_r is None:
                break
            # cntr += 1 # diag
            trace_tr0 = trigger_r.traces['b0tr0'] ## do we need to create 3 variables?
            trace_tr1 = trigger_r.traces['b0tr1']
            trace_tr2 = trigger_r.traces['b0tr2']
            # open_file.close()
            for ch in channel_list:
                trace_ch = [trace_tr0, trace_tr1, trace_tr2][ch]
                # if len(trace_ch) == 4070:                                            ## accepting waveforms with length 4070
                if len(trace_ch) > 2:                                                  ## accepting waveforms with length as small as 3
                    
                    baseline_subtracted_ch = trace_ch - np.mean(trace_ch[pretrigger_begin : pretrigger_end])
                    pulse_win_integration_ch = np.sum(baseline_subtracted_ch[pulse_integration_window_begin:pulse_integration_window_end])      ## sum or integrate?
                    integration_vector_dict[f"tr{ch}"] = np.append(integration_vector_dict[f"tr{ch}"], pulse_win_integration_ch)
                # else: # diag
                    # not_4070_cntr += 1 # diag
                                
    return integration_vector_dict

def plot_data(run_folder_name:str, integration_vector_dict:np.array, t0:int=2)->None:
    colour_ch = {'tr0': 'C0', 'tr1': 'C1', 'tr2': 'C2'}
    for ch_key in integration_vector_dict:
        if integration_vector_dict[ch_key].shape[0] > 0:
            plt.figure(figsize=(12,8))
            plt.hist(integration_vector_dict[ch_key][t0:], bins=np.arange(0, 5500, 10), color=colour_ch[ch_key], label=ch_key)
            # plt.plot(np.arange(500, 700), integration_vector_dict_l['tr1'][500:700], color='red', label='pulse integration window')
            plt.yscale('log')
            plt.grid(which='both')
            plt.legend()
            plt.title('Histogram of charge values dervied from integrating ADC values over pulses')
            plt.suptitle(run_folder_name)
            plt.savefig(f"output_folder/{run_folder_name}_{ch_key}_hist_integration.pdf")
            plt.close()

def argument_collector() ->argparse.Namespace:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--data_path")
    argParser.add_argument("-f", "--run_folder_name")
    argParser.add_argument("-c", "--channel_list")
    argParser.add_argument("-t0", "--t0", default=2, help='skip t0 initial bins')
    args = argParser.parse_args()
    return args

def main() -> None:
    # t0 = 2
    ### using argparse
    args = argument_collector()
    channel_list = eval(args.channel_list)
    print('Data is being processed')
    integration_vector_dict = process_over_entire_run(args.data_path, args.run_folder_name, channel_list)
    print('Data is being plotted')
    plot_data(args.run_folder_name, integration_vector_dict)

    ### using sys
    # data_path = '/work/sarthak/ArgSet/'
    # run_folder_name = sys.argv[1]
    # channel_list = eval(sys.argv[2])
    # t0 = int(sys.argv[3])
    # print('Data is being processed')
    # for run_nr in range(int(sys.argv[4]), int(sys.argv[5])+1):
    #     run_folder_name = f"Run_{run_nr}"
    #     print(f"processing Run {run_nr}")
    #     integration_vector_dict = process_over_entire_run(data_path, 
    #                                                     run_folder_name, 
    #                                                     channel_list, t0)
    #     print(f'Plotting {run_nr}')
    #     plot_data(run_folder_name, integration_vector_dict, t0)

if __name__ == "__main__":
    main()

# execute like this: python read_analyze_ArgSet_data.py -p /mnt/e/ArGSet/data/2023_12_15/  -f Run_16 -c [0,1,2]