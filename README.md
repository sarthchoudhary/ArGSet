Data processing code for ArGSet.
- Main Scripts:
    - create_event_catalogue.py
        Writes pickle file containing separate DataFrames for each channels.
    - calculate_wf_param.py
        Does all processing from finding clean waveforms to fitting with pulse model. Write results to pickle files.
    - notebook/histogram_wf_param.ipynb
        1. Channelwise concatenation of DataFrames from several runs
        2. Performs histogramming on fit parameters, applies cut, and fit the resultant histogram to gaussian.
        3. Makes Fingerplots from event catalogue generated after updating the ARMA parameters in Pyreco. 

- TODO: re-organization of this repository will happen at some point.  
- For the time being no new code should be added to Scripts/. It is quite messy due to Cython compiled code.
- Move all old notebooks and scripts to Legacy/. 
- All other notebooks are primarily for experimentation and/or documentation.