Data processing codes for ArGSet. 

**About ArGSet**
Argon Gas Setup for measurement of wavelength shifting materials at cryogenic temperature.
\
[Learn about ArGSet here](https://gitlab.camk.edu.pl/mkuzniak/cezamat/-/wikis/ArgSet)

**Requirements**\
To satify all requirements do these:
1. rename jar_*.yml to jar.yml
2. edit jar.yml to change prefix a/c to your miniconda installation.
3. `conda env create -f jar.yml`
4. This will not install peakdetect package which needs to be installed via pip. Also, peakdetect is outdated package, its source code needs to tweaked slightly to make use of newer versions of scipy FFT. Pyreco has separate installation steps.

**Data Processing:**\
Main Scripts:
- create_event_catalogue.py
    Writes pickle file containing separate DataFrames for each channels.
- calculate_wf_param.py
    Does all processing from finding clean waveforms to fitting with pulse model. Write results to pickle files.
- notebooks/histogram_wf_param.ipynb
    1. Channelwise concatenation of DataFrames from several runs
    2. Performs histogramming on fit parameters, applies cut, and fit the resultant histogram to gaussian.
    3. Makes Fingerplots from event catalogue generated after updating the ARMA parameters in Pyreco. 

**Future:**\
Re-organization of this repository will happen at some point.
- All files will be re-organized and put into folders a/c projects and not their file type.
- For the time being no new code should be added to 'Scripts/'.
- All Cython code will be moved to 'Legacy/cython_code'.
- Rest of scripts will be moved to 'Legacy/'.
- Rest of notebooks will be moved to 'Legacy/notebooks'.
- Notebooks outside project related folders should only be for experimentation and/or documentation.
