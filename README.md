
# nwb_datajoint

## Setup 
### Instructions as of 08-12-2020

1. Clone this repository
2. Set up and activate the conda environment: 
   ```bash
   conda env create -f environment.yml
   conda activate nwb_datajoint_test
   ```
3. Install the repo: `python setup.py develop`
   - Ignore the error `error: numba 0.50.1 is installed but numba<0.49,>=0.37.0 is required by {'datashader'}`
4. [Use the datajoint docker installation](https://tutorials.datajoint.io/setting-up/local-database.html)

5. Temporary development packages:
    
    hdmf: 
    
    	pip install -U hdmf --find-links https://github.com/hdmf-dev/hdmf/releases/tag/latest --no-index
    
    pynwb:
    
    	pip install -U pynwb --find-links https://github.com/NeurodataWithoutBorders/pynwb/releases/tag/latest --no-index
    
    spikeextractors (pip not currently up to date):
    
        pip uninstall spikeextractors
	
        git clone https://github.com/SpikeInterface/spikeextractors.git
	
        cd spikeextractors
	
        python setup.py develop
	
        cd ..
    
    ndx-franklab-novela (conda version too restrictive)
    
        git clone https://github.com/NovelaNeuro/ndx-franklab-novela.git
	
        cd ndx-franklab-novela
	
        python setup.py develop
	
        cd ..

    ml-ms4alg

        git clone https://github.com/LorenFrankLab/ml_ms4alg.git
	
        cd ml_ms4alg
	
        python setup.py develop
	
        cd ..
			
6. Run `populate_from_NWB.ipynb` from the notebooks directory.
7. Other examples: 
	NWB_DJ_LFP.ipynwb to extract LFP 
	nwbdj_spikeinterface.pynwb to setup and run spikesorting

