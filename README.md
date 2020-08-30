
# nwb_datajoint

## Setup 
### Instructions as of 08-28-2020
1. Clone repository
2. Setup conda environment
```conda env create -f environment.yml```
3. Install the repo ```python setup.py develop``` 
4. [Use the datajoint docker installation](https://tutorials.datajoint.io/setting-up/local-database.html)

5. Temporary development packages:
    
 
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
			
6. Run populate_from_NWB.ipynb from the notebooks directory.
7. Other examples: 
	NWB_DJ_LFP.ipynwb to extract LFP 
	nwbdj_spikeinterface.pynwb to setup and run spikesorting
a
