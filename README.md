# Formant Estimation and Tracking using Probabilistic Heat-Maps
Yosi Shrem (joseph.shrem@campus.technion.ac.il),\
Felix Kreuk,\
 Joseph Keshet (jkeshet@technion.ac.il).      


FormantsTracker is a software package for Formant Tracking and Estiamtion using deep learning. 

We propose a new modeling for measuring the formants' frequencies using probabilistic heat-maps rather than traditional regression. This technique allows for flexibility in the predictions to support both in-distribution and out-of-distribution (OOD) samples with greater precision.

The paper was present at Interspeech 2022 -  [Formant Estimation and Tracking using Probabilistic Heat-Maps](https://www.isca-speech.org/archive/pdfs/interspeech_2022/shrem22_interspeech.pdf). If you find our work useful please cite :
```
@article{shrem2022formant,
  title={Formant Estimation and Tracking using Probabilistic Heat-Maps},
  author={Shrem, Yosi and Kreuk, Felix and Keshet, Joseph},
  journal={arXiv preprint arXiv:2206.11632},
  year={2022}
}
```






## Installation instructions:
1. First, create a conda virtual environment and activate it:
```
conda create -n FormantsTracker python=3.9 -y
conda activate FormantsTracker
```
2. Then, clone this repository and install dependencies with:
```
git clone https://github.com/MLSpeech/FormantsTracker.git
cd FormantsTracker
pip install -r requirements.txt
```
## How to use: 
You can either set the paths for the run (opt1) or use the default values (opt2).
The generated predictions are for every 10ms frame.

#### Option 1 :
- Provide the path for  ```test_dir``` and ```predictions_dir``` as arguments.
For example:
  ```
  python main.py test_dir=<data_dir_path> predictions_dir=<predictions_dir_path>
  ``` 
- Note: You can also change the default values at ```./conf/config.yaml```.
#### Option 2 :
- Place your ```.wav``` files in the ```./data/ ``` directory. 
- Then, run :
  ```
  python main.py
  ```
- The predictions will be at ```./predictions``` directory.
- Note: You can also place directories that contain the ```.wav``` files, there is no need to re-arrange your data.
For example:
  ```
    ./data
          └───dir1
          │   │   1.wav
          │   │   2.wav
          │   │   3.wav
          │               │   
          └───dir2
              │   1.wav
              │   2.wav
              │   3.wav
  ```


>>>>>>> aa9ef71 (commit message)
