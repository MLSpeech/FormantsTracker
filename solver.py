import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import  collate_fn_padd,get_test_dataset
import utils
from model import FormantTracker
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Solver:
    def __init__(self,hp):
        self.hp = hp
        #saved ckpt supports 257 bins -> default n_fft=512 
        self.n_bins = (self.hp.n_fft//2)+1
        self.bin_resolution = (self.hp.sample_rate/2)/self.n_bins
        self.test_loader = None
        self.init_test_loader()
        self.model= None
        self.build_model()
    
    
    def init_test_loader(self):
        print("Initializing test loader")
        test_dataset = get_test_dataset(self.hp)
        self.test_loader = DataLoader(test_dataset,batch_size=self.hp.test_batch_size,shuffle=False,
                                      collate_fn=collate_fn_padd, num_workers=self.hp.num_workers)
        print("Test loader initialized")

    def build_model(self):
        print("Building model")
        self.model = FormantTracker(self.hp)
        self.model.load_state_dict(torch.load(self.hp.ckpt))
        self.model = self.model.to(self.hp.device)
        self.model.eval()
        print("Model built")
    
    def test(self):
        print("Testing")
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                spects,lengths,fnames = batch
                spects, lengths = spects.to(self.hp.device), lengths.to(self.hp.device)
                out,_ = self.model(spects)
                predictions=self.get_predicted_formants(out)
                self.write_predictions(predictions,lengths,fnames)
        print(f"Predictions dir - [{self.hp.predictions_dir}]")
            
    def get_predicted_formants(self,out):
        #smooth output
        kernel = utils.get_smoothing_kernel(self.hp.gaussian_kernel_size,self.hp.gaussian_kernel_sigma).to(out.device)
        out_smoothed= F.conv2d(out.view(-1,1,out.shape[2],out.shape[3]), kernel, padding=(self.hp.gaussian_kernel_size-1)//2).view(-1,3,out.shape[2],out.shape[3])
        
        # prediction is set to be the median value corresponding to the arg_max bin
        return ((torch.max(out_smoothed, dim=3)[1])*self.bin_resolution)+self.bin_resolution/2  
            
    def write_predictions(self,predictions,lengths,fnames):
            for fname,length,pred in zip(fnames,lengths,predictions):
                pred =pred[:length]
                pred_fname =os.path.join(self.hp.predictions_dir,fname[len(self.hp.test_dir)+1:]).replace('.wav','.pred')
                os.makedirs(os.path.dirname(pred_fname),exist_ok=True)
                with open(pred_fname,'w') as f:
                    for i in range(length):
                        f.write(f"{i/100}\t{int(pred[0][i])}\t{int(pred[1][i])}\t{int(pred[2][i])}\n")
                if self.hp.plot_formants:
                    # Read prediction file
                    times, f1, f2, f3 = np.loadtxt(pred_fname, unpack=True)

                    # Plot formants and optionally spectrogram
                    plt.figure(figsize=(10, 6))
                    plt.plot(times, f1, label='F1', color='r')
                    plt.plot(times, f2, label='F2', color='g')
                    plt.plot(times, f3, label='F3', color='b')

                    plt.xlabel('Time [sec]')
                    plt.ylabel('Frequency [Hz]')
                    plt.title(f'Formants for {fname}')
                    plt.legend()
                    # Save plot as PNG file in the predictions directory
                    png_filename = os.path.join(pred_fname.replace('.pred', '.png'))
                    plt.savefig(png_filename)
                    plt.close()


