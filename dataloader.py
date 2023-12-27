import torch
from torch.utils.data import Dataset
from boltons.fileutils import iter_find_files
import torchaudio as taudio
from scipy import signal

def collate_fn_padd(batch):
    spects = [t[0] for t in batch]
    lengths = [t[1] for t in batch]
    fnames = [t[2] for t in batch]

    # pad and stack
    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True) #batch_first = first dim will be #batch
    lengths = torch.LongTensor(lengths)

    return padded_spects,lengths,fnames#, padded_formants, padded_heatmaps, phonemes, lengths, fnames, masked_phonemes, n_formants, dataset_str

def preemphasis(x, coeff=0.97):
    return torch.from_numpy(signal.lfilter([1, -coeff], [1], x)).float()

def extract_features(wav_file, hp):
    wav, sr = taudio.load(wav_file)
    # If SR is not 16kHz, resample
    if sr != 16000:
        print(f"Resampling {wav_file} from {sr} to 16000")
        wav = taudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000

    # Pre-emphasis
    if hp.emph>0:
        wav=preemphasis(wav,coeff=hp.emph)
    
    spect = taudio.transforms.Spectrogram(n_fft=hp.n_fft,
                                          win_length=hp.n_fft,
                                          hop_length=sr//100,
                                          power=2,
                                          normalized=hp.normalize)(wav)
    spect = torch.transpose(spect, 1, 2)[0]
    return spect

def get_test_dataset(hp):
    return WavDataset(hp)
    
class WavDataset(Dataset):
    def __init__(self,hp):
        self.wavs=list(iter_find_files(hp.test_dir, "*.wav"))
        self.test_dir= hp.test_dir
        self.hp = hp
    
    def __getitem__(self, index):
        spect = extract_features(self.wavs[index],self.hp)
        return spect,spect.shape[0],self.wavs[index] #spect,lenght,fname
    
    def __len__(self):
        return len(self.wavs)
    