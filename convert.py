import hydra
import hydra.utils as utils

import pickle
import json
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm
import pyloudnorm
import matplotlib.pyplot as plt
import random
from preprocess import preemphasis
from model import Encoder, Decoder
import os
from extractor_utils import SaveOutput

## SETTINGS ##
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/ZeroSpeech2020/'
DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'
ROOTDIR = '/Users/gt/Documents/GitHub/ZeroSpeech/'

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

speaker_id = 'S015'
run_only_missing_files = False
rand_netw = False

files = [f for f in os.listdir(DATADIR) if os.path.isfile(os.path.join(DATADIR, f))]
wav_files_identifiers = [f for f in files if f.endswith('wav')]
wav_files_paths = [DATADIR + f for f in wav_files_identifiers]


@hydra.main(config_path="config/convert.yaml")
def convert(cfg):
    dataset_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    with open(dataset_path / "speakers.json") as file:
        speakers = sorted(json.load(file))

    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    
    ## Get random network ##
    if rand_netw:
        state_dict = encoder.state_dict()
        state_dict_rand = {}
        print('OBS! RANDOM NETWORK!')
    
        ## The following code was used to generate indices for random permutation ##
        if not os.path.exists(os.path.join(ROOTDIR, 'ZeroSpeech2020_randnetw_indices.pkl')):
            d_rand_idx = {}  # create dict for storing the indices for random permutation
            for k, v in state_dict.items():
                w = state_dict[k]
                idx = torch.randperm(w.nelement())  # create random indices across all dimensions
                d_rand_idx[k] = idx

            with open(os.path.join(ROOTDIR, 'ZeroSpeech2020_randnetw_indices.pkl'), 'wb') as f:
                pickle.dump(d_rand_idx, f)
        else:
            d_rand_idx = pickle.load(open(os.path.join(ROOTDIR, 'ZeroSpeech2020_randnetw_indices.pkl'), 'rb'))
    
        for k, v in state_dict.items():
            w = state_dict[k]
            # Load random indices
            print(f'________ Loading random indices from permuted architecture for {k} ________')
            idx = d_rand_idx[k]
            rand_w = w.view(-1)[idx].view(w.size())  # permute using the stored indices, and reshape back to original shape
            state_dict_rand[k] = rand_w

        # # Test
        # for i, (k, v) in enumerate(state_dict.items()):
        #     plt.plot(state_dict[k].flatten())
        #     plt.plot(state_dict_rand[k].flatten())
        #     plt.show()
            # if i == 5:
            #     break
        #     # assert(state_dict[k].flatten() != state_dict_rand[k].flatten()).all()

        encoder.load_state_dict(state_dict_rand)

    meter = pyloudnorm.Meter(cfg.preprocessing.sr)

    ### LOOP OVER AUDIO FILES ###
    for filename in tqdm(wav_files_paths):
        print(filename)
        encoder.eval()
        decoder.eval()
        
        # Write hooks for the model
        save_output = SaveOutput(rand_netw=rand_netw)
    
        hook_handles = []
        layer_names = []
        for idx, layer in enumerate(encoder.modules()):
            layer_names.append(layer)
            # print(layer)
            if isinstance(layer, torch.nn.modules.ReLU):
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)
        
        wav_path = in_dir / filename
        wav, _ = librosa.load(
            wav_path.with_suffix(".wav"),
            sr=cfg.preprocessing.sr)
        ref_loudness = meter.integrated_loudness(wav)
        wav = wav / np.abs(wav).max() * 0.999

        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)

        # plt.imshow(mel, interpolation=None)
        # plt.show()
        
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1

        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        speaker = torch.LongTensor([speakers.index(speaker_id)]).to(device)
        with torch.no_grad():
            z, _ = encoder.encode(mel)
            output = decoder.generate(z, speaker)

        output_loudness = meter.integrated_loudness(output)
        output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
        
        # Get identifier (sound file name)
        id1 = filename.split('/')[-1]
        identifier = id1.split('.')[0]
        
        path = out_dir / f'{identifier}_{speakers.index(speaker_id)}_randnetw={rand_netw}'
        librosa.output.write_wav(path.with_suffix(".wav"), output.astype(np.float32), sr=cfg.preprocessing.sr)
        
        ## Detach activations
        detached_activations = save_output.detach_activations()

        # Store and save activations
        save_output.store_activations(RESULTDIR=RESULTDIR, identifier=identifier)


if __name__ == "__main__":
    convert()
