from tkinter import W
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import subprocess
import os
import torch

import numpy
import sys
import audiofile as af
import numpy as np
sys.path.append('/home/ubuntu/voice_code/tacotron-mod/tacotron2/')
sys.path.append('/home/ubuntu/voice_code/tacotron-mod/tacotron2/waveglow/')
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
hparams = create_hparams()

hparams.sampling_rate=22050
hparams.filter_length=1024
hparams.hop_length=256
hparams.win_length=1024
hparams.n_mel_channels=80
hparams.mel_fmin=0.0
hparams.mel_fmax=8000.0
hparams.global_mean = None
hparams.max_decoder_steps=2000
hparams.idx = 0
tacotron_path_base = "/home/ubuntu/voice_code/tacotron-mod/tacotron2/outdir/checkpoint_from_grid02"
waveglow_path_base = '/home/ubuntu/voice_code/waveglow/checkpoints/best_22050_base'
tacotron_path_cadu = "/home/ubuntu/voice_code/tacotron-mod/tacotron2/outdir/checkpoint_from_grid01"
waveglow_path_cadu = '/home/ubuntu/voice_code/waveglow_grid/waveglow/checkpoints/waveglow_18900'
def load_synth_model(tacotron_path,waveglow_path):
    model = load_model(hparams)
    model.load_state_dict(torch.load(tacotron_path)['state_dict'])
    _ = model.cuda().eval().half()
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    current_model = 'base'
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow,filter_length=hparams.filter_length, n_overlap=4,
                            win_length=hparams.win_length, mode='zeros',n_mels=hparams.n_mel_channels)
    return model, waveglow, denoiser
    
model, waveglow, denoiser = load_synth_model(tacotron_path_base,waveglow_path_base)
current_model = 'voz_base'
current_sigma = 0.8
current_denoiser_alpha = 0.01

app = Flask(__name__,static_url_path='')


@app.route('/',methods = ['GET'])
def upload_file():
    return render_template('upload.html')
    
@app.route('/', methods = ['GET', 'POST'])
def upload_file2():
    global current_model, model, current_denoiser_alpha,current_sigma, waveglow, denoiser
    if request.method == 'POST':
        print('debug')
        synth_text = request.form.get("synth_text")
        selected_model = request.form.get("select_model")
        print(f'Selected model : {selected_model}')
        print(synth_text)
        if selected_model != current_model:
            current_model = selected_model
            if selected_model == 'voz_base':
                model, waveglow, denoiser = load_synth_model(tacotron_path_base,waveglow_path_base)
                current_sigma = 0.8
                current_denoiser_alpha = 0.01  
            elif selected_model=='cadu_2h':
                model, waveglow, denoiser = load_synth_model(tacotron_path_cadu,waveglow_path_cadu)
                current_sigma = 0.7
                current_denoiser_alpha = 0.1 
        
                



        
        sequence = np.array(text_to_sequence(synth_text, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=current_sigma)
        audio_denoised = denoiser(audio, strength=current_denoiser_alpha)[:, 0]
        hparams.idx+=1
        af.write(f'audio_result{hparams.idx}.wav',audio_denoised.cpu().numpy(),22050)
        
        return render_template('upload.html', audio_filename=f'audio_result{hparams.idx}.wav')


@app.route('/<audio_file_name>')
def returnAudioFile(audio_file_name):
    path_to_audio_file =  audio_file_name
    return send_file(
         path_to_audio_file,
         as_attachment=True, 
         attachment_filename=audio_file_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
