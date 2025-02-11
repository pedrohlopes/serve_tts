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
sys.path.append('/home/pedro_hleite_g_globo/tacotron2-custom-code-main/')
sys.path.append('/home/pedro_hleite_g_globo/tacotron2-custom-code-main/waveglow/')
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import noisereduce as nr
from num2words import num2words
import re

def fix_periods(str):
    if str == "":                    # Don't change empty strings.
        return str
    if str[-1] in ["?", ".", "!"]:   # Don't change if already okay.
        return str
    if str[-1] == ",":               # Change trailing ',' to '.'.
        return str[:-1] + "."
    return str + "."                 # Otherwise, add '.'.

def convert_helper(obj):
    num_in_text = int(obj.group(0))
    return num2words(num_in_text,lang='pt_BR').replace(",","")

def convert_numbers(text):
    return re.sub(r'\d+',convert_helper,text)



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

waveglow_paths ={}
tacotron_paths = {}
sigmas ={}
denoiser_alphas = {}
tacotron_paths['voz_base'] = '/home/pedro_hleite_g_globo/tacotron2-custom-code-main/latest_voz_base_22'
waveglow_paths['voz_base'] = '/home/pedro_hleite_g_globo/tacotron2-custom-code-main/waveglow_base_22_cond'
tacotron_paths['dirceu'] = "/home/pedro_hleite_g_globo/best_dirceu_22_tacotron" #checkpoint_from_grid01 era o cadu
waveglow_paths['dirceu'] = "/home/pedro_hleite_g_globo/best_dirceu_22_waveglow"
sigmas['voz_base'] = .93
denoiser_alphas['voz_base'] = 0.01
sigmas['dirceu'] = 0.8
denoiser_alphas['dirceu'] = 0.1
tacotron_paths['dirceu_aberto'] = "/home/pedro_hleite_g_globo/taco_dirceu_aberto" #checkpoint_from_grid01 era o cadu
waveglow_paths['dirceu_aberto'] = "/home/pedro_hleite_g_globo/best_dirceu_22_waveglow"
tacotron_paths['dirceu_fechado'] = "/home/pedro_hleite_g_globo/taco_dirceu_fechado"
waveglow_paths['dirceu_fechado'] = "/home/pedro_hleite_g_globo/best_dirceu_22_waveglow"
sigmas['dirceu_aberto'] = 0.8
denoiser_alphas['dirceu_aberto'] = 0.1
sigmas['dirceu_fechado'] = 0.8
denoiser_alphas['dirceu_fechado'] = 0.1

def load_synth_model(tacotron_path,waveglow_path):
    model = load_model(hparams)
    model.load_state_dict(torch.load(tacotron_path)['state_dict'])
    _ = model.cuda().eval().half()
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow,filter_length=hparams.filter_length, n_overlap=4,
                            win_length=hparams.win_length, mode='zeros',n_mels=hparams.n_mel_channels)
    return model, waveglow, denoiser
    
current_model = 'dirceu'
model, waveglow, denoiser = load_synth_model(tacotron_paths[current_model],waveglow_paths[current_model])
current_sigma = 0.93
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
        synth_text = fix_periods(convert_numbers(request.form.get("synth_text")))
        selected_model = request.form.get("select_model")
        print(f'Selected model : {selected_model}')
        print(synth_text)
        if selected_model != current_model:
            current_model = selected_model
            model, waveglow, denoiser = load_synth_model(tacotron_paths[current_model],waveglow_paths[current_model])
            current_sigma = sigmas[current_model]
            current_denoiser_alpha = denoiser_alphas[current_model]
        
                



        
        sequence = np.array(text_to_sequence(synth_text, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=current_sigma)
        audio_denoised = nr.reduce_noise(y=audio[0].data.cpu().numpy(), sr=hparams.sampling_rate, prop_decrease=0.7)
        hparams.idx+=1
        af.write(f'audio_result{hparams.idx}.wav',audio_denoised.astype(np.float32),22050)
        
        return render_template('upload.html', audio_filename=f'audio_result{hparams.idx}.wav')


@app.route('/<audio_file_name>')
def returnAudioFile(audio_file_name):
    path_to_audio_file =  audio_file_name
    return send_file(
         path_to_audio_file,
         as_attachment=True, 
         attachment_filename=audio_file_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9191)
