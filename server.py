from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import subprocess
import os
from endtoend import separate_from_model
import torch
import numpy
import sys
import audiofile as af
import numpy as np
sys.path.append('tacotron_utils/')
sys.path.append('tacotron_utils/waveglow/')
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
app = Flask(__name__,static_url_path='')


@app.route('/',methods = ['GET'])
def upload_file():
    return render_template('upload.html')
    
@app.route('/', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        print('debug')
        synth_text = request.form.get("synth_text")
        print(synth_text)
        checkpoint_path = "/home/ubuntu/voice_code/tacotron-mod/tacotron2/outdir/checkpoint_from_grid"
        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = model.cuda().eval().half()
        waveglow_path = '/home/ubuntu/voice_code/waveglow/checkpoints/best_22050_base'
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()
        denoiser = Denoiser(waveglow,filter_length=hparams.filter_length, n_overlap=4,
                        win_length=hparams.win_length, mode='zeros',n_mels=hparams.n_mel_channels)
        sequence = np.array(text_to_sequence(synth_text, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=1.0)
        audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        hparams.idx+=1
        af.write(f'audio_result{hparams.idx}.wav',audio_denoised.cpu().numpy(),22050)
        
        return render_template('upload.html', audio_filename=f'audio_result{hparams.idx}.wav')


@app.route('/<audio_file_name>')
def returnAudioFile(audio_file_name):
    path_to_audio_file =  audio_file_name
    return send_file(
         path_to_audio_file, 
         mimetype="audio/wav", 
         as_attachment=True, 
         attachment_filename=audio_file_name)
                                    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
