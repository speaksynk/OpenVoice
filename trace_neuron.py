
################################
# TRACE A NEURON MODEL FOR INF2
# Author: Seth Climenhaga
################################
import torch
import torch_neuronx

from openvoice.api import ToneColorConverter
from openvoice.mel_processing import spectrogram_torch
from openvoice import se_extractor, utils
from openvoice.models import SynthesizerTrn

from melo.api import TTS

import librosa

import pickle


output_dir = "./output"
device = "cpu"
language = "EN_NEWEST"

reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
audio_src_path = "generatedvoice.mp3"
save_path = f'{output_dir}/output_crosslingual_{0}.wav'

hps = utils.get_hparams_from_file('checkpoints_v2/converter/config.json')
model = SynthesizerTrn(len(getattr(hps, 'symbols', [])), hps.data.filter_length // 2 + 1,n_speakers=hps.data.n_speakers, **hps.model).to(device)

#load model
checkpoint_dict = torch.load(f'checkpoints_v2/converter/checkpoint.pth', map_location=torch.device(device))
a, b = model.load_state_dict(checkpoint_dict['model'], strict=False)


text = "MyShell is a decentralized and comprehensive platform for discovering, creating, and staking AI-native apps."

tts_model = TTS(language=language, device=device)


speaker_ids = tts_model.hps.data.spk2id

speaker_key = None
for sk in speaker_ids.keys():
    speaker_key = sk
    break
speaker_id = speaker_ids[speaker_key]
speaker_key = speaker_key.lower().replace('_', '-')

# tts_model.tts_to_file(text, speaker_id, audio_src_path, speed=1.0)

source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)

tone_color_converter = ToneColorConverter('checkpoints_v2/converter/config.json', device=device)

# target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)


# # Save to pickle file
# with open("se_data.pkl", "wb") as f:
#     pickle.dump((target_se, audio_name), f)


# Run the tone color converter
encode_message = "@MyShell"

audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)

#run on cuda locally to test
##################################################################################################################
# with torch.no_grad():
#     y = torch.FloatTensor(audio).to("cuda")
#     y = y.unsqueeze(0)
#     spec = spectrogram_torch(y, hps.data.filter_length,
#                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
#                             center=False).to(device)
#     spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)
#     # Run the original PyTorch BERT model on CPU
#     audio = model.voice_conversion(spec, spec_lengths, sid_src=source_se, sid_tgt=target_se, tau=0.3)[0]
##################################################################################################################

with open("se_data.pkl", "rb") as f:
    target_se, audio_name = pickle.load(f)

y = torch.FloatTensor(audio)
y = y.unsqueeze(0)
spec = spectrogram_torch(y, hps.data.filter_length,
                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                        center=False).to('xla')
spec_lengths = torch.LongTensor([spec.size(-1)]).to('xla')

xla_src_se = source_se.to('xla')
xla_tar_se = target_se.to('xla')


print(f'spec {spec.type()} spec_lengths {spec_lengths.type()} xla_src_se {xla_src_se.type()} xla_tar_se {xla_tar_se.type()}')

model.to('xla')

# print(f"zero_g {model.zero_g}")
print(f'spec {spec.shape} spec_lengths {spec_lengths.shape} xla_src_se {xla_src_se.shape} xla_tar_se {xla_tar_se.shape}')

g_zeros_src = torch.zeros(xla_src_se.shape, dtype=xla_src_se.dtype, device=xla_src_se.device)
g_zeros_tgt=g2= torch.zeros(xla_tar_se.shape, dtype=xla_tar_se.dtype, device=xla_tar_se.device)

print("Device:", g_zeros_src.device)
print("Shape:", g_zeros_src.shape)
print("Sum:", g_zeros_src.sum()) 

print("Device:", g_zeros_tgt.device)
print("Shape:", g_zeros_tgt.shape)
print("Sum:", g_zeros_tgt.sum()) 


import torch_xla.core.xla_model as xm

# force materialization by performing real operations
_ = spec + 0
_ = spec_lengths + 0
_ = xla_src_se + 0
_ = xla_tar_se + 0
_ = g_zeros_src + 0
_ = g_zeros_tgt + 0

xm.mark_step()

# for name, t in [
#     ("spec", spec),
#     ("spec_lengths", spec_lengths),
#     ("xla_src_se", xla_src_se),
#     ("xla_tar_se", xla_tar_se),
#     ("g_zeros_src", g_zeros_src),
# ]:
#     print(f"{name}: shape={t.shape}, dtype={t.dtype}, device={t.device}, type={type(t)}")

# print(f'spec {spec.shape} spec_lengths {spec_lengths.shape} xla_src_se {xla_src_se.shape} xla_tar_se {xla_tar_se.shape} g_zeros_src {g_zeros_src.shape}, g_zeros_tgt {g_zeros_tgt.shape}')
# # print(f'HERE ------------------ > g_zeros_tgt {g_zeros_tgt} ')

# def register_hooks(model):
#     for name, module in model.named_modules():
#         def hook_fn(module, input, output):
#             print(f"[Hook] {name}: {type(module).__name__}")
#         module.register_forward_hook(hook_fn)

# # register_hooks(model)

# # spec, spec_lengths, xla_src_se, xla_tar_se, g_zeros_src, g_zeros_tgt, 

# # Compile the model for Neuron
model_neuron = torch_neuronx.trace(model.voice_conversion, (spec, spec_lengths, xla_src_se, xla_tar_se, g_zeros_src,g_zeros_tgt), compiler_workdir='/home/ec2-user/work')
