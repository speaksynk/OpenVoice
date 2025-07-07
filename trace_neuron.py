
################################
# TRACE A NEURON MODEL FOR INF2
# Author: Seth Climenhaga
################################
import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

import pickle
import argparse

from openvoice.api import ToneColorConverter
from openvoice.mel_processing import spectrogram_torch
from openvoice import se_extractor, utils
from openvoice.models import SynthesizerTrn

from melo.api import TTS
import librosa


def options():
    parser = argparse.ArgumentParser(description='trace inferentia args for OpenVoice')
    parser.add_argument('--reference_speaker', "-r", type=str, default='resources/example_reference.mp3')
    parser.add_argument('--generate_target_input', "-g",  action='store_true')
    parser.add_argument('--output_folder', "-f", type=str, default="./output")
    parser.add_argument('--cpu_backend', "-c", action='store_true')
    parser.add_argument('--compiler_work_dir', "-w", type=str, default="./work")

    args = parser.parse_args()
    return args


def main():

    args = options()

    language = "EN_NEWEST"
    audio_src_path = "generated_voice.mp3"

    tts_device = "cpu"
    tts_model = TTS(language=language, device=tts_device)

    speaker_ids = tts_model.hps.data.spk2id

    speaker_key = None
    for sk in speaker_ids.keys():
        speaker_key = sk
        break

    if args.generate_target_input:
        tone_color_converter = ToneColorConverter('checkpoints_v2/converter/config.json', device=tts_device)
        
        speaker_id = speaker_ids[speaker_key]
        text = "MyShell is a decentralized and comprehensive platform for discovering, creating, and staking AI-native apps."
        
        tts_model.tts_to_file(text, speaker_id, audio_src_path, speed=1.0)

        target_se, audio_name = se_extractor.get_se(args.reference_speaker, tone_color_converter, vad=True)

        with open("se_data.pkl", "wb") as f:
            pickle.dump((target_se, audio_name), f)


    hps = utils.get_hparams_from_file('checkpoints_v2/converter/config.json')
    audio, _ = librosa.load(audio_src_path, sr=hps.data.sampling_rate)

    trace_device='xla'

    speaker_file_key = speaker_key.lower().replace('_', '-')
    source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_file_key}.pth', map_location=tts_device)

    model = SynthesizerTrn(len(getattr(hps, 'symbols', [])), hps.data.filter_length // 2 + 1,n_speakers=hps.data.n_speakers, **hps.model).to(trace_device)

    if not args.generate_target_input:
        with open("se_data.pkl", "rb") as f:
            target_se, audio_name = pickle.load(f)

    y = torch.FloatTensor(audio)
    y = y.unsqueeze(0)
    spec = spectrogram_torch(y, hps.data.filter_length,
                            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                            center=False).to(trace_device)
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(trace_device)

    source_se = source_se.to(trace_device)
    target_se = target_se.to(trace_device)


    g_zeros_src = torch.zeros(source_se.shape, dtype=source_se.dtype, device=source_se.device)
    g_zeros_tgt = torch.zeros(target_se.shape, dtype=target_se.dtype, device=target_se.device)


    # force materialization by performing real operations
    _ = spec + 0
    _ = spec_lengths + 0
    _ = source_se + 0
    _ = target_se + 0
    _ = g_zeros_src + 0
    _ = g_zeros_tgt + 0

    xm.mark_step()

    example_inputs = (spec, spec_lengths, source_se, target_se, g_zeros_src, g_zeros_tgt)

    compiler_args=""
    cpu_backend=False

    if args.cpu_backend:
        compiler_args="--target inf2"
        cpu_backend=True

    model_neuron = torch_neuronx.trace(model.voice_conversion, example_inputs, compiler_args=compiler_args, cpu_backend=cpu_backend, compiler_workdir=args.compiler_work_dir)

    # Save the TorchScript for inference deployment
    # filename = 'model.pt'
    # torch.jit.save(model_neuron, filename)


if __name__ == "__main__":
    main()
