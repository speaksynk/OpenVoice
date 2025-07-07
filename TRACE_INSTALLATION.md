# OpenVoice Inferentia Trace

## Installation

### Pytorch NeuronX Installtion Doc

[https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx)

install rest of dependecies needed for OpenVoice

```
pip install -r cleaned_requirements.txt
```

download averaged_perceptron_tagger_eng

```
python dl_averaged_perceptron_tagger_eng.py
```

Download the checkpoint from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract it to the `checkpoints_v2` folder.

## Trace

first time run with --generate_target_input or -g
to generate target audio and target_se

```
python trace_neuron.py
```

--cpu_backend or -c to trace on CPU
