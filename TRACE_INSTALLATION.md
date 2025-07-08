# OpenVoice Inferentia Trace

## Installation

### PyTorch NeuronX Installtion Doc

[https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx)

Install rest of the dependencies needed for OpenVoice

```
pip install --no-deps -r cleaned_requirements.txt
```

Download the averaged_perceptron_tagger_eng:

```
python dl_averaged_perceptron_tagger_eng.py
```

Download the checkpoint from [here](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip) and extract it to the `checkpoints_v2` folder.

```
python -m unidic download
```

## Trace

first time run with **--generate_target_input** or **-g**
to generate target audio and target_se

```
python trace_neuron.py
```

> **--cpu_backend** or **-c** to trace on CPU
