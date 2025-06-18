# MES-Diff-Federated-Probabilistic-Multi-energy-Load-Forecasting
>This work proposes a secure federated multi-energy load probabilistic forecasting framework utilizing diffusion models. The local model and server model quantify the aleatoric and epistemic uncertainty, respectively, based on forecast residuals. A transformer-based diffusion model is designed as the server model to separately process exogenous and endogenous features. The security of the proposed VFL framework, as demonstrated by the noise filtering capacity of the diffusion model, has been evaluated.

Codes for the paper "Secure Federated Probabilistic Load Forecasting for Multi-energy Systems with Diffusion Model". 

Authors: Yangze Zhou*, Xiaorong Wang*, Zhixian Wang, Nan Lu, Yi Wang（* These authors contributed equally to this work).

## Requirements
The must-have packages can be installed by running
```
pip install requirements.txt
```
```
conda env create -f environment.yml
```

## Data
The load data applied in this work is selected from the Tempe campus of Arizona State University. This dataset contains the electricity, steam, and chillwater loads with 1-hour resolution. The weather data are obtained from the National Renewable Energy Laboratory. The dataset spans the period from 2016 to 2019. The combined dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1BlgvRejA1p6mDQ7ScUPMkvRQTbWEztqU/view?usp=drive_link).

## Reproduction
If you want to run the proposed approach, you can run ```run.ipynb```. Other benchmarks can also be realized by changing ```args.local_model``` and ```args.server_model```.

If you want to test the robustness of the diffusion model to noise, you can run ```run_noise.ipynb``

If you want to run the central methods, you can run ```run_central.ipynb```.

