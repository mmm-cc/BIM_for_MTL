# BIM_for_MTL
Enhancing Mamba Decoder with Bidirectional Interaction in Multi-Task Dense Prediction (ICCV2025)

This repository contains codes and models for the following papers:


> Mang Cao, Sanping Zhou, Yizhe Li, Ye Deng, Wenli Huang, Le Wang. Enhancing Mamba Decoder with Bidirectional Interaction in Multi-Task Dense Prediction. In *International Conference on Computer Vision*, 2025.


## Requirements

- PyTorch 2.0.0

- timm 0.9.16

- mmsegmentation 1.2.2

- mamba-ssm 1.1.2

- CUDA 11.8
  
  

## Usage

1. Prepare the pretrained Swin-Large checkpoint by running the following command
   
   ```shell
   cd pretrained_ckpts
   bash run.sh
   cd ../
   ```

2. Download the data from [PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab), [NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c), and then extract them. You need to modify the dataset directory as ```db_root``` variable in ```configs/mypath.py```.

3. Train the model. Taking training NYUDv2 as an example, you can run the following command
   
   ```shell
   python -m torch.distributed.launch --nproc_per_node 8 main.py --run_mode train --config_exp ./configs/mtmamba_nyud.yml 
   ```

        You can download the pretrained models from [mtmamba_nyud.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/EdP6lzTOEIRLggFVLlbzPWUBZrsRPoEkdtNpYjm_H2K54A?e=IwsaaG), [mtmamba_pascal.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/ET0zoRo2mq9OoYJlHZZy2eQB5lh6W-yayKzih6ejwD7awQ?e=DUZFGE), [mtmamba_cityscapes.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/EVfY4W2qn85Ihe8rANBiKisBM0xxGn4OnmuOjRJ9FWNGeA?e=TsyE5B), [mtmamba_plus_nyud.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/Ecjm9MJ5SwBGlPfg4YAxGGABagrzm81LM_TI3h6jADkpvA?e=KePvfD), [mtmamba_plus_pascal.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/EaVpHcqrNihIsfyMeyPR614BpzSrk2ubRSIdBUHLcwZTjA?e=DpRajc), [mtmamba_plus_cityscapes.pth.tar](https://hkustgz-my.sharepoint.com/:u:/g/personal/blin241_connect_hkust-gz_edu_cn/EZHHVmXbGChFsvyorMKOvncBU06opYPC0FuVCg8X8Yg8gw?e=8lnvdI).

4. Evaluation. You can run the following command,
   
   ```shell
   python -m torch.distributed.launch --nproc_per_node 1 main.py --run_mode infer --config_exp ./configs/mtmamba_nyud.yml --trained_model ./ckpts/mtmamba_nyud.pth.tar
   ```

Acknowledgement
---------------

We would like to thank the authors that release the public repositories:


## Citation

If you found this code/work to be useful in your own research, please cite the following:

```latex

```
