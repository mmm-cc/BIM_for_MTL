# BIM_for_MTL

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

        You can download the pretrained models from

4. Evaluation. You can run the following command,
   
   ```shell
   python -m torch.distributed.launch --nproc_per_node 1 main.py --run_mode infer --config_exp ./configs/mtmamba_nyud.yml --trained_model ./ckpts/mtmamba_nyud.pth.tar
   ```

Acknowledgement
---------------

We would like to thank the authors that release the public repositories:  [Multi-Task-Transformer](https://github.com/prismformore/Multi-Task-Transformer), [Mamba](https://github.com/state-spaces/mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [MTMamba](https://github.com/EnVision-Research/MTMamba/tree/main).


## Citation

If you found this code/work to be useful in your own research, please cite the following:

```latex
@inproceedings{cao2025bim,
  title={Enhancing Mamba Decoder with Bidirectional Interaction in Multi-Task Dense Prediction},
  author={Cao, Mang and Zhou, Sanping and Li, Yizhe and Deng, Ye and Huang, Wenli and Wang, Le},
  booktitle={International Conference on Computer Vision},
  year={2025}
}
```
