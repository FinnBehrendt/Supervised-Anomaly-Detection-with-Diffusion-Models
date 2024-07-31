from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.DDPM_encoder import get_encoder
import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any
import torchio as tio
from src.utils.generate_noise import gen_noise
import wandb
from omegaconf import open_dict
from collections import OrderedDict
from src.models.LDM.modules.diffusionmodules.util import timestep_embedding
from src.models.modules.SegUnet import SegUnet as SegUnet 
from src.models.modules.DRAEM.augmenter import Augmenter

class SADM_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        
        self.cfg = cfg
        
        # conditioning net
        if cfg.get('condition',True):
            with open_dict(self.cfg):
                self.cfg['cond_dim'] = cfg.get('unet_dim',64) * 4

            self.encoder, out_features = get_encoder(cfg)
        else: 
            out_features = None


        model = OpenAI_UNet(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = cfg.get('unet_dim',64),
                            out_channels = 1,
                            num_res_blocks = cfg.get('num_res_blocks',3),
                            attention_resolutions = tuple(cfg.get('att_res',[3,6,12])), # 32, 16, 8
                            dropout=cfg.get('dropout_unet',0), # default is 0.1
                            channel_mult=cfg.get('dim_mults',[1, 2, 4, 8]),
                            conv_resample=True,
                            dims=2,
                            num_classes=out_features,
                            use_checkpoint=False,
                            use_fp16=True,
                            num_heads=1,
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=cfg.get('spatial_transformer',False),    
                            transformer_depth=1,                            
                            )
        model.convert_to_fp16()

        timesteps = cfg.get('timesteps',1000)
        sampling_timesteps = cfg.get('sampling_timesteps',timesteps)
        self.test_timesteps = cfg.get('test_timesteps',500) 

        self.diffusion = GaussianDiffusion(
                                            model,
                                            image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)), # only important when sampling
                                            timesteps = timesteps,   # number of steps
                                            sampling_timesteps = sampling_timesteps,
                                            objective = cfg.get('objective','pred_x0'), # pred_noise or pred_x0
                                            channels = 1,
                                            loss_type = cfg.get('loss','l1'),    # L1 or L2
                                            p2_loss_weight_gamma = cfg.get('p2_gamma',0),
                                            cfg=cfg
                                            )
        
        # load diffusion state
        ckpt = self.cfg.get('ddpm_ckpt',None) 
        assert ckpt is not None, "Please provide a checkpoint for the diffusion model. You can train the cDDPM by python run.py experiment=cDDPM/DDPM_cond_spark_2D. Aftwerwards you can use the checkpoint in this SADM model by +model.cfg.ddpm_ckpt=<path_to_cDDPM.ckpt>."
        if ckpt is not None:
            state_dickt = torch.load(ckpt,map_location=self.device)['state_dict']
            self.load_state_dict(state_dict=state_dickt,strict=False)
            print('Loaded model from checkpoint: ', ckpt)
        # freeze weights
        for param in self.diffusion.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Segmentation Network
        self.SegUnet = SegUnet(cfg)
        
        # Anomaly Generator
        if self.cfg.get('gen_ano',False):
            self.AnoGen = Augmenter(p_anomalous=cfg.get('p_anom',1),cfg=cfg)
        
        self.prefix = prefix
        
        self.save_hyperparameters()

    def forward(self, x): # encode input
        if self.cfg.get('condition',True):
            x = self.encoder(x)
        else: 
            x = None
        return x


    def training_step(self, batch, batch_idx: int):
        # process batch
        input = batch['vol'][tio.DATA].squeeze(-1)
        # in case of weakly supervised learning, use the segmentation mask 
        if 'seg_orig' in batch:
            seg = batch['seg_orig'][tio.DATA].squeeze(-1)
        else:
            seg = torch.zeros_like(input)
        seg = torch.where(seg > 0, torch.ones_like(seg), seg)

        if 'seg_orig' in batch:
            slice_ = torch.randint(0, input.shape[-1], (input.shape[0],), device=self.device)
            input = input[torch.arange(input.shape[0]),...,slice_]
            seg = seg[torch.arange(seg.shape[0]),...,slice_]

        # calc features for guidance     
        features = self(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None

        if self.cfg.get('gen_ano',False): # in case of generated anomalies
            # generate anomalies
            input_ano, seg_ano = self.AnoGen.augment_batch(input)
            seg = seg + seg_ano
            # update features for guidance
            features = self(input_ano)
            loss_dm, reco = self.diffusion(input_ano,cond=features, noise = noise)
        else:
            # reconstruct
            input_ano = input
            loss_dm, reco = self.diffusion(input,cond=features, noise = noise)

        # score anomalies
        loss_siam = self.SegUnet.score_diff(reco, input, input_ano, seg, deepsup=self.cfg.get('deepsup',False))
        
        loss = loss_siam

        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        input = batch['vol'][tio.DATA].squeeze(-1) 
        # in case of weakly supervised learning, use the segmentation mask 
        if 'seg_orig' in batch:
            seg = batch['seg_orig'][tio.DATA].squeeze(-1)
        else:
            seg = torch.zeros_like(input)
        seg = torch.where(seg > 0, torch.ones_like(seg), seg)

        if 'seg_orig' in batch:
            slice_ = torch.randint(0, input.shape[-1], (input.shape[0],), device=self.device)
            input = input[torch.arange(input.shape[0]),...,slice_]
            seg = seg[torch.arange(seg.shape[0]),...,slice_]

        # calc features for guidance     
        features = self(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None

        if self.cfg.get('gen_ano',False): # in case of generated anomalies
            # generate anomalies
            input_ano, seg_ano = self.AnoGen.augment_batch(input)
            seg = seg + seg_ano
            # update features for guidance
            features = self(input_ano)
            loss_dm, reco = self.diffusion(input_ano,cond=features, noise = noise)
        else:
            # reconstruct
            input_ano = input
            loss_dm, reco = self.diffusion(input,cond=features, noise = noise)

        # score anomalies
        loss_siam = self.SegUnet.score_diff(reco, input, input_ano, seg, deepsup=self.cfg.get('deepsup',False))
        
        loss = loss_siam

        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []
        # load diffusion state
        ckpt = self.cfg.get('ddpm_ckpt',None)
        if ckpt is not None:
            state_dict = torch.load(ckpt,map_location=self.device)['state_dict']
            # change encoder.encoder to encoder 
            new_statedict = OrderedDict()
            for key in zip(state_dict):
                    new_key = key[0].replace('encoder.encoder','encoder')
                    new_statedict[new_key] = state_dict[key[0]]

            self.load_state_dict(state_dict=new_statedict,strict=False)

        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_seg = data_seg[...,start_slice:start_slice+num_slices]
            data_mask = data_mask[...,start_slice:start_slice+num_slices]
            ind_offset = start_slice
        else: 
            ind_offset = 0 

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)


        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]

        # calc features for guidance
       
        features = self(input)
        features_single = features

        if self.cfg.condition:
            latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
        else: 
            latentSpace.append(torch.tensor([0],dtype=float).repeat(input.shape[0]))

        if self.cfg.get('noise_ensemble',False): # evaluate with different noise levels
            timesteps = self.cfg.get('step_ensemble',[250,500,750]) # timesteps to evaluate
            reco_ensemble = torch.zeros_like(input)
            cam_ensemble = torch.zeros_like(input)
            for t in timesteps:
                # generate noise
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, input.shape).to(self.device)
                else: 
                    noise = None
                loss_diff, reco = self.diffusion(input,cond=features,t=t-1,noise=noise)
                loss_siam = self.SegUnet.score_diff(reco,input,input)     
                cam = loss_siam
                # average loss_siam
                loss_siam = torch.mean(loss_siam)
                loss = loss_siam + loss_diff
                reco_ensemble += reco
                cam_ensemble += cam
            reco = reco_ensemble / len(timesteps) # average over timesteps
            cam = cam_ensemble / len(timesteps)
        else :
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input.shape).to(self.device)
            else: 
                noise = None
            loss_diff, reco = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
            loss_siam = self.SegUnet.score_diff(reco,input,input)     
            cam = loss_siam
            loss_siam = torch.mean(loss_siam)
            loss = loss_siam + loss_diff
        # calculate loss and Anomalyscores
        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(loss_diff.cpu())
        AnomalyScoreReco.append(loss_diff.cpu())

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD
        cam = cam.squeeze().permute(1,2,0) # to HxWxD

       

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))
        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
        AnomalyScoreReg_vol = np.mean(AnomalyScoreReg)
        AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol * AnomalyScoreReg_vol)
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * 0)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol * 0)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        cam = cam.unsqueeze(0)
        cam = cam.unsqueeze(0)

        if cam is not None:
            cam = cam.cpu().float()

        # calculate metrics
        _test_step(self, final_volume.float().cpu(), data_orig.float().cpu(), data_seg.float().cpu(), data_mask.float().cpu(), batch_idx, ID, label, cam=cam) # everything that is independent of the model choice

           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 