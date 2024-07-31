from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import src.datamodules.create_dataset as create_dataset


class Brats21(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats21.IDs.val
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats21'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1',cfg.mode).str.replace('FLAIR.nii.gz',f'{cfg.mode.lower()}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)




class ATLAS_v2(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(ATLAS_v2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}

        self.csvpath_val = cfg.path.ATLAS_v2.IDs.val
        self.csvpath_test = cfg.path.ATLAS_v2.IDs.test

        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'ATLAS_v2'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

class IXI_syn(LightningDataModule): # IXI with synthetic anomalies
    
    def __init__(self, cfg, ano, fold = None):
        super(IXI_syn, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.ano = ano 
        self.imgpath = {}
        self.csvpath_val = cfg.path.IXI.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI.IDs.test
        self.csv = {}
        states = ['val','test']
        if cfg.mode == 't2' or cfg.cold_diff:
            keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2) 
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        ano = ano.split('.')[1].split('art_')[-1]
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI_syn_' + ano
            if 'DRAEM' in ano:
                self.csv[state]['img_path'] = cfg.path.pathBase + f'/Data/Train/IXI_uniso/t1/' + self.csv[state]['img_path'].str.split('/').str[-1]
                self.csv[state]['mask_path'] = cfg.path.pathBase + f'/Data/Train/IXI_uniso/mask/' + self.csv[state]['mask_path'].str.split('/').str[-1]
                self.csv[state]['seg_path'] = None
            else:
                self.csv[state]['img_path'] = cfg.path.pathBase + f'/Data/Test/IXI_syn/t1/{ano}/' + self.csv[state]['img_path'].str.split('/').str[-1]
                self.csv[state]['mask_path'] = cfg.path.pathBase + f'/Data/Train/IXI_uniso/mask/' + self.csv[state]['mask_path'].str.split('/').str[-1]
                self.csv[state]['seg_path'] = cfg.path.pathBase + f'/Data/Test/IXI_syn/t1/{ano}/' + self.csv[state]['img_path'].str.split('/').str[-1].str.replace('_t1.nii.gz','_seg.nii.gz')

            if cfg.mode == 't2':
                self.csv[state] = self.csv[state][self.csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1','t2')
                self.csv[state]['mask_path'] = self.csv[state]['mask_path'].str.replace('t1','t2')
                self.csv[state]['seg_path'] = self.csv[state]['seg_path'].str.replace('t1','t2')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                self.val= create_dataset.Eval(self.csv['val'][0:2],self.cfg, ano = self.ano)
                self.test = create_dataset.Eval(self.csv['test'][0:2],self.cfg, ano = self.ano)
            else: 
                self.val = create_dataset.Eval(self.csv['val'],self.cfg, ano = self.ano)
                self.test= create_dataset.Eval(self.csv['test'],self.cfg, ano = self.ano)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
