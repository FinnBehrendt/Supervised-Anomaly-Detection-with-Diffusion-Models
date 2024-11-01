from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd


class IXI(LightningDataModule):

    def __init__(self, cfg, fold = None):
        super(IXI, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        # IXI

        self.cfg.permute = False # no permutation for IXI


        self.imgpath = {}
        self.csvpath_train = cfg.path.IXI.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI.IDs.test
        self.csv = {}
        states = ['train','val','test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI'


            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                self.train = create_dataset.Train(self.csv['train'][0:50],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'][0:50],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8],self.cfg)
            else: 
                self.train = create_dataset.Train(self.csv['train'],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast',False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
      
IXI_syn = IXI # dummy class 

class ATLAS_v2(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(ATLAS_v2, self).__init__()
        self.cfg = cfg
        self.preload = True
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.ATLAS_v2.IDs.val
        self.csvpath_test = cfg.path.ATLAS_v2.IDs.test

        self.csv = {}
        states = ['train']

        self.csv['train'] = pd.read_csv(self.csvpath_val)

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
                self.val_eval = create_dataset.Eval_train(self.csv['train'][0:20], self.cfg) 
                self.val_eval_val = create_dataset.Eval_train(self.csv['train'][0:10], self.cfg) 
            else :
                self.val_eval = create_dataset.Eval_train(self.csv['train'][6:6+int(len(self.csv['train'])*self.cfg.get('prop_train',1))], self.cfg) 
                self.val_eval_val = create_dataset.Eval_train(self.csv['train'][0:6], self.cfg) 

    def train_dataloader(self):
        return DataLoader(self.val_eval, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_eval_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)




class Brats21(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = True
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats21.IDs.val
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['train']

        self.csv['train'] = pd.read_csv(self.csvpath_val)

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
                self.val_eval = create_dataset.Eval_train(self.csv['train'][0:2], self.cfg) 
                self.val_eval_val = create_dataset.Eval_train(self.csv['train'][0:2], self.cfg) 
            else :
                self.val_eval = create_dataset.Eval_train(self.csv['train'][10:10+int(len(self.csv['train'])*self.cfg.get('prop_train',1))], self.cfg) 
                self.val_eval_val = create_dataset.Eval_train(self.csv['train'][0:10], self.cfg) 

    def train_dataloader(self):
        return DataLoader(self.val_eval, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_eval_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
