# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
 
import json
import os
import random
import numpy as np
from distutils.dir_util import copy_tree
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
import time
from myDataset import myDataset
from surrogate.Surrogate_Model import DNN,DNN2,DNN3, DNN4, DNN5, DNN_utilization
import shutil
import logging as log

class TrainSurrogate():
    INPUT_SIZE = 21
    OUTPUT_SIZE = 3
    def __init__(self, epoch : int, 
                 batch_size : int,
                 lr : float,
                 surrogate_type : str,
                 output_path : str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DNN_utilization(inputsize=self.INPUT_SIZE,outputsize=self.OUTPUT_SIZE).to(self.device) # surrogated for simulator
        
        self.train_ds = torch.load(os.path.join(os.getcwd(), 'surrogate', 'dataset', surrogate_type,'train_ds.pt'))
        self.test_ds = torch.load(os.path.join(os.getcwd(), 'surrogate', 'dataset', surrogate_type, 'test_ds.pt'))
        
        self.current_epoch = 0
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.train_loader = DataLoader(dataset=self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_ds, batch_size=self.batch_size)
        self.output_path = output_path

        if os.path.isdir(os.path.join('surrogate', 'model', self.output_path)):
            shutil.rmtree(os.path.join('surrogate', 'model', self.output_path))
        os.makedirs(os.path.join('surrogate', 'model', self.output_path), exist_ok=True)
        
    
    def train(self):
        num_batches = len(self.train_loader)
        
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (x,y) in enumerate(tqdm(self.train_loader)):
            x, y = x.to(torch.float32).to(self.device), y.to(torch.float32).to(self.device)
            
            output = self.model(x)
            loss = self.loss_func(output, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss/num_batches
    
    def test(self, is_test=False):
        num_batches = len(self.test_loader)
        
        self.model.eval()
        epoch_loss = 0
        epoch_delivered_rate_loss = 0
        epoch_utilization_rate_loss = 0
        epoch_num_risk_loss = 0
        
        with torch.no_grad():
            for batch_idx, (x,y) in enumerate(tqdm(self.test_loader)):
                x, y = x.to(torch.float32).to(self.device), y.to(torch.float32).to(self.device)
                output = self.model(x)
                
                loss = self.loss_func(output, y)
                # loss = customize_MSE(output, y)
                
                epoch_loss += loss.item()

                epoch_delivered_rate_loss += self.MAE(output[:,0],y[:,0])
                epoch_utilization_rate_loss += self.MAE(output[:,1],y[:,1])
                epoch_num_risk_loss += self.MAE(output[:,2], y[:,2])

        
        if is_test:      
            return epoch_loss/num_batches, epoch_delivered_rate_loss/num_batches, epoch_utilization_rate_loss/num_batches, epoch_num_risk_loss/num_batches
        else:
            return epoch_loss/num_batches
        
    def MAE(self, x,y):
        output = torch.abs((x - y))
        output = torch.mean(output).item()
        
        return output
    
    def run_train(self):
        
        log.info(f"Now train data size : {len(self.train_ds)}")
        log.info("Starting simulation in 5 seconds...")

        time.sleep(5)
        
        now = time.time() 
        
        best_model_name = ''
        min_loss = 100000
        train_losses = np.zeros((self.epoch))
        test_losses = np.zeros((self.epoch))
        delivered_rate_losses = np.zeros((self.epoch))
        utilization_rate_losses = np.zeros((self.epoch))
        num_risk_losses = np.zeros((self.epoch))
        
        for ep_idx, epoch in enumerate(tqdm(range(self.current_epoch, self.current_epoch + self.epoch))):
            train_loss = self.train()
            test_loss, delivered_rate_loss, utilization_rate_loss, num_risk_loss  = self.test(is_test=True)
            if test_loss < min_loss:
                min_loss = test_loss
                torch.save(self.model.state_dict(), os.path.join('surrogate', "model", self.output_path, f'ep_{epoch}_loss_{min_loss:.6f}.pth'))
                best_model_name = os.path.join('surrogate', "model", self.output_path, f'ep_{epoch}_loss_{min_loss:.6f}.pth')
                
            train_losses[ep_idx] = train_loss
            test_losses[ep_idx] = test_loss
            delivered_rate_losses[ep_idx] = delivered_rate_loss
            utilization_rate_losses[ep_idx] = utilization_rate_loss
            num_risk_losses[ep_idx] = num_risk_loss
        
        self.current_epoch = self.current_epoch + self.epoch
        train_time = time.time() - now
        
        if os.path.isfile(os.path.join('surrogate', "model", self.output_path, f"train_result_log.json")):
            with open(os.path.join('surrogate', "model", self.output_path, f"train_result_log.json"), "r") as f:
                result_log = json.load(f)
                result_log['epoch'] = result_log['epoch'] + self.epoch
                result_log['train_time'] = result_log['train_time'] + train_time
                result_log['train_loss'] = result_log['train_loss'] + train_losses.tolist()
                result_log['test_loss'] = result_log['test_loss'] + test_losses.tolist()
                result_log['delivered_rate_loss'] = result_log['delivered_rate_loss'] + delivered_rate_losses.tolist()
                result_log['utilization_rate_loss'] = result_log['utilization_rate_loss'] + utilization_rate_losses.tolist()
                result_log['num_risk_loss'] = result_log['num_risk_loss'] + num_risk_losses.tolist()
                
            with open(os.path.join('surrogate', "model", self.output_path, f"train_result_log.json"), "w") as f:
                json.dump(result_log, f, indent=2)
        else:
            # save result log
            result_log = {'epoch' : self.epoch,
                        'batch_size' : self.batch_size,
                        'train_ds_size' : len(self.train_ds),
                        'test_ds_size' : len(self.test_ds),
                        'learning_rate': self.lr,
                        'train_time' : train_time,
                        'best_model' : best_model_name,
                        'train_loss' : train_losses.tolist(),
                        'test_loss': test_losses.tolist(),
                        'delivered_rate_loss': delivered_rate_losses.tolist(),
                        'utilization_rate_loss': utilization_rate_losses.tolist(),
                        'num_risk_loss' : num_risk_losses.tolist()}
            with open(os.path.join('surrogate', "model", self.output_path, f"train_result_log.json"), "w") as f:
                json.dump(result_log, f, indent=2)
        
        best_model = DNN_utilization(self.INPUT_SIZE, self.OUTPUT_SIZE)
        best_model.load_state_dict(torch.load(best_model_name, map_location=torch.device('cpu')))
        return best_model

class TrainIncrementalData():
    INPUT_SIZE = 21
    OUTPUT_SIZE = 3
    def __init__(self, epoch : int, 
                 ID : int,
                 batch_size : int,
                 lr : float,
                 surrogate_type : str,
                 output_path : str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DNN_utilization(inputsize=self.INPUT_SIZE,outputsize=self.OUTPUT_SIZE).to(self.device) # surrogated for simulator
        
        self.train_ds = torch.load(os.path.join(os.getcwd(), 'surrogate', 'dataset', surrogate_type,'train_ds.pt'))
        self.test_ds = torch.load(os.path.join(os.getcwd(), 'surrogate', 'dataset', surrogate_type, 'test_ds.pt'))
        
        self.type = 'surrogate12'
        self.ID = ID
        self.current_epoch = 0
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.train_loader = DataLoader(dataset=self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_ds, batch_size=self.batch_size)
        self.output_path = output_path

        if os.path.isdir(os.path.join('surrogate', 'model', self.output_path, self.type,  str(self.ID))):
            shutil.rmtree(os.path.join('surrogate', 'model', self.output_path, self.type, str(self.ID)))
        os.makedirs(os.path.join('surrogate', 'model', self.output_path, self.type, str(self.ID)), exist_ok=True)
        
    
    def train(self):
        num_batches = len(self.train_loader)
        
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (x,y) in enumerate(tqdm(self.train_loader)):
            x, y = x.to(torch.float32).to(self.device), y.to(torch.float32).to(self.device)
            
            output = self.model(x)
            loss = self.loss_func(output, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss/num_batches
    
    def test(self, is_test=False):
        num_batches = len(self.test_loader)
        
        self.model.eval()
        epoch_loss = 0
        epoch_delivered_rate_loss = 0
        epoch_utilization_rate_loss = 0
        epoch_num_risk_loss = 0
        
        with torch.no_grad():
            for batch_idx, (x,y) in enumerate(tqdm(self.test_loader)):
                x, y = x.to(torch.float32).to(self.device), y.to(torch.float32).to(self.device)
                output = self.model(x)
                
                loss = self.loss_func(output, y)
                # loss = customize_MSE(output, y)
                
                epoch_loss += loss.item()

                epoch_delivered_rate_loss += self.MAE(output[:,0],y[:,0])
                epoch_utilization_rate_loss += self.MAE(output[:,1],y[:,1])
                epoch_num_risk_loss += self.MAE(output[:,2], y[:,2])

        
        if is_test:      
            return epoch_loss/num_batches, epoch_delivered_rate_loss/num_batches, epoch_utilization_rate_loss/num_batches, epoch_num_risk_loss/num_batches
        else:
            return epoch_loss/num_batches
        
    def MAE(self, x,y):
        output = torch.abs((x - y))
        output = torch.mean(output).item()
        
        return output
    
    def run_train(self):
        os.makedirs(os.path.join('surrogate', 'model', self.output_path, self.type, str(self.ID)), exist_ok=True)
        
        log.info(f"Now train data size : {len(self.train_ds)}")
        log.info("Starting simulation in 5 seconds...")

        time.sleep(5)
        
        now = time.time() 
        
        best_model_name = ''
        min_loss = 100000
        train_losses = np.zeros((self.epoch))
        test_losses = np.zeros((self.epoch))
        delivered_rate_losses = np.zeros((self.epoch))
        utilization_rate_losses = np.zeros((self.epoch))
        num_risk_losses = np.zeros((self.epoch))
        
        for ep_idx, epoch in enumerate(tqdm(range(self.epoch))):
            train_loss = self.train()
            test_loss, delivered_rate_loss, utilization_rate_loss, num_risk_loss  = self.test(is_test=True)
            self.loss = test_loss
            if test_loss < min_loss:
                min_loss = test_loss
                torch.save(self.model.state_dict(), os.path.join('surrogate', "model", self.output_path, self.type, str(self.ID),  f'ep_{epoch}_loss_{min_loss:.6f}.pth'))
                best_model_name = os.path.join('surrogate', "model", self.output_path, self.type, str(self.ID), f'ep_{epoch}_loss_{min_loss:.6f}.pth')
                
            train_losses[ep_idx] = train_loss
            test_losses[ep_idx] = test_loss
            delivered_rate_losses[ep_idx] = delivered_rate_loss
            utilization_rate_losses[ep_idx] = utilization_rate_loss
            num_risk_losses[ep_idx] = num_risk_loss
        
        train_time = time.time() - now
        
        if os.path.isfile(os.path.join('surrogate', "model", self.output_path, self.type, str(self.ID),  f"train_result_log.json")):
            with open(os.path.join('surrogate', "model", self.output_path, self.type, str(self.ID), f"train_result_log.json"), "r") as f:
                result_log = json.load(f)
                result_log['epoch'] = result_log['epoch'] + self.epoch
                result_log['train_time'] = result_log['train_time'] + train_time
                result_log['train_loss'] = result_log['train_loss'] + train_losses.tolist()
                result_log['test_loss'] = result_log['test_loss'] + test_losses.tolist()
                result_log['delivered_rate_loss'] = result_log['delivered_rate_loss'] + delivered_rate_losses.tolist()
                result_log['utilization_rate_loss'] = result_log['utilization_rate_loss'] + utilization_rate_losses.tolist()
                result_log['num_risk_loss'] = result_log['num_risk_loss'] + num_risk_losses.tolist()
                
            with open(os.path.join('surrogate', "model", self.output_path, self.type, str(self.ID), f"train_result_log.json"), "w") as f:
                json.dump(result_log, f, indent=2)
        else:
            # save result log
            result_log = {'epoch' : self.epoch,
                        'batch_size' : self.batch_size,
                        'train_ds_size' : len(self.train_ds),
                        'test_ds_size' : len(self.test_ds),
                        'learning_rate': self.lr,
                        'train_time' : train_time,
                        'best_model' : best_model_name,
                        'train_loss' : train_losses.tolist(),
                        'test_loss': test_losses.tolist(),
                        'delivered_rate_loss': delivered_rate_losses.tolist(),
                        'utilization_rate_loss': utilization_rate_losses.tolist(),
                        'num_risk_loss' : num_risk_losses.tolist()}
            with open(os.path.join('surrogate', "model", self.output_path, self.type, str(self.ID), f"train_result_log.json"), "w") as f:
                json.dump(result_log, f, indent=2)
        
        best_model = DNN_utilization(self.INPUT_SIZE, self.OUTPUT_SIZE)
        best_model.load_state_dict(torch.load(best_model_name, map_location=torch.device('cpu')))
        return best_model