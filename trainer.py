import json
import os
from logging import Logger

import torch
import tqdm
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from Datasets import getTrainDataset, getTestDataset
from Model.TRCaptionNet import TRCaptionNet
from eval import evaluate_on_coco_caption, predict
from utils import TBLog
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, args, tb_logger: TBLog = None, logger: Logger = None):

        # initialize parameters
        self.args = args
        self.experiment_root = args.save_path
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{args.gpu}")
        self.lr = float(args.lr)
        self.lr_proj = float(args.lr_proj)
        self.betas = args.betas
        self.grad_acc = args.grad_acc
        self.weight_decay = args.weight_decay
        self.max_iter = args.max_iter
        self.warm_up_iter = args.warm_up_iter
        self.target_metric = args.target_metric
        self.it = 0
        self.best_eval_val = -1
        self.best_it = -1
        self.pretrain_ckpt_path = args.pretrain if hasattr(args, "pretrain") else None

        # dataset parameters
        self.train_dataset_name = args.train_dataset_name
        self.test_dataset_name = args.test_dataset_name
        self.train_dataset_root = args.train_dataset_root
        self.test_dataset_root = args.test_dataset_root
        self.train_json_path = args.train_json_path
        self.val_json_path = args.val_json_path

        # set tensorboard logger
        self.tb_logger = tb_logger

        # set logger function
        self.logger_fn = logger.info if logger is not None else print
        self.logger_fn(f"USE: {self.device} for training")
        return

    def __call__(self):

        # set dataloaders
        self.train_loader, self.test_loader = self.getDataloaders()

        # initialize model
        self.model = TRCaptionNet(self.args.model)
        self.model = self.model.to(self.device)

        # initialize optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.language_decoder.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay, "lr": self.lr},
            {'params': [p for n, p in self.model.language_decoder.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.lr},
            {'params': [p for n, p in self.model.proj.named_parameters() if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay, "lr": self.lr_proj},
            {'params': [p for n, p in self.model.proj.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': self.lr_proj},

        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=self.betas)

        # initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.warm_up_iter, self.max_iter)

        # load pretrain
        if self.pretrain_ckpt_path is not None:
            self.load_pretrain_model(self.pretrain_ckpt_path)

        self.logger_fn("Train is starting...")
        self.train()
        return

    def train(self):
        # train
        self.model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        tbar = tqdm.tqdm(total=len(self.train_loader), colour='BLUE')
        for image, caption, ids in self.train_loader:
            tbar.update(1)
            self.it += 1

            end_batch.record()
            start_run.record()

            image = image.to(self.device)
            loss = self.model(image, caption)

            loss.backward()
            self.scheduler.step()

            if self.it % self.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {'train/loss': loss.detach().cpu().item(),
                       'lr': self.optimizer.param_groups[0]['lr'],
                       'train/prefecth_time': start_batch.elapsed_time(end_batch) / 1000.,
                       'train/run_time': start_run.elapsed_time(end_run) / 1000.}

            # if eval iter
            if self.it % self.args.num_eval_iter == 0:
                eval_dict = self.eval(self.it)
                tb_dict.update(eval_dict)

                # if best model
                if eval_dict[self.target_metric] > self.best_eval_val:
                    self.best_eval_val = eval_dict[self.target_metric]
                    self.best_it = self.it
                    self.save_model('model_best.pth')

                # log
                self.logger_fn(f"\n {self.it} iteration, {eval_dict},"
                               f" \n BEST {self.target_metric}: {self.best_eval_val}, at {self.best_it} iters")
                self.logger_fn(f" {self.it} iteration, {self.target_metric}:"
                               f" {eval_dict[self.target_metric]}\n")

            # update tensorboard
            if self.tb_logger is not None:
                self.tb_logger.update(tb_dict, self.it)
            del tb_dict
            start_batch.record()

        # save last model
        self.save_model('model_last.pth')
        return

    def getDataloaders(self):

        # load train dataset
        if type(self.train_dataset_name) is str:
            assert type(self.train_dataset_root) is str
            assert type(self.train_json_path) is str

            train_dataset = getTrainDataset(dataset_name=self.train_dataset_name,
                                            dataset_root=self.train_dataset_root,
                                            train_json_path=self.train_json_path,
                                            vision_model=self.args.model)

        elif type(self.train_dataset_name) is list:  # concat all datasets
            assert type(self.train_dataset_root) is list
            assert type(self.train_json_path) is list

            # for all train datasets
            train_datasets = []
            for i in range(len(self.train_dataset_name)):
                train_dataset = getTrainDataset(dataset_name=self.train_dataset_name[i],
                                                dataset_root=self.train_dataset_root[i],
                                                train_json_path=self.train_json_path[i],
                                                vision_model=self.args.model)
                train_datasets.append(train_dataset)

            train_dataset = ConcatDataset(train_datasets)
        else:
            raise Exception("What do u want to do!! ")

        # init train dataloader
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  sampler=RandomSampler(data_source=train_dataset,
                                                        replacement=True,
                                                        num_samples=self.args.max_iter * self.args.batch_size),
                                  pin_memory=True, drop_last=True)

        # load test dataset
        test_dataset = getTestDataset(dataset_name=self.test_dataset_name,
                                      dataset_root=self.test_dataset_root,
                                      test_json_path=self.val_json_path,
                                      vision_model=self.args.model)

        # init test dataloader
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        return train_loader, test_loader

    def eval(self, iter=-1):
        """ evaluate model """
        self.logger_fn("Start evaluating")
        self.model.eval()

        # get predictions
        val_result = predict(self.model, self.test_loader, self.device
                             , max_length=None, min_length=6, num_beams=3, repetition_penalty=1.2)

        # save predictions and evaluate
        self.save_result(val_result, f"prediction_{iter}.json")
        result = evaluate_on_coco_caption(os.path.join(self.experiment_root, f"prediction_{iter}.json"),
                                          self.val_json_path,
                                          os.path.join(self.experiment_root, f"result_{iter}.json"))
        self.logger_fn(result)
        self.model.train()
        return result

    def save_model(self, model_name: str):
        """ save model parameters, optimizer and scheduler """
        save_filename = os.path.join(self.experiment_root, model_name)
        self.model.eval()
        save_obj = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'it': self.it,
        }
        torch.save(save_obj, save_filename)
        self.model.train()
        self.logger_fn(f"model saved: {save_filename}\n")
        return

    def load_pretrain_model(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        self.logger_fn(f'model pretrain loaded from {ckpt_path}')
        return
    def load_model(self, load_dir, load_name):
        """ load saved model with scheduler and optimizer """
        load_path = str(os.path.join(load_dir, load_name))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.logger_fn(f'model loaded from {load_path}')
        return

    def save_result(self, result, filename):
        """ save prediction results """
        result_file = os.path.join(self.experiment_root, '%s' % filename)
        json.dump(result, open(result_file, 'w'))
        return
