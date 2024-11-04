import sys
sys.path.append('/home/songjian/project/MSCD/github_code')
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from models.BCD.Benchmark_model import FCEF, FCSiamDiff, FCSiamConc
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.make_dataset import OpenMapCDeDatset_BCD
from utils.metrics import Evaluator
import time
from datetime import datetime


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = FCEF(in_dim=6) 
        self.deep_model = self.deep_model.cuda()
        print(args.model_type + ' is running')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            f"{args.model_type}_{timestamp}")
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_dataset = OpenMapCDeDatset_BCD(self.args.dataset_path, self.args.train_data_name_list, self.args.crop_size, self.args.max_iters * self.args.batch_size, 'train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=16, drop_last=False)
        
        elem_num = len(train_data_loader)
        
        train_enumerator = enumerate(train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)

            output_1 = self.deep_model(input_data)
          
            self.optim.zero_grad()
            ce_loss_1 = F.cross_entropy(output_1, labels, ignore_index=255)

            final_loss = ce_loss_1

            final_loss.backward()
            self.optim.step()
            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        eval_dataset = OpenMapCDeDatset_BCD(self.args.dataset_path, self.args.eval_data_name_list, 512, None, 'test')
        val_data_loader = DataLoader(eval_dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        for _, data in enumerate(val_data_loader):
            pre_change_imgs, post_change_imgs, labels, _ = data
            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            output_1 = self.deep_model(input_data)

            output_1 = output_1.data.cpu().numpy()
            output_1 = np.argmax(output_1, axis=1)
            labels = labels.cpu().numpy()

            self.evaluator.add_batch(labels, output_1)
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


def main():
    parser = argparse.ArgumentParser(description="Training on OpenMapCD dataset")
    parser.add_argument('--dataset', type=str, default='OpenMapCD', help='The name of the used dataset')
    parser.add_argument('--dataset_path', type=str, help='The path of used data')
    parser.add_argument('--train_data_list_path', type=str, help='The path of the file recording the image names for training')
    parser.add_argument('--eval_data_list_path', type=str, help='The path of the file recording the image names for evalution')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--eval_data_name_list', type=list)

    parser.add_argument('--max_iters', type=int)
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_param_path', type=str)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.eval_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.eval_data_name_list = data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
