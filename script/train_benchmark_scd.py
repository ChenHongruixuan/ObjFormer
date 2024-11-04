import sys
sys.path.append('/home/songjian/project/MSCD/github_code') # Please change the path here to yours
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.metrics import Evaluator
import time
from datetime import datetime
from utils.make_dataset import OpenMapCDeDatset_SCD

from models.SCD.Benchmark_model import HRSCD_S4


class Trainer(object):
    def __init__(self, args):
        self.args = args

        print(args.model_type + ' is running')
        self.evaluator_bcd = Evaluator(num_class=2)
        self.evaluator_mcd = Evaluator(num_class=50)
        self.evaluator_lcm = Evaluator(num_class=8)

        self.deep_model = HRSCD_S4(in_dim_clf=3, in_dim_cd=6, out_dim_clf=8, out_dim_cd=2)

        self.deep_model = self.deep_model.cuda()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            f"{args.model_type}_{timestamp}")
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.value_dict = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 9, 10, 11, 12, 13, 14],
            [0, 15, 1, 16, 17, 18, 19, 20],
            [0, 21, 22, 1, 23, 24, 25, 26],
            [0, 27, 28, 29, 1, 30, 31, 32],
            [0, 33, 34, 35, 36, 1, 37, 38],
            [0, 39, 40, 41, 42, 43, 1, 44],
            [0, 45, 46, 47, 48, 49, 50, 1]
        ])

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        self.deep_model.train()

        train_dataset = OpenMapCDeDatset_SCD(self.args.dataset_path, self.args.train_data_name_list, self.args.crop_size, self.args.max_iters * self.args.batch_size, 'train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=16, drop_last=False)
        
        elem_num = len(train_data_loader)
        
        train_enumerator = enumerate(train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            map_data, satellite_images, map_labels, _, mcd_labels, _ = data

            map_data = map_data.cuda()
            satellite_images = satellite_images.cuda()

            map_labels = map_labels.cuda().long()
            mcd_labels = mcd_labels.cuda().long()
            changed_mask = mcd_labels > 0
            unchanged_mask = mcd_labels == 0
            bcd_label = torch.where((mcd_labels > 1) & (mcd_labels < 255), torch.tensor(1).cuda(), mcd_labels).long()
            

            input_data = torch.cat([map_data, satellite_images], dim=1)
            bcd_output, clf_output_1, clf_output_2 = self.deep_model(map_data=map_data, satellite_img=satellite_images,
                                                        concat_data=input_data)
            # output_2 = self.deep_model(post_change_imgs, pre_change_imgs)
            # output_2 = self.deep_model(post_change_imgs, pre_change_imgs)

            self.optim.zero_grad()

            clf_loss_1 = F.cross_entropy(clf_output_1, map_labels, ignore_index=255)
            clf_loss_2 = F.cross_entropy(clf_output_2, map_labels, reduction='none', ignore_index=255)
            clf_loss_2 = torch.sum(clf_loss_2[unchanged_mask]) / torch.sum(unchanged_mask)

            B, C, H, W = clf_output_2.shape
            penalty_one_hot = torch.zeros(B, C, H, W).to(clf_output_2.device)
            penalty_one_hot.scatter_(1, (map_labels).unsqueeze(1), 1)
            penalty = (F.softmax(clf_output_2, dim=1) * penalty_one_hot).sum(dim=1) * changed_mask
            rce_loss = torch.sum(penalty) / torch.sum(changed_mask)


            bcd_loss = F.cross_entropy(bcd_output, bcd_label, ignore_index=255)

            main_loss = bcd_loss + clf_loss_1 + clf_loss_2 + rce_loss
            main_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(
                    f'iter is {itera + 1}, classification loss is {clf_loss_1 + clf_loss_2}, change detection loss is {bcd_loss}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    clfoa, clfmiou, clfkc, cdoa, cdkc, trkc = self.validation()
                    if trkc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))

                        best_kc = trkc
                        best_round = [clfoa, clfmiou, clfkc, cdoa, cdkc, trkc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator_bcd.reset()
        self.evaluator_mcd.reset()
        self.evaluator_lcm.reset()

        eval_dataset = OpenMapCDeDatset_SCD(self.args.dataset_path, self.args.eval_data_name_list, 512, None, 'test')
        val_data_loader = DataLoader(eval_dataset, batch_size=1, num_workers=4, drop_last=False)
        
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        for itera, data in enumerate(val_data_loader):
            map_data, satellite_images, map_labels, satellite_labels, mcd_labels, _ = data

            map_data = map_data.cuda()
            satellite_images = satellite_images.cuda()

            map_labels = map_labels.cuda().long()
            satellite_labels = satellite_labels.cuda().long()
            mcd_labels = mcd_labels.cuda().long()
            bcd_label = torch.where((mcd_labels > 1) & (mcd_labels < 255), torch.tensor(1).cuda(), mcd_labels).long()

            input_data = torch.cat([map_data, satellite_images], dim=1)
            bcd_output, _, clf_output = self.deep_model(map_data=map_data, satellite_img=satellite_images,
                                                        concat_data=input_data)
            clf_output = clf_output.data.cpu().numpy()
            clf_output = np.argmax(clf_output, axis=1)
            bcd_output = bcd_output.data.cpu().numpy()
            bcd_output = np.argmax(bcd_output, axis=1)

            map_labels = map_labels.cpu().numpy()
            
            multi_cd_result = self.value_dict[map_labels, clf_output]
            multi_cd_result[bcd_output == 0] = 1
            multi_cd_result = multi_cd_result - 1
            multi_cd_result[multi_cd_result == -1] = 0

            mcd_labels = mcd_labels.cpu().numpy()
            bcd_label = bcd_label.cpu().numpy()
            satellite_labels = satellite_labels.cpu().numpy()

            self.evaluator_lcm.add_batch(satellite_labels, clf_output)            
            self.evaluator_bcd.add_batch(bcd_label, bcd_output)
            self.evaluator_mcd.add_batch(mcd_labels, multi_cd_result)

        cdoa = self.evaluator_mcd.Pixel_Accuracy()        
        cdkc = self.evaluator_bcd.Kappa_coefficient()
        trkc = self.evaluator_mcd.Kappa_coefficient()
        clfoa = self.evaluator_lcm.Pixel_Accuracy()        
        clfmiou = self.evaluator_lcm.Mean_Intersection_over_Union()
        clfkc = self.evaluator_lcm.Kappa_coefficient()
        print(f'clfOA is {clfoa}, clfmIoU is {clfmiou}, clfKC is {clfkc}, cdOA is {cdoa}, cdKC is {cdkc}, trKC is {trkc}')
        return clfoa, clfmiou, clfkc, cdoa, cdkc, trkc


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
