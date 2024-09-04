import torch
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
import dev_process
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loss.jsd_loss import JSDLoss
from loss.nce_loss import NCELoss


JSD_critertion = JSDLoss(weight=0.5)
NCE_critertion = NCELoss(temperature=0.5)

# 蒸馏损失
def distillation_loss(student_output, teacher_output, temperature=3.0):
    student_log_softmax = F.log_softmax(student_output / temperature, dim=1)
    teacher_softmax = F.softmax(teacher_output / temperature, dim=1)
    loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (temperature ** 2)
    return loss


def train_process(opt, train_loader, dev_loader, test_loader, cl_model, critertion, log_summary_writer: SummaryWriter = None, tokenizer = None, image_id_list = None):
    optimizer = None

    pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param],
            "lr": 0,
        },
        {
            "params": [p for n, p in cl_model.named_parameters() if n not in pre_train_model_param],
            "lr": opt.fuse_lr,
        },
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2), weight_decay=1e-4)
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2), weight_decay=1e-4)
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum, weight_decay=1e-4)

    orgin_param = ModelParam()
    augment_param = ModelParam()

    last_F1 = 0
    last_Accuracy = 0
    for epoch in trange(opt.epoch, desc='Epoch:'):
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0

        cl_model.train()
        cl_model.zero_grad()

        if epoch >= opt.train_fuse_model_epoch:
            optimizer.param_groups[0]['lr'] = opt.lr
            optimizer.param_groups[1]['lr'] = opt.lr

        train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
        epoch_step_num = epoch * train_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(train_loader_tqdm):
            texts_origins, bert_attention_mask, image_origin, text_image_mask, labels, \
            texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels, images_path = data

            texts_origin, text = texts_origins

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                texts_augment = texts_augment.cuda()
                bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                image_augment = image_augment.cuda()
                text_image_mask_augment = text_image_mask_augment.cuda()
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask)
            augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=image_augment, text_image_mask=text_image_mask_augment)

            origin_res, student_output, teacher_output = cl_model(orgin_param, augment_param, labels, target_labels, text)

            classify_loss = critertion(origin_res, labels)
            distill_loss = distillation_loss(student_output, teacher_output)
            jsd_loss = JSD_critertion(student_output, teacher_output)
            nce_loss = NCE_critertion(student_output, teacher_output, labels)
            loss = classify_loss + nce_loss + distill_loss / opt.batch_size + jsd_loss / opt.batch_size

            loss.backward()
            train_loader_tqdm.set_description("Train Iteration, loss: %.6f, lr: %e" %
                                              (loss, optimizer.param_groups[0]['lr']))

            if (index + 1) % opt.acc_grad == 0:
                if log_summary_writer:
                    log_summary_writer.add_scalar('train_info/loss', loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/classify_loss', classify_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/distill_loss', distill_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/lr', optimizer.param_groups[0]['lr'], global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/fuse_lr', optimizer.param_groups[1]['lr'], global_step=step_num + epoch_step_num)
                optimizer.step()
                optimizer.zero_grad()
            step_num += 1

            _, predicted = torch.max(origin_res, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)

        # run_loss /= total_labels
        run_loss = run_loss
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)

        # 评价指标
        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        # 按照原论文
        # train_accuracy = accuracy_score(y_true, y_pre)
        # train_F1_T = f1_score(y_true, y_pre, average='binary', pos_label=0)
        # train_P_T = precision_score(y_true, y_pre, average='binary', pos_label=0)
        # train_R_T = recall_score(y_true, y_pre, average='binary', pos_label=0)
        #
        # train_F1_F = f1_score(y_true, y_pre, average='binary', pos_label=1)
        # train_P_F = precision_score(y_true, y_pre, average='binary', pos_label=1)
        # train_R_F = recall_score(y_true, y_pre, average='binary', pos_label=1)

        # save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, train_F1_T: %.6f, train_P_T: %.6f, train_R_T: %.6f, train_F1_F: %.6f, train_P_F: %.6f, train_R_F: %.6f, loss: %.6f' % \
        #                (epoch, train_accuracy, train_F1_T, train_P_T, train_R_T, train_F1_F, train_P_F, train_R_F, run_loss)
        # print(save_content, ' ' * 200)
        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, train_F1_weighted: %.6f, train_R_weighted: %.6f, train_precision_weighted: %.6f, train_F1: %.6f, train_R: %.6f, train_precision: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_R_weighted, train_precision_weighted, train_F1, train_R, train_precision,
                        run_loss)
        print(save_content, ' ' * 200)

        if log_summary_writer:
            log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
            log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('train_info/train_F1_weighted', train_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/train_R_weighted', train_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/train_precision_weighted', train_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/train_F1', train_F1, global_step=epoch)
            log_summary_writer.add_scalar('train_info/train_R', train_R, global_step=epoch)
            log_summary_writer.add_scalar('train_info/train_precision', train_precision, global_step=epoch)
            log_summary_writer.flush()

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1_weighted": train_F1_weighted,
            "train_R_weighted": train_R_weighted,
            "train_precision_weighted": train_precision_weighted,
            "train_F1": train_F1,
            "train_R": train_R,
            "train_precision": train_precision,
            "run_loss": run_loss
        }

        # if log_summary_writer:
        #     log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/train_F1_T', train_F1_T, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/train_P_T', train_P_T, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/train_R_T', train_R_T, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/train_F1_F', train_F1_F, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/train_P_F', train_P_F, global_step=epoch)
        #     log_summary_writer.add_scalar('train_info/train_R_F', train_R_F, global_step=epoch)
        #     log_summary_writer.flush()
        #
        # train_log = {
        #     "epoch": epoch,
        #     "train_accuracy": train_accuracy,
        #     "train_F1_T": train_F1_T,
        #     "train_P_T": train_P_T,
        #     "train_R_T": train_R_T,
        #     "train_F1_F": train_F1_F,
        #     "train_P_F": train_P_F,
        #     "train_R_F": train_R_F,
        #     "run_loss": run_loss
        # }

        last_F1, last_Accuracy = dev_process.dev_process(opt, critertion, cl_model, dev_loader, test_loader, last_F1, last_Accuracy, train_log, log_summary_writer)
