"""
Name: dev_process
Date: 2022/4/11 上午10:26
Version: 1.0
"""
import math
from model import ModelParam
import torch
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from util.compare_to_save import compare_to_save
import test_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from SCAttention import *


# 蒸馏损失
def distillation_loss(student_output, teacher_output, temperature=3.0):
    student_log_softmax = nn.functional.log_softmax(student_output / temperature, dim=1)
    teacher_softmax = nn.functional.softmax(teacher_output / temperature, dim=1)
    loss = nn.functional.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (temperature ** 2)
    return loss


def dev_process(opt, critertion, cl_model, dev_loader, test_loader=None, last_F1=None, last_Accuracy=None, train_log=None, log_summary_writer:SummaryWriter=None):
    y_true = []
    y_pre = []
    total_labels = 0
    dev_loss = 0

    orgin_param = ModelParam()
    augment_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        dev_loader_tqdm = tqdm(dev_loader, desc='Dev Iteration')
        epoch_step_num = train_log['epoch'] * dev_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(dev_loader_tqdm):
            texts_origins, bert_attention_mask, image_origin, text_image_mask, labels, \
            texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels, images_path = data

            texts_origin, text = texts_origins

            if opt.cuda:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                texts_augment = texts_augment.cuda()
                bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                image_augment = image_augment.cuda()
                text_image_mask_augment = text_image_mask_augment.cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask)
            augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=image_augment, text_image_mask=text_image_mask_augment)

            # origin_res, image_init, text_init, text_length, loss_ita = cl_model(orgin_param, augment_param, labels, None, text)
            origin_res, student_output, teacher_output = cl_model(orgin_param, augment_param, labels, target_labels, text)

            classify_loss = critertion(origin_res, labels)
            distill_loss = distillation_loss(student_output, teacher_output)
            loss = classify_loss + distill_loss / opt.batch_size
            dev_loss += loss.item()

            _, predicted = torch.max(origin_res, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())

            dev_loader_tqdm.set_description("Dev Iteration, loss: %.6f" % loss)
            if log_summary_writer:
                log_summary_writer.add_scalar('dev_info/loss', loss.item(), global_step=step_num + epoch_step_num)
            step_num += 1

        # dev_loss /= total_labels
        dev_loss = dev_loss
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)

        # 评价指标
        # dev_accuracy = accuracy_score(y_true, y_pre)
        # dev_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        # dev_R_weighted = recall_score(y_true, y_pre, average='weighted')
        # dev_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        # dev_F1 = f1_score(y_true, y_pre, average='macro')
        # dev_R = recall_score(y_true, y_pre, average='macro')
        # dev_precision = precision_score(y_true, y_pre, average='macro')

        # 按照原论文
        dev_accuracy = accuracy_score(y_true, y_pre)
        dev_F1_T = f1_score(y_true, y_pre, average='binary', pos_label=0)
        dev_P_T = precision_score(y_true, y_pre, average='binary', pos_label=0)
        dev_R_T = recall_score(y_true, y_pre, average='binary', pos_label=0)

        dev_F1_F = f1_score(y_true, y_pre, average='binary', pos_label=1)
        dev_P_F = precision_score(y_true, y_pre, average='binary', pos_label=1)
        dev_R_F = recall_score(y_true, y_pre, average='binary', pos_label=1)

        save_content = 'Dev  : Accuracy: %.6f, dev_F1_T: %.6f, dev_P_T: %.6f, dev_R_T: %.6f, dev_F1_F: %.6f, dev_P_F: %.6f, dev_R_F: %.6f, loss: %.6f' % \
                       (dev_accuracy, dev_F1_T, dev_P_T, dev_R_T, dev_F1_F, dev_P_F, dev_R_F, dev_loss)

        print(save_content)

        if log_summary_writer:
            log_summary_writer.add_scalar('dev_info/loss_epoch', dev_loss, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/acc', dev_accuracy, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/dev_F1_T', dev_F1_T, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/dev_P_T', dev_P_T, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/dev_R_T', dev_R_T, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/dev_F1_F', dev_F1_F, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/dev_P_F', dev_P_F, global_step=train_log['epoch'])
            log_summary_writer.add_scalar('dev_info/dev_R_F', dev_R_F, global_step=train_log['epoch'])
            log_summary_writer.flush()

        if last_F1 is not None:
            WriteFile(
                opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
            # 运行测试集
            test_process.test_process(opt, critertion, cl_model, test_loader, last_F1, log_summary_writer, train_log['epoch'])

            dev_log = {
                "dev_accuracy": dev_accuracy,
                "dev_F1_T": dev_F1_T,
                "dev_P_T": dev_P_T,
                "dev_R_T": dev_R_T,
                "dev_F1_F": dev_F1_F,
                "dev_P_F": dev_P_F,
                "dev_R_F": dev_R_F,
                "dev_loss": dev_loss
            }

            last_Accuracy, is_save_model, model_name = compare_to_save(last_Accuracy, dev_accuracy, opt, cl_model, train_log, dev_log, 'Acc', opt.save_acc, add_enter=False)
            if is_save_model is True:
                if opt.data_type == 'HFM':
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1_T, opt, cl_model, train_log, dev_log, 'F1-marco', opt.save_F1, 'F1-marco', model_name)
                else:
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_P_F, opt, cl_model, train_log, dev_log, 'F1', opt.save_F1, 'F1', model_name)
            else:
                if opt.data_type == 'HFM':
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_F1_T, opt, cl_model, train_log, dev_log, 'F1-marco', opt.save_F1)
                else:
                    last_F1, is_save_model, model_name = compare_to_save(last_F1, dev_P_F, opt, cl_model, train_log, dev_log, 'F1', opt.save_F1)

            return last_F1, last_Accuracy