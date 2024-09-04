"""
Name: test_process
Date: 2022/4/11 上午10:26
Version: 1.0
"""

from model import ModelParam
import torch
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
import math
from SCAttention import *
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


def test_process(opt, critertion, cl_model, test_loader, last_F1=None, log_summary_writer: SummaryWriter=None, epoch=None):
    y_true = []
    y_pre = []
    embeddings = []
    total_labels = 0
    test_loss = 0

    orgin_param = ModelParam()
    augment_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        test_loader_tqdm = tqdm(test_loader, desc='Test Iteration')
        epoch_step_num = epoch * test_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(test_loader_tqdm):

            texts_origins, bert_attention_mask, image_origin, text_image_mask, labels, \
            texts_augment, bert_attention_mask_augment, image_augment, text_image_mask_augment, target_labels, images_path = data

            texts_origin, text = texts_origins

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask)
            augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment,
                                         images=image_augment, text_image_mask=text_image_mask_augment)

            origin_res, student_output, teacher_output = cl_model(orgin_param, augment_param, labels, target_labels, text)

            classify_loss = critertion(origin_res, labels)
            distill_loss = distillation_loss(student_output, teacher_output)
            jsd_loss = JSD_critertion(student_output, teacher_output)
            nce_loss = NCE_critertion(student_output, teacher_output, labels)
            loss = classify_loss + nce_loss + distill_loss / opt.batch_size + jsd_loss / opt.batch_size

            test_loss += loss.item()
            _, predicted = torch.max(origin_res, 1)
            # print(predicted)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            embeddings.extend(origin_res.cpu())
            # print(y_pre)

            test_loader_tqdm.set_description("Test Iteration, loss: %.6f" % loss)
            if log_summary_writer:
                log_summary_writer.add_scalar('test_info/loss', loss.item(), global_step=step_num + epoch_step_num)
            step_num += 1

        # embeddings
        # print(embeddings)
        out_array = [emb.detach().cpu().numpy() for emb in embeddings]
        # out_array = np.array(embeddings)
        np.save('out_data_PHEME.npy', out_array)
        # test_loss /= total_labels
        test_loss = test_loss
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        np.save('PHEME_labels.npy', y_pre)
        # print(y_pre)

        # 评价指标
        test_accuracy = accuracy_score(y_true, y_pre)
        test_F1 = f1_score(y_true, y_pre, average='macro')
        test_R = recall_score(y_true, y_pre, average='macro')
        test_precision = precision_score(y_true, y_pre, average='macro')
        test_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        test_R_weighted = recall_score(y_true, y_pre, average='weighted')
        test_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        # categories_report = classification_report(y_true, y_pre, digits=4)

        save_content = 'Test : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
            (test_accuracy, test_F1_weighted, test_precision_weighted, test_R_weighted, test_F1, test_precision, test_R, test_loss)

        print(save_content)
        # print(categories_report)

        if log_summary_writer:
            log_summary_writer.add_scalar('test_info/loss_epoch', test_loss, global_step=epoch)
            log_summary_writer.add_scalar('test_info/acc', test_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('test_info/test_F1_weighted', test_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/test_precision_weighted', test_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/test_R_weighted', test_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/test_F1', test_F1, global_step=epoch)
            log_summary_writer.add_scalar('test_info/test_precision', test_precision, global_step=epoch)
            log_summary_writer.add_scalar('test_info/test_R', test_R, global_step=epoch)
            # log_summary_writer.add_scalar('classifcation_report', categories_report, global_step=epoch)
            log_summary_writer.flush()

        if last_F1 is not None:
            WriteFile(
                opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')

        # 原论文
        # test_accuracy = accuracy_score(y_true, y_pre)
        # test_F1_T = f1_score(y_true, y_pre, average='binary', pos_label=0)
        # test_P_T = precision_score(y_true, y_pre, average='binary', pos_label=0)
        # test_R_T = recall_score(y_true, y_pre, average='binary', pos_label=0)
        #
        # test_F1_F = f1_score(y_true, y_pre, average='binary', pos_label=1)
        # test_P_F = precision_score(y_true, y_pre, average='binary', pos_label=1)
        # test_R_F = recall_score(y_true, y_pre, average='binary', pos_label=1)
        #
        # save_content = 'Test  : Accuracy: %.6f, test_F1_T: %.6f, test_P_T: %.6f, test_R_T: %.6f, test_F1_F: %.6f, test_P_F: %.6f, test_R_F: %.6f, loss: %.6f' % \
        #                (test_accuracy, test_F1_T, test_P_T, test_R_T, test_F1_F, test_P_F, test_R_F, test_loss)

        # save_content = 'Test : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
        #     (test_accuracy, test_F1_weighted, test_precision_weighted, test_R_weighted, test_F1, test_precision, test_R, test_loss)

        # print(save_content)
        # # print(categories_report)
        #
        # if log_summary_writer:
        #     log_summary_writer.add_scalar('test_info/loss_epoch', test_loss, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/acc', test_accuracy, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/test_F1_T', test_F1_T, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/test_P_T', test_P_T, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/test_R_T', test_R_T, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/test_F1_F', test_F1_F, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/test_P_F', test_P_F, global_step=epoch)
        #     log_summary_writer.add_scalar('test_info/test_R_F', test_R_F, global_step=epoch)
        #     # log_summary_writer.add_scalar('classifcation_report', categories_report, global_step=epoch)
        #     log_summary_writer.flush()
        #
        # if last_F1 is not None:
        #     WriteFile(
        #         opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
