"""
Name: MCDL
Date: 2024/07/18
Version: 1.0
"""

import torch.nn.modules as nn
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
import math
import copy
from torch.nn import TransformerEncoderLayer, Transformer, MultiheadAttention
from transformers import ViTModel
import torch
from torch import nn
import numpy as np
import layers
from Orthographic_pytorch import *


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class TextModel(nn.Module):
    def __init__(self, opt, pretrained=True):
        super(TextModel, self).__init__()
        abl_path = './weights/'

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = False

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output


class FuseModel(nn.Module):
    def __init__(self, opt, pretrained=True):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type
        self.text_model = TextModel(opt, pretrained=pretrained)
        self.vit = ViTModel.from_pretrained(
            'facebook/deit-base-patch16-224' if pretrained else 'facebook/deit-tiny-patch16-224')

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(256 if pretrained else 192, opt.tran_dim),
            ActivateFun(opt)
        )

        self.output_attention = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 1)
        )
        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)
        )
        self.transformer = Transformer(nhead=1, num_encoder_layers=1, num_decoder_layers=1, d_model=768,
                                       dim_feedforward=128)  # 4,1,1,768,128;

    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)

        image_feature = self.vit(image_inputs)
        image_init = image_feature.last_hidden_state
        image_init = self.image_change(image_init)

        # transformer
        image_init = image_init.permute(1, 0, 2).contiguous()
        text_init = text_init.permute(1, 0, 2).contiguous()
        text_image_transformer = self.transformer(image_init, text_init)
        text_image_transformer = text_image_transformer.permute(1, 2, 0).contiguous()

        # text_image_transformer = torch.cat((text_init, image_init), dim=1)
        # text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()

        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()
            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:text_image_output.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_alpha = self.output_attention(text_image_output)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
            text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(2)
            text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
        else:
            raise Exception('fuse_type设定错误')

        return text_image_output


class TeacherModel(FuseModel):
    def __init__(self, opt):
        super(TeacherModel, self).__init__(opt, pretrained=True)


class StudentModel(FuseModel):
    def __init__(self, opt, fuse_type='ave'):
        super(StudentModel, self).__init__(opt, pretrained=False)


class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.student_model = StudentModel(opt)
        self.teacher_model = TeacherModel(opt)

        # freeze the teacher
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)
        )

    def forward(self, data_orgin, data_augment=None, labels=None, target_labels=None, text=None):
        student_output = self.student_model(data_orgin.texts, data_orgin.bert_attention_mask, data_orgin.images,
                                            data_orgin.text_image_mask)
        # with torch.no_grad():
        teacher_output = self.teacher_model(data_orgin.texts, data_orgin.bert_attention_mask, data_orgin.images,
                                            data_orgin.text_image_mask)

        output = self.output_classify(student_output)
        return output, student_output, teacher_output


class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = FuseModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
