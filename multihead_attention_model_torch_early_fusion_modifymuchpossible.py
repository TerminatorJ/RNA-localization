import torch.nn as nn
import click
import torch
from collections import OrderedDict
# from torch_position_embedding import PositionEmbedding
from hier_attention_mask_torch import Attention_mask
from hier_attention_mask_torch import QKVAttention
import sys
import numpy as np
import torch
# import tensorflow as tf
import torch.nn as nn
# from keras.layers import Embedding,Input
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# from torchsummary import summary
from torchinfo import summary
import time
import math
import tensorflow as tf
from multihead_attention_model_torch_embed import *

from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import BinaryAccuracy
import gin
import inspect
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef
# from multihead_attention_model_torch import DM3Loc
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class ParentEmbedFloat(nn.Module):
    def __init__(self):
        super(ParentEmbedFloat, self).__init__()

    def forward(self, x):
        x = x.to(device="cuda")
        # print("parnet output:", x, x.shape)
        return x.float()


class Pooling(nn.Module):
    def __init__(self, type, pooling_size):
        super(Pooling, self).__init__()
        self.type = type
        self.maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        self.meanpool = nn.AvgPool1d(pooling_size, stride = pooling_size)
        if self.type == "None":
            self.layer_name = "NoPooling"
        else:
            self.layer_name = f"{self.type}_pooling_{pooling_size}"
    def forward(self, x):
        if self.type == "max":
            x = self.maxpool(x)
        elif self.type == "mean":
            x = self.meanpool(x)
        elif self.type == "None":
            pass
        return x
    

    
class Actvation(nn.Module):
    def __init__(self, name):
        super(Actvation, self).__init__()
        self.name = name
        self.layer_name = None
    def gelu(self, input_tensor):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415

        Args:
            input_tensor: float Tensor to perform activation.

        Returns:
            `input_tensor` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + torch.erf(input_tensor / math.sqrt(2.0)))
        return input_tensor * cdf

    def forward(self, x):
        if self.name == "relu":
            x = torch.nn.functional.relu(x)
            self.layer_name = "Activation_ReLU"
        elif self.name == "gelu":
            x = self.gelu(x)
            self.layer_name = "Activation_GeLU"
        elif self.name == "leaky":
            x = torch.nn.functional.leaky_relu(x)
            self.layer_name = "Activation_Leaky"

        return x


@gin.configurable
class DM3Loc_earlyfusion(nn.Module):
    def __init__(self, batchnorm, predict, drop_cnn, drop_flat, drop_input, pooling_size, nb_filters,
                 filters_length1, filters_length2, filters_length3, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, hidden, fusion, mode):
                                                                                          
        super(DM3Loc_earlyfusion, self).__init__()
        self.batchnorm = bool(batchnorm)
        self.predict = predict
        self.drop_cnn = drop_cnn
        self.drop_flat = drop_flat
        self.drop_input = drop_input
        self.pooling_size = pooling_size
        self.nb_filters = nb_filters
        self.filters_length1 = filters_length1
        self.filters_length2 = filters_length2
        self.filters_length3 = filters_length3
        self.fc_dim = fc_dim
        self.nb_classes = nb_classes
        self.dim_attention = dim_attention
        self.activation = activation
        self.activation_att = activation_att
        self.attention = attention
        self.headnum = headnum
        self.Att_regularizer_weight = Att_regularizer_weight
        self.normalizeatt = normalizeatt
        self.sharp_beta = sharp_beta
        self.attmod = attmod
        self.W1_regularizer = W1_regularizer
        self.activation = activation
        self.activation_att = activation_att
        self.attention = attention
        self.pool_type = pool_type

        self.cnn_scaler = cnn_scaler
        self.att_type = att_type
        self.input_dim = input_dim
        self.hidden = hidden
        self.fusion = fusion
        self.mode = mode
        encoding_seq = OrderedDict([
            ('UNK', [0, 0, 0, 0]),
            ('A', [1, 0, 0, 0]),
            ('C', [0, 1, 0, 0]),
            ('G', [0, 0, 1, 0]),
            ('T', [0, 0, 0, 1]),
            ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
        ])

        embedding_vec = np.array(list(encoding_seq.values()))

        #layer define
        self.dropout1 = nn.Dropout(drop_cnn)
        dropout1 = nn.Dropout(drop_cnn)
        self.dropout2 = nn.Dropout(drop_flat)
        dropout2 = nn.Dropout(drop_flat)
        self.dropout3 = nn.Dropout(drop_input)
        dropout3 = nn.Dropout(drop_input)
        self.maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        self.meanpool = nn.AvgPool1d(pooling_size, stride = pooling_size)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        #CNN


        self.CNN1_1 = nn.Conv1d(input_dim, nb_filters, kernel_size = filters_length1, padding='same', bias = False)
        CNN1_1 = nn.Conv1d(input_dim, nb_filters, kernel_size = filters_length1, padding='same', bias = False)
        self.CNN1_2 = nn.Conv1d(nb_filters, hidden, kernel_size = filters_length1, padding='same', bias = False)
        CNN1_2 = nn.Conv1d(nb_filters, hidden, kernel_size = filters_length1, padding='same', bias = False)

        self.CNN2_1 = nn.Conv1d(input_dim, nb_filters, kernel_size = filters_length2, padding='same', bias = False)
        CNN2_1 = nn.Conv1d(input_dim, nb_filters, kernel_size = filters_length2, padding='same', bias = False)
        self.CNN2_2 = nn.Conv1d(nb_filters, hidden, kernel_size = filters_length2, padding='same', bias = False)
        CNN2_2 = nn.Conv1d(nb_filters, hidden, kernel_size = filters_length2, padding='same', bias = False)

        self.CNN3_1 = nn.Conv1d(input_dim, nb_filters, kernel_size = filters_length3, padding='same', bias = False)
        CNN3_1 = nn.Conv1d(input_dim, nb_filters, kernel_size = filters_length3, padding='same', bias = False)
        self.CNN3_2 = nn.Conv1d(nb_filters, hidden, kernel_size = filters_length3, padding='same', bias = False)
        CNN3_2 = nn.Conv1d(nb_filters, hidden, kernel_size = filters_length3, padding='same', bias = False)

        if attention == True:
            if att_type == "transformer":
                neurons = int(hidden*3*cnn_scaler/3)
            elif att_type == "self_attention":
                neurons = int(headnum*hidden*3*cnn_scaler/3)
        elif attention == False:
            neurons = int(1000*hidden*3*cnn_scaler/3)
        
        self.fc1 = nn.Linear(neurons, fc_dim)
        fc1 = nn.Linear(neurons, fc_dim)
        self.fc2 = nn.Linear(fc_dim, nb_classes)
        fc2 = nn.Linear(fc_dim, nb_classes)
        #attention layers
        if att_type == "self_attention":
            self.Attention1 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)
            Attention1 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)

            self.Attention2 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)
            Attention2 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)

            self.Attention3 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)
            Attention3 = Attention_mask(hidden=hidden, att_dim=dim_attention, r=headnum, activation= activation_att,return_attention=True, 
                                        attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                        sharp_beta=sharp_beta)

        elif att_type == "transformer":
            self.Attention1 = QKVAttention(hidden=hidden, att_dim=dim_attention, headnum=headnum)
            self.Attention2 = QKVAttention(hidden=hidden, att_dim=dim_attention, headnum=headnum)
            self.Attention3 = QKVAttention(hidden=hidden, att_dim=dim_attention, headnum=headnum)
        #embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]),_weight=torch.tensor(embedding_vec))
        embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]),_weight=torch.tensor(embedding_vec))

        # self.embedding_layer.weight.requires_grad = False

        #activation
        #regulazation
        self.L1loss = nn.L1Loss(W1_regularizer)
        #flatten
        self.flatten = nn.Flatten()

        self.att1_A = None
        self.att2_A = None
        self.att3_A = None

        self.softmax = nn.Softmax()
        #batch regulatization
        
        self.batchnorm1_1 = nn.BatchNorm1d(nb_filters)
        batchnorm1_1 = nn.BatchNorm1d(nb_filters)
        self.batchnorm1_2 = nn.BatchNorm1d(int(nb_filters)//2)
        batchnorm1_2 = nn.BatchNorm1d(int(nb_filters)//2)

        self.batchnorm2_1 = nn.BatchNorm1d(nb_filters)
        batchnorm2_1 = nn.BatchNorm1d(nb_filters)
        self.batchnorm2_2 = nn.BatchNorm1d(int(nb_filters)//2)
        batchnorm2_2 = nn.BatchNorm1d(int(nb_filters)//2)

        self.batchnorm3_1 = nn.BatchNorm1d(int(nb_filters)//2)
        batchnorm3_1 = nn.BatchNorm1d(int(nb_filters)//2)
        self.batchnorm3_2 = nn.BatchNorm1d(int(nb_filters)//2)
        batchnorm3_2 = nn.BatchNorm1d(int(nb_filters)//2)

        self.batchnormfc1 = nn.BatchNorm1d(fc_dim)
        batchnormfc1 = nn.BatchNorm1d(fc_dim)

        # self.parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt")
        # parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt")


       
        ###Building variant of the model structions
        # self.parnet_encode_block = nn.Sequential(
        #                     parnet_model,
        #                     ParentEmbedFloat())
        # self.parnet_encode_block = nn.Sequential(
        #                     parnet_model_cpu,
        #                     ParentEmbedFloat())

        self.CNN1_batchnorm_block = nn.Sequential(CNN1_1, 
                            batchnorm1_1,
                            Actvation(activation),
                            CNN1_2,
                            batchnorm1_2,
                            Actvation(activation),
                            Pooling(pool_type, pooling_size),
                            nn.Dropout(drop_cnn))
        self.CNN1_block = nn.Sequential(CNN1_1, 
                            Actvation(activation),
                            CNN1_2,
                            Actvation(activation),
                            Pooling(pool_type, pooling_size),
                            nn.Dropout(drop_cnn))
        self.CNN2_batchnorm_block = nn.Sequential(CNN2_1, 
                            batchnorm2_1,
                            Actvation(activation),
                            CNN2_2,
                            batchnorm2_2,
                            Actvation(activation),
                            Pooling(pool_type, pooling_size),
                            nn.Dropout(drop_cnn))
        self.CNN2_block = nn.Sequential(CNN2_1, 
                            Actvation(activation),
                            CNN2_2,
                            Actvation(activation),
                            Pooling(pool_type, pooling_size),
                            nn.Dropout(drop_cnn))
        self.CNN3_batchnorm_block = nn.Sequential(CNN3_1, 
                            batchnorm3_1,
                            Actvation(activation),
                            CNN3_2,
                            batchnorm3_2,
                            Actvation(activation),
                            Pooling(pool_type, pooling_size),
                            nn.Dropout(drop_cnn))
        self.CNN3_block = nn.Sequential(CNN3_1, 
                            Actvation(activation),
                            CNN3_2,
                            Actvation(activation),
                            Pooling(pool_type, pooling_size),
                            nn.Dropout(drop_cnn))
        
        self.FC_block_batchnorm = nn.Sequential(fc1,
                                           batchnormfc1,
                                           Actvation(activation),
                                           dropout2,
                                           fc2,
                                           nn.Sigmoid())
        self.FC_block = nn.Sequential(fc1,
                                      Actvation("relu"),
                                      dropout2,
                                      fc2,
                                      nn.Sigmoid())
        
        self.Actvation = Actvation(activation)
             
        self.Pooling = Pooling(pool_type, pooling_size)

    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            print(f"{param_name}: {param_value}")


    def signal_preprocess(self, test, cutoff):
        test[test>cutoff] = 1
        test[test!=1] = 0
        return test
    def CNNandAtt(self, concate_out, x_mask):
        
        
        
        if self.cnn_scaler == 1:
            if self.batchnorm:
                x1 = self.CNN1_batchnorm_block(concate_out)
                cnn_mask_output1 = x1*x_mask 
            else:
                x1 = self.CNN1_block(concate_out)
                cnn_mask_output1 = x1*x_mask 
            if self.attention:
                cnn_mask_output1 = torch.cat((cnn_mask_output1,x_mask), dim = 1)
                if self.att_type == "self_attention":
                    att1,att1_A = self.Attention1(cnn_mask_output1, masks = True)
                    self.att1_A = att1_A
                    att1 = att1.transpose(1,2)
                elif self.att_type == "transformer":
                    att1,att1_A = self.Attention1(cnn_mask_output1, masks = True)
                    self.att1_A = att1_A
                    att1 = self.globalavgpool(att1)#[b, 32, 1]
                output = att1
            else:
                output = cnn_mask_output1
                
            
        elif self.cnn_scaler == 2:
            if self.batchnorm:
                x1 = self.CNN1_batchnorm_block(concate_out)
                cnn_mask_output1 = x1*x_mask 
                x2 = self.CNN2_batchnorm_block(concate_out)
                cnn_mask_output2 = x2*x_mask
            else:
                x1 = self.CNN1_block(concate_out)
                cnn_mask_output1 = x1*x_mask 
                x2 = self.CNN2_block(concate_out)
                cnn_mask_output2 = x2*x_mask
           
                
            if self.attention:
                cnn_mask_output1 = torch.cat((cnn_mask_output1,x_mask), dim = 1)
                cnn_mask_output2 = torch.cat((cnn_mask_output2,x_mask), dim = 1)
                if self.att_type == "self_attention":
                    att1,att1_A = self.Attention1(cnn_mask_output1, masks = True)
                    att2,att2_A = self.Attention2(cnn_mask_output2, masks = True)
                    self.att1_A = att1_A
                    self.att2_A = att2_A
                    att1 = att1.transpose(1,2)
                    att2 = att2.transpose(1,2)
                elif self.att_type == "transformer":
                    att1,att1_A = self.Attention1(cnn_mask_output1, masks = True)
                    att2,att2_A = self.Attention2(cnn_mask_output2, masks = True)
                    self.att1_A = att1_A
                    self.att2_A = att2_A
                    att1 = self.globalavgpool(att1)#[b, 32, 1]
                    att2 = self.globalavgpool(att2)
                output = torch.cat((att1, att2), dim = 2)
            else:
                output = torch.cat((cnn_mask_output1, cnn_mask_output2), dim = 1)
        elif self.cnn_scaler == 3:
            if self.batchnorm:
                x1 = self.CNN1_batchnorm_block(concate_out)
                cnn_mask_output1 = x1*x_mask 
                x2 = self.CNN2_batchnorm_block(concate_out)
                cnn_mask_output2 = x2*x_mask
                x3 = self.CNN3_batchnorm_block(concate_out)
                cnn_mask_output3 = x3*x_mask
            else:
                #test
                # t1 = self.CNN1_1(concate_out)
                # print("CNN1_1 before act:", t1)
                # t2=self.Actvation(t1)
                # print("CNN1_1 after act:", t2)
                # t3 = self.CNN1_2(t2)
                # print("CNN1_2 before act:", t3)
                # t4 = self.Actvation(t3)
                # print("CNN1_2 after act:", t4)
                # t5 = self.Pooling(t4)
                # print("CNN1_2 after pooling:", t5)
                # t6 = self.dropout1(t5)
                # print("CNN1_2 after dropout:", t6)
                
            

                x1 = self.CNN1_block(concate_out)
                # print("CNN1 with two cnn", x1, x1.shape)
                cnn_mask_output1 = x1*x_mask 
                # print("cnn1 output lambda:", cnn_mask_output1)
                x2 = self.CNN2_block(concate_out)
                cnn_mask_output2 = x2*x_mask
                x3 = self.CNN3_block(concate_out)
                cnn_mask_output3 = x3*x_mask

                
            if self.attention:
                cnn_mask_output1 = torch.cat((cnn_mask_output1,x_mask), dim = 1)
                cnn_mask_output2 = torch.cat((cnn_mask_output2,x_mask), dim = 1)
                cnn_mask_output3 = torch.cat((cnn_mask_output3,x_mask), dim = 1)
                if self.att_type == "self_attention":
                    att1,att1_A = self.Attention1(cnn_mask_output1, masks = True)
                    att2,att2_A = self.Attention2(cnn_mask_output2, masks = True)
                    att3,att3_A = self.Attention3(cnn_mask_output3, masks = True)
                    # print("Att 1", att1, att1.shape)
                    self.att1_A = att1_A
                    self.att2_A = att2_A
                    self.att3_A = att3_A
                    att1 = att1.transpose(1,2)
                    att2 = att2.transpose(1,2)
                    att3 = att3.transpose(1,2)
                elif self.att_type == "transformer":
                    att1,att1_A = self.Attention1(cnn_mask_output1, masks = True)
                    att2,att2_A = self.Attention2(cnn_mask_output2, masks = True)
                    att3,att3_A = self.Attention3(cnn_mask_output3, masks = True)
                    self.att1_A = att1_A
                    self.att2_A = att2_A
                    self.att3_A = att3_A
                    att1 = self.globalavgpool(att1)#[b, 32, 1]
                    att2 = self.globalavgpool(att2)
                    att3 = self.globalavgpool(att3)
                output = torch.cat((att1, att2, att3), dim = 2)
                # print("concate three att:", output)
            else:
                output = torch.cat((cnn_mask_output1, cnn_mask_output2, cnn_mask_output3), dim = 1)
                
        concat_output = self.flatten(output) 
        # print("flatten:", concat_output)   
        output = self.dropout1(concat_output)
        return output.to("cuda")
    def run_parnet(self, embedding_output):
        embedding_output = embedding_output.to("cpu")
        parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt")
        parnet_out = parnet_model(embedding_output)
        parnet_out = parnet_out.float()
        return parnet_out.to("cuda")
    
    def forward(self, x, x_mask, parnet_embed=None):
        #[batch,223,8000]
        # if self.fusion:
        #     self.input_dim = 4
        x = x.long()
        x_mask = x_mask.unsqueeze(1).float()
        embedding_output = self.embedding_layer(x)
        embedding_output = embedding_output.transpose(1,2)
        if self.input_dim == 227 or self.input_dim == 5:

            concate_out =  torch.cat((embedding_output, parnet_embed), axis=1)
            concate_out = self.dropout3(concate_out)
        elif self.input_dim == 4:
            
            embedding_output = self.dropout3(embedding_output)
            concate_out =  embedding_output
            # print("concate_out", concate_out, concate_out.shape)

        concate_out = concate_out.to(torch.float32)
        output = self.CNNandAtt(concate_out, x_mask)#[5,96]
        if self.mode == "feature":
            return output
        elif self.mode == "full":
            if self.batchnorm:
                pred = self.FC_block_batchnorm(output)
            else:           
                pred = self.FC_block(output)    
            return pred

    def mask_func(x):
        return x[0] * x[1]


@gin.configurable
class myModel2(pl.LightningModule):
    def __init__(self, batchnorm, predict, drop_cnn, drop_flat, drop_input, pooling_size, nb_filters,
                 filters_length1, filters_length2, filters_length3, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, hidden, 
                 lr, gradient_clip, class_weights, optimizer, pooling, device, fusion, mode):
        super(myModel2, self).__init__()
        self.network = DM3Loc_earlyfusion(batchnorm, predict, drop_cnn, drop_flat, drop_input, pooling_size, nb_filters,
                 filters_length1, filters_length2, filters_length3, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, hidden, fusion, mode)
        network = DM3Loc_earlyfusion(batchnorm, predict, drop_cnn, drop_flat, drop_input, pooling_size, nb_filters,
                 filters_length1, filters_length2, filters_length3, fc_dim, nb_classes, dim_attention,
                 headnum, Att_regularizer_weight, normalizeatt, sharp_beta, attmod, W1_regularizer, 
                 activation, activation_att, attention, pool_type, cnn_scaler, att_type, input_dim, hidden, fusion, mode)
        network.to(device = device)
        self.network.to(device = device)
        # if pooling:
        #     summary(network, input_size = [(2,8000),(2,1000),(2,1,8000)], device = device)
        # else:
        #     summary(network, input_size = [(2,8000),(2,8000), (2,1,8000)], device = device)
        # for name, param in network.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name} is trainable")
        #     else:
        #         print(f"{name} is not trainable")

        self.lr = lr
        self.cnn_scaler = cnn_scaler
        self.gradient_clip = gradient_clip
        self.class_weights = torch.tensor(class_weights)
        class_weights = class_weights
        # weights = torch.tensor([0.038, 0.038, 0.269, 0.038, 0.115, 0.192, 0.307])
        weights = torch.tensor([1,1,7,1,3,5,8])
        self.loss_fn = nn.BCELoss(weight = None)
        # self.loss_fn = nn.BCELoss(weight = weights)
        self.optimizer_cls = eval(optimizer)
        self.train_loss = []
        self.val_binary_acc = []
        self.val_Multilabel_acc = []
        self.attention = attention
        self.att_type = att_type 
        
    def weighted_binary_cross_entropy(self, output, target):

        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        return torch.neg(torch.mean(loss))

    def binary_accuracy(self, y_pred, y_true):
        # Round the predicted values to 0 or 1
        y_pred_rounded = torch.round(y_pred)
        # Calculate the number of correct predictions
        correct = (y_pred_rounded == y_true).float().sum()
        # Calculate the accuracy
        accuracy = correct / y_true.numel()
        return accuracy
    
    def categorical_accuracy(self, y_pred, y_true):
        # Get the index of the maximum value (predicted class) along the second dimension
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        # Compare the predicted class with the target class and calculate the mean accuracy
        return (y_pred == y_true).float().mean()

    def forward(self, x, mask, parnet_embed=None):
        pred = self.network(x, mask, parnet_embed=parnet_embed)
        return pred
    
    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr = self.lr, weight_decay = 5e-5)
        # optimizer = self.optimizer_cls(self.parameters(), lr = self.lr, weight_decay = 0.001)
        return optimizer

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch duration: {epoch_time:.2f} seconds")
    def _attention_regularizer(self, attention):
        batch_size = attention.shape[0]
        headnum = self.network.headnum
        identity = torch.eye(headnum).to(attention.device)  # [r,r]
        temp = torch.bmm(attention, attention.transpose(1, 2)) - identity  # [none, r, r]
        penal = 0.001 * torch.sum(temp**2) / batch_size
        return penal
    def training_step(self, batch, batch_idx, **kwargs):
        x, mask, y, parnet_embed = batch
        # x, y = batch
        y_pred = self.forward(x, mask, parnet_embed)
        # y_pred = self.forward(x)
        # print("y_pred", y_pred)
        # print("y", y)
        loss = self.loss_fn(y_pred, y)

        #Using the gradient clip to protect from gradient exploration
        if self.gradient_clip:
            nn.utils.clip_grad_value_(self.network.parameters(), 0.1)

        #for training the dm3loc
        l1_regularization = torch.tensor(0., device="cuda")
        for name, param in self.network.named_parameters(): 
            if 'Attention1.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W2' in name:
                l1_regularization += torch.norm(param, p=1)

            elif 'Attention1.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            
            elif 'Attention2.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            
            elif 'Attention3.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            
            
            #adding l1 regularization for weight parnet
            # elif 'parnet_weight' in name:
            #     l1_regularization += torch.norm(param, p=1)
        if self.attention and self.att_type == "self_attention":
            # l1_regularization += torch.norm(self.network.att1_A, p='fro')
            # l1_regularization += torch.norm(self.network.att2_A, p='fro')
            # l1_regularization += torch.norm(self.network.att3_A, p='fro')
            loss += l1_regularization*0.001
            #add the Attention regulizer
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att3_A, 1, 2))
        if self.attention and self.att_type == "transformer":
            loss += l1_regularization*0.001
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(self.network.att1_A)
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)    
                loss += self._attention_regularizer(self.network.att3_A)
            
  
        self.log("train_loss", loss, on_epoch = True, on_step = True)
        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)
        
        self.log('train categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)
 

        return loss
    def categorical_accuracy_strict(self, y_pred, y_true):
    # Find the index of the maximum value in each row (i.e., the predicted class)
        y_pred_class = torch.round(y_pred)
        com = y_pred_class == y_true
        correct = com.all(dim=1).sum()
        sample_num = y_true.size(0)
        accuracy = correct / sample_num
        return accuracy
    def validation_step(self, batch, batch_idx):
        x, mask, y, parnet_embed = batch
        # x, y = batch
        y_pred = self.forward(x, mask, parnet_embed)
        # y_pred = self.forward(x, mask, parnet_input)
        # y_pred = self.forward(x)
        # print("validation x ", x, x.shape)
        # print("validation y_pred 0 ", y_pred, y_pred.shape)
        # print("validation y_true 0 ", y, y.shape)
        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)
        # auroc = roc_auc_score(y[:,0], y_pred[:,0])
        self.log('val categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        # val_loss = self.loss_fn(y_pred, y)
        # self.log("val_loss", val_loss, on_epoch = True, on_step = True)
        # loss = self.weighted_binary_cross_entropy(y_pred, y)
        loss = self.loss_fn(y_pred, y)
        # print("this loss:", loss)
        l1_regularization = torch.tensor(0., device="cuda")
        for name, param in self.network.named_parameters(): 
            if 'Attention1.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W2' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W2' in name:
                l1_regularization += torch.norm(param, p=1)

            elif 'Attention1.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention1.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            
            elif 'Attention2.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention2.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            
            elif 'Attention3.W_q' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_k' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_v' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention3.W_o' in name:
                l1_regularization += torch.norm(param, p=1)
            
            
            #adding l1 regularization for weight parnet
            # elif 'parnet_weight' in name:
            #     l1_regularization += torch.norm(param, p=1)
        if self.attention and self.att_type == "self_attention":
            # l1_regularization += torch.norm(self.network.att1_A, p='fro')
            # l1_regularization += torch.norm(self.network.att2_A, p='fro')
            # l1_regularization += torch.norm(self.network.att3_A, p='fro')
            loss += l1_regularization*0.001
            #add the Attention regulizer
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(torch.transpose(self.network.att1_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att2_A, 1, 2))
                loss += self._attention_regularizer(torch.transpose(self.network.att3_A, 1, 2))
        if self.attention and self.att_type == "transformer":
            loss += l1_regularization*0.001
            if self.cnn_scaler == 1:
                loss += self._attention_regularizer(self.network.att1_A)
            elif self.cnn_scaler == 2:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)
            elif self.cnn_scaler == 3:
                loss += self._attention_regularizer(self.network.att1_A)
                loss += self._attention_regularizer(self.network.att2_A)    
                loss += self._attention_regularizer(self.network.att3_A)


        # with tf.Session() as sess:
        #     loss = torch.tensor(sess.run(loss))
        self.log("val_loss", loss, on_epoch = True, on_step = True)
        # self.log("auROC", auroc, on_epoch = True, on_step = True)

        return {"categorical_accuracy": categorical_accuracy, "categorical_accuracy_strict":categorical_accuracy_strict,
                "binary_accuracy": binary_accuracy}
    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            print(f"{param_name}: {param_value}")



