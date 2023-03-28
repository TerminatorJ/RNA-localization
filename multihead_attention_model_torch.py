import torch.nn as nn
import click
import torch
from collections import OrderedDict
from torch_position_embedding import PositionEmbedding
from hier_attention_mask_torch import Attention_mask
import sys
import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
from keras.layers import Embedding,Input
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from pytorch_lightning import Trainer
import pytorch_lightning as pl
# from torchsummary import summary
from torchinfo import summary
import time
import math
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import BinaryAccuracy

class DM3Loc(nn.Module):
    def __init__(self, max_len, nb_classes, save_path, kfold_index, embedding_vec,
                       dim_attention, headnum,nb_filters=32,filters_length1=1,
                       filters_length2=5,filters_length3=10,pooling_size=8,
                       drop_input=0.2, drop_cnn=0.2, drop_flat=0.26, W1_regularizer=0.001,
                       W2_regularizer=0.005, Att_regularizer_weight=0.0005,
                       BatchNorm=False, fc_dim = 50, fcnum=0, posembed=False,
                       posmod = 'concat',normalizeatt=False,
                       attmod = "softmax", sharp_beta=1):
                                                 
                                                 
        super(DM3Loc, self).__init__()
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.is_built = False
        global OUTPATH
        OUTPATH = save_path

        #layer define
        self.dropout1 = nn.Dropout(drop_cnn)
        dropout1 = nn.Dropout(drop_cnn)
        self.dropout2 = nn.Dropout(drop_flat)
        self.dropout3 = nn.Dropout(drop_input)
        # print("BatchNorm1d",nb_filters,type(nb_filters))
        # self.bn = nn.BatchNorm1d(nb_filters//2)
        self.maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        #First scale of CNN
        CNN1_1 = nn.Conv1d(4, nb_filters, kernel_size = filters_length1, padding='same', bias = False)
        CNN1_2 = nn.Conv1d(nb_filters, int(nb_filters)//2, kernel_size = filters_length1, padding='same', bias = False)
        self.first_cnn_out = nn.Sequential(CNN1_1, CNN1_2, maxpool, dropout1)
        
        CNN2_1 = nn.Conv1d(4, nb_filters, kernel_size = filters_length2, padding='same', bias = False)
        CNN2_2 = nn.Conv1d(nb_filters, int(nb_filters)//2, kernel_size = filters_length2, padding='same', bias = False)
        self.second_cnn_out = nn.Sequential(CNN2_1, CNN2_2, maxpool, dropout1)  

        CNN3_1 = nn.Conv1d(4, int(nb_filters)//2, kernel_size = filters_length3, padding='same', bias = False)
        CNN3_2 = nn.Conv1d(int(nb_filters)//2, int(nb_filters)//2, kernel_size = filters_length3, padding='same', bias = False)
        self.third_cnn_out = nn.Sequential(CNN3_1, CNN3_2, maxpool, dropout1)  
 

        # self.position_embedding1 = PositionEmbedding(num_embeddings = int(max_len)//int(pooling_size), embedding_dim=int(nb_filters)//2, mode=PositionEmbedding.MODE_ADD)
        # self.position_embedding2 = PositionEmbedding(num_embeddings = int(max_len)//int(pooling_size), embedding_dim=int(nb_filters)//2, mode=PositionEmbedding.MODE_ADD)               
        # self.position_embedding3 = PositionEmbedding(num_embeddings = int(max_len)//int(pooling_size), embedding_dim=int(nb_filters)//2, mode=PositionEmbedding.MODE_ADD)

        self.fc1 = nn.Linear(480, fc_dim)
        self.fc2 = nn.Linear(fc_dim, nb_classes)
        #attention layers
        self.Attention1 = Attention_mask(hidden=int(nb_filters)//2, da=dim_attention, r=headnum, activation='tanh',return_attention=True, 
                                     attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                     sharp_beta=sharp_beta)
        self.Attention2 = Attention_mask(hidden=int(nb_filters)//2, da=dim_attention, r=headnum, activation='tanh',return_attention=True, 
                                     attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                     sharp_beta=sharp_beta)
        self.Attention3 = Attention_mask(hidden=int(nb_filters)//2, da=dim_attention, r=headnum, activation='tanh',return_attention=True, 
                                     attention_regularizer_weight=Att_regularizer_weight,normalize=normalizeatt,attmod=attmod,
                                     sharp_beta=sharp_beta)

        #embedding layer
        self.embedding = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=max_len,
                                    trainable=False)
        # self.embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]),_weight=torch.tensor(embedding_vec))
        self.embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]))
 
        self.embedding_layer.weight.requires_grad = False

        #activation
        # self.activation = nn.GELU()
        #regulazation
        self.L1loss = nn.L1Loss(W1_regularizer)
        #flatten
        self.flatten = nn.Flatten()
        # summary(self.forward, input_size = [(2,8000),(2,1000)], device = "cpu")
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

    def forward(self, x, x_mask):
        # print("x_mask",x_mask.shape)
        # print("x",x)
        # print("x dtype", x.dtype)
        # print("x shape",x.shape)
        # input = Input(shape=(x.size(1),), dtype='int8')
        x = x.long()
        # out = self.embedding(input)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     x = torch.tensor(sess.run(out, feed_dict={input: x}))
        # print("after embedding:",x)
        # print("########checking#########")
        # print("before embedding:", x)
        # embedding_output = self.dropout3(x)
        # embedding_output = self.dropout3(self.embedding_layer(x))
        embedding_output = self.embedding_layer(x)
        # print("embedding_output:",embedding_output)
        # print("after embedding:", embedding_output)
        # print("embedding_output",embedding_output,embedding_output.shape)
        embedding_output = embedding_output.transpose(1,2)
        embedding_output = embedding_output.float()
        # print("embedding_output dtype", embedding_output.dtype)
        # print("embedding_output:", embedding_output.shape, embedding_output)
        # print("ckp1:", embedding_output)
        # print("first cnn out before activation",self.first_cnn_out(embedding_output).transpose(1,2),self.first_cnn_out(embedding_output).transpose(1,2).shape)
        x1 = self.gelu(self.first_cnn_out(embedding_output))
        # print("first cnn out",x1.transpose(1,2),x1.transpose(1,2).shape)
        x2 = self.gelu(self.second_cnn_out(embedding_output))
        x3 = self.gelu(self.third_cnn_out(embedding_output))
        # print("ckp2:", x1)
        # print("x_mask",x_mask.shape)
        # print("x1 shape", x1.shape)
        x_mask = x_mask.unsqueeze(1)
        # print("position_embedding1",self.position_embedding1(x1).shape)
        # pos_embx1 = self.position_embedding1(x1)*x_mask
        # cnn_mask_output1_1 = torch.bmm(x1, x_mask)
        # print("cnn_mask_output1-1 shape", cnn_mask_output1_1.shape)
        cnn_mask_output1 = x1*x_mask
        # print("ckp3:", cnn_mask_output1)
        # print("cnn_mask_output1 shape", cnn_mask_output1.shape)
        cnn_mask_output1 = torch.cat((cnn_mask_output1,x_mask), dim = 1)
        # print("ckp4:", cnn_mask_output1)
        # pos_embx2 = self.position_embedding2(x2)*x_mask
        cnn_mask_output2 = x2*x_mask
        cnn_mask_output2 = torch.cat((cnn_mask_output2,x_mask), dim = 1)
        # pos_embx3 = self.position_embedding3(x3)*x_mask
        cnn_mask_output3 = x3*x_mask
        cnn_mask_output3 = torch.cat((cnn_mask_output3,x_mask), dim = 1)
        # print("cnn_mask_output1.shape", cnn_mask_output1.shape)
        att1,att1_A = self.Attention1(cnn_mask_output1)
        att2,att2_A = self.Attention2(cnn_mask_output2)
        att3,att3_A = self.Attention3(cnn_mask_output3)
        # print("ckp5:", att1,att1_A)
        # att1 = self.bn(att1)
        # att2 = self.bn(att2)
        # att3 = self.bn(att3)
        
        output = self.dropout2(self.flatten(torch.cat((att1, att2, att3), dim = 1)))
        # print("ckp6:", output)
        # print("output1 shape", output.shape)
        output = self.dropout2(torch.relu(self.fc1(output)))
        # print("ckp7:", output)
        # print("output2 shape", output.shape)
        pred = torch.sigmoid(self.fc2(output))
        # print("ckp8:", pred)
        # print("pred shape:", pred.shape)
        
        return pred

    def mask_func(x):
        return x[0] * x[1]



class myModel(pl.LightningModule):
    def __init__(self, network = None, optimizer = None, lr = 1e-3, class_weights = None):
        super(myModel, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.lr = lr
        self.class_weights = class_weights
        # self.loss_fn = nn.BCEWithLogitsLoss(weight = class_weights)
        self.loss_fn = nn.BCELoss(weight = class_weights)
        self.optimizer_cls = optimizer
        self.train_loss = []
        self.val_binary_acc = []
        self.val_Multilabel_acc = []

    def weighted_binary_cross_entropy_with_logits(self, output, target, weights):
        if weights is not None:
            loss = weights * (target * torch.log(torch.sigmoid(output))) + \
                weights * ((1 - target) * torch.log(1 - torch.sigmoid(output)))
        else:
            loss = target * torch.log(torch.sigmoid(output)) + (1 - target) * torch.log(1 - torch.sigmoid(output))

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

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), lr = self.lr, weight_decay = 5e-5)
        return optimizer

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch duration: {epoch_time:.2f} seconds")

    def training_step(self, batch, batch_idx, **kwargs):
        x, mask, y = batch
        y_pred = self.forward(x, mask)
        
        loss = self.loss_fn(y_pred, y)
        # loss = self.weighted_binary_cross_entropy_with_logits(y_pred, y, weights = self.class_weights)
        # print("training step loss:", y_pred, y, loss)
        self.log("train_loss", loss, on_epoch = True, on_step = True)
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
        # print("length of val batch:", len(batch))
        # print("batch:", batch)
        x, mask, y = batch
        # print()
        y_pred = self.forward(x, mask)
        #using the same metric between training and validation steps
        # metric_MultilabelAccuracy = MultilabelAccuracy(num_labels=7).to("cuda")
        # Multilabel_acc = metric_MultilabelAccuracy(y_pred, y)
        # print("y_pred, y, x, mask",y_pred, y, x, mask)

        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)
        self.log('categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        val_loss = self.loss_fn(y_pred, y)
        # val_loss = self.weighted_binary_cross_entropy_with_logits(y_pred, y, weights = self.class_weights)
        # print("validation step loss:", y_pred, y, val_loss)
        self.log("val_loss", val_loss, on_epoch = True, on_step = True)

        return {"categorical_accuracy": categorical_accuracy}



