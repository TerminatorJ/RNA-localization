import datetime
import itertools
from collections import OrderedDict
import argparse
import os
import sys
import pickle
import h5py
from tqdm import tqdm
# from data_generator import *
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
basedir='./'
sys.path.append(basedir)
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
set_session(session=sess)

from multihead_attention_model_2 import *
from Genedata import Gene_data
from keras.preprocessing.sequence import pad_sequences
from keras.layers import MaxPooling1D
from sklearn.model_selection import KFold, StratifiedKFold

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))

gene_ids = None

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights


def get_id_label_seq_Dict(gene_data):
    id_label_seq_Dict = OrderedDict()
    for gene in gene_data:
         label = gene.label
         gene_id = gene.id.strip()
         id_label_seq_Dict[gene_id] = {}
         id_label_seq_Dict[gene_id][label]= (gene.seqleft,gene.seqright)
    
    return id_label_seq_Dict


def get_label_id_Dict(id_label_seq_Dict):
    label_id_Dict = OrderedDict()
    for eachkey in id_label_seq_Dict.keys():
        label = list(id_label_seq_Dict[eachkey].keys())[0]
        label_id_Dict.setdefault(label,set()).add(eachkey)
    
    return label_id_Dict

def typeicalSampling(ids, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    folds = kf.split(ids)
    train_fold_ids = OrderedDict()
    val_fold_ids = OrderedDict()
    test_fold_ids=OrderedDict()
    for i, (train_indices, test_indices) in enumerate(folds):
        size_all = len(train_indices)
        train_fold_ids[i] = []
        val_fold_ids[i] = []
        test_fold_ids[i]  =[]
        train_indices2 = train_indices[:int(size_all * 0.8)]
        val_indices = train_indices[int(size_all * 0.8):]
        for s in train_indices2:
             train_fold_ids[i].append(ids[s])
        
        for s in val_indices:
             val_fold_ids[i].append(ids[s])
        
        for s in test_indices:
              test_fold_ids[i].append(ids[s])
        
    
    return train_fold_ids,val_fold_ids,test_fold_ids

def group_sample(label_id_Dict,datasetfolder,foldnum=8):
    Train = OrderedDict()
    Test = OrderedDict()
    Val = OrderedDict()
    for i in range(foldnum):
        Train.setdefault(i,list())
        Test.setdefault(i,list())
        Val.setdefault(i,list())
    
    for eachkey in label_id_Dict:
        label_ids = list(label_id_Dict[eachkey])
        if len(label_ids)<foldnum:
            for i in range(foldnum):
                Train[i].extend(label_ids)
            
            continue
        
        [train_fold_ids, val_fold_ids,test_fold_ids] = typeicalSampling(label_ids, foldnum)
        for i in range(foldnum):
            Train[i].extend(train_fold_ids[i])
            Val[i].extend(val_fold_ids[i])
            Test[i].extend(test_fold_ids[i])
            print('label:%s finished sampling! Train length: %s, Test length: %s, Val length:%s'%(eachkey, len(train_fold_ids[i]), len(test_fold_ids[i]),len(val_fold_ids[i])))
    
    for i in range(foldnum):
        print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
        #print(type(Train[i]))
        #print(Train[0][:foldnum])
        np.savetxt(datasetfolder+'/Train8'+str(i)+'.txt', np.asarray(Train[i]),fmt="%s")
        np.savetxt(datasetfolder+'/Test8'+str(i)+'.txt', np.asarray(Test[i]),fmt="%s")
        np.savetxt(datasetfolder+'/Val8'+str(i)+'.txt', np.asarray(Val[i]),fmt="%s")
    
    return Train, Test, Val

def label_dist(dist):
    #assert (len(dist) == 4)
    return [int(x) for x in dist]

def maxpooling_mask(input_mask,pool_length=3):
    #input_mask is [N,length]
    max_index = int(input_mask.shape[1]/pool_length)-1
    max_all=np.zeros([input_mask.shape[0],int(input_mask.shape[1]/pool_length)])
    for i in range(len(input_mask)):
        index=0
        for j in range(0,len(input_mask[i]),pool_length):
            if index<=max_index:
                max_all[i,index] = np.max(input_mask[i,j:(j+pool_length)])
                index+=1
    
    return max_all
def new_one_hot(seq):
    encoded = np.zeros((4,len(seq)))
    ref = np.array(["A","C","G","T"])
    for i in range(len(seq)):
        encoded[:,i] = np.array(ref == seq[i],dtype="float32")
    return encoded





def preprocess_data(left, right,dataset,padmod='center',pooling_size=3, onehot=False):
    gene_data = Gene_data.load_sequence(dataset, left, right)
    gene_data = gene_data
    id_label_seq_Dict = get_id_label_seq_Dict(gene_data)
    label_id_Dict = get_label_id_Dict(id_label_seq_Dict)
    Train=OrderedDict()
    Test=OrderedDict()
    Val=OrderedDict()
    datasetfolder=os.path.dirname(dataset)
    if os.path.exists(datasetfolder+'/Train8'+str(0)+'.txt'):
        for i in range(8):
            Train[i] = np.loadtxt(datasetfolder+'/Train8'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Train')[:]
            Test[i] = np.loadtxt(datasetfolder+'/Test8'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Test')[:]
            Val[i] = np.loadtxt(datasetfolder+'/Val8'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Val')[:]
    else:
        [Train, Test,Val] = group_sample(label_id_Dict,datasetfolder)
    
    Xtrain={}
    Xtest={}
    Xval={}
    Ytrain={}
    Ytest={}
    Yval={}
    Train_mask_label={}
    Test_mask_label={}
    Val_mask_label={}
    maxpoolingmax = int((left+right)/pooling_size)
    
    for i in range(8):
        #   continue
        
        print('padding and indexing data')
        encoding_keys = seq_encoding_keys
        encoding_vectors = seq_encoding_vectors
        #train
        #padd center(initial)
        if onehot==False:
            X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Train[i]]
            X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Train[i]]
        #using different dimension as input
        #padd center(new)
        else:
            X_left = [new_one_hot(list(id_label_seq_Dict[id].values())[0][0]) for id in Train[i]]
            X_right = [new_one_hot(list(id_label_seq_Dict[id].values())[0][1]) for id in Train[i]]

        if padmod =='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xtrain[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
           #merge left and right and padding after sequence 4*len
           Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
           #padding(initial)
           if onehot == False:
               Xtrain[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
           #padding(new)
           else:
               Xtrain[i] = np.array(list(map(lambda x: pad_sequences(x,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post'),Xall)))
           mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
           #Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
           #old version
           if onehot == False:
               Train_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
           else:
               Train_mask_label[i]=np.array([np.concatenate([np.ones(int(gene.shape[1]/pooling_size)),np.zeros(maxpoolingmax-int(gene.shape[1]/pooling_size))]) for gene in Xall],dtype='float32')
           print(Train_mask_label)
        #one seq one label
        Ytrain[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Train[i]])
        print("training shapes"+str(Xtrain[i].shape)+" "+str(Ytrain[i].shape))
        
        #test
        if onehot == False:
            X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Test[i]]
            X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Test[i]]
        else:
            X_left = [new_one_hot(list(id_label_seq_Dict[id].values())[0][0]) for id in Test[i]]
            X_right = [new_one_hot(list(id_label_seq_Dict[id].values())[0][1]) for id in Test[i]]
        if padmod =='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Test_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xtest[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
            #merge left and right and padding after sequence
            Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
            if onehot == False:
                Xtest[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
            else:
                Xtest[i] = np.array(list(map(lambda x: pad_sequences(x,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post'),Xall)))
            #mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
            #Test_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            if onehot == False:
               Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
            else:
               Test_mask_label[i]=np.array([np.concatenate([np.ones(int(gene.shape[1]/pooling_size)),np.zeros(maxpoolingmax-int(gene.shape[1]/pooling_size))]) for gene in Xall],dtype='float32')
        
#         Ytest[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Test[i]])
        #new one
        Ytest[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Test[i]])
        #validation
        if onehot == False:
            X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Val[i]]
            X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Val[i]]
        else:
            X_left = [new_one_hot(list(id_label_seq_Dict[id].values())[0][0]) for id in Val[i]]
            X_right = [new_one_hot(list(id_label_seq_Dict[id].values())[0][1]) for id in Val[i]]
        if padmod=='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Val_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xval[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
            #merge left and right and padding after sequence
            Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
            if onehot == False:
                Xval[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
            #mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
            else:
                Xval[i] = np.array(list(map(lambda x: pad_sequences(x,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post'),Xall)))
            #Val_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            if onehot == False:
               Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
            else:
               Val_mask_label[i]=np.array([np.concatenate([np.ones(int(gene.shape[1]/pooling_size)),np.zeros(maxpoolingmax-int(gene.shape[1]/pooling_size))]) for gene in Xall],dtype='float32')
        
        Yval[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Val[i]])
    print("test mask shape and value",Train_mask_label[0])
    return Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors


def train_data_generator(batchs=256,fold=0):
    
    all_mask = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_mask.pickle","rb"))
    all_target = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_y.pickle","rb"))
    
    train_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Train8{fold}.npy") 
    print("loading the batched training data")
    with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Train8{fold}_4dim.h5","r") as f1:
        Xtrain = f1["RBP_encoded"].value
    train_batch_size = int(np.ceil(len(train_fold_index)/batchs))
    Xtrain = np.swapaxes(Xtrain,1,2)
    print(all_mask)
    while True:
        for batch in tqdm(range(batchs)):
            #read Xtrain
            Xtrain_out = Xtrain[batch*train_batch_size:min((batch+1)*train_batch_size,len(train_fold_index))]
    
            
            

            Train_mask_label = all_mask[train_fold_index][batch*train_batch_size:min((batch+1)*train_batch_size,len(train_fold_index))]      
            Ytrain = all_target[train_fold_index][batch*train_batch_size:min((batch+1)*train_batch_size,len(train_fold_index))]
            yield ([Xtrain_out, Train_mask_label.reshape(-1, Train_mask_label.shape[1],1)], Ytrain)
            
def val_data_generator(batchs=256,fold=0):
    
    all_mask = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_mask.pickle","rb"))
    all_target = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_y.pickle","rb"))
    
    val_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Val8{fold}.npy") 
    print("loading the batched validation data")
    with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Val8{fold}_4dim.h5","r") as f1:
         Xval = f1["RBP_encoded"].value
    Xval = np.swapaxes(Xval,1,2)
    val_batch_size = int(np.ceil(len(val_fold_index)/batchs))
#     print("validation all shape:",Xval.shape)
    #GPU initiated?
    print("GPU?",len(tf.test.gpu_device_name()))
    while True:
        for batch in tqdm(range(batchs)):
            #read Xtrain
            
            Xval_out = Xval[batch*val_batch_size:min((batch+1)*val_batch_size,len(val_fold_index))]
            
            
            print("validation shape:",Xval.shape)

            Val_mask_label = all_mask[val_fold_index][batch*val_batch_size:min((batch+1)*val_batch_size,len(val_fold_index))]        
            Yval = all_target[val_fold_index][batch*val_batch_size:min((batch+1)*val_batch_size,len(val_fold_index))] 
            yield ([Xval_out, Val_mask_label.reshape(-1, Val_mask_label.shape[1],1)], Yval)
            

def load_train(fold=0):
    all_mask = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_mask.pickle","rb"))
    all_target = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_y.pickle","rb"))
    
    train_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Train8{fold}.npy") 
    print("loading the batched training data")
    with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Train8{fold}_4dim.h5","r") as f1:
        Xtrain = f1["RBP_encoded"].value
              

    Train_mask_label = all_mask[train_fold_index]     
    Ytrain = all_target[train_fold_index]
    
    Xtrain = np.swapaxes(Xtrain,1,2)
    return Xtrain, Train_mask_label, Ytrain

def load_val(fold=0):
    all_mask = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_mask.pickle","rb"))
    all_target = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_y.pickle","rb"))
    
    val_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Val8{fold}.npy") 
    print("loading the batched validation data")
    with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Val8{fold}_4dim.h5","r") as f1:
        Xval = f1["RBP_encoded"].value
              

    Val_mask_label = all_mask[val_fold_index]     
    Yval = all_target[val_fold_index]
    Xval = np.swapaxes(Xval,1,2)
    return Xval, Val_mask_label, Yval


          

'''
def load_saved_data(fold=0,train_batch=50,test_batch=10,val_batch=10,dim=106,batch=243,pooling_size=1000):

    all_mask = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_mask.pickle","rb"))
    all_target = pickle.load(open("/binf-isilon/winthergrp/jwang/rbpnet/data/all_y.pickle","rb"))
    train_list = []
    test_list = []
    val_list = []
    print("loading all X")
    train_num = 0
    
    train_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Train8{fold}.npy") 
    test_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Test8{fold}.npy")
    val_fold_index = np.load(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Val8{fold}.npy")
    
    for i in tqdm(range(train_batch)):
        #read Xtrain
        with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Train8{fold}_{i}.h5","r") as f1:
            Xtrain = f1.value
        
        Train_mask_label = all_mask[train_fold_index]        
        Ytrain = all_target[train_fold_index]
        
        training_generator = DataGenerator(Xtrain, Ytrain, Train_mask_label, **params)

    for i in tqdm(range(test_batch)):
        with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Test8{fold}_{i}.h5","r") as f1:
            Xtest = f1.value
        
        Test_mask_label = all_mask[test_fold_index]        
        Ytest = all_target[test_fold_index]
        
        test_generator = DataGenerator(Xtest, Ytest, Test_mask_label, **params)
    for i in tqdm(range(val_batch)):
        with h5py.File(f"/binf-isilon/winthergrp/jwang/rbpnet/data/train_test_val/Val8{fold}_{i}.h5","r") as f1:
            Xval = f1.value
        
        Val_mask_label = all_mask[val_fold_index]        
        Yval = all_target[val_fold_index]        
        val_generator = DataGenerator(Xval, Yval, Val_mask_label, **params) 
        
    # Parameters
    params = {'dim': (50, 50),
          'batch_size': 20,
          'n_classes': 3,
          'n_channels': 3,
          'shuffle': True} 
    
    partition = {}
    partition["train"] = np.random.normal(size=(200,50,50,3))
    partition["val"] = np.random.normal(size=(10,50,50,3))

    labels = {}
    labels["train"] = np.random.randint(1,4,size=(200))
    labels["val"] = np.random.randint(1,4,size=(10))



    # Generators
    training_generator = DataGenerator(partition['train'], labels["train"], **params)
    validation_generator = DataGenerator(partition['val'], labels["val"], **params)


    
        
    Xtrain = np.vstack(train_list)      
    Train_mask_label = all_mask[:train_num*500]
    Ytrain = all_target[:train_num*500]
    
    Xtest = np.vstack(test_list)     
    Test_mask_label = all_mask[train_num*500:train_num*500+test_num*500]
    Ytest = all_target[train_num*500:train_num*500+test_num*500]
    
    Xval = np.vstack(val_list)    
    Val_mask_label = all_mask[train_num*500+test_num*500:min(train_num*500+test_num*500+val_num*500,all_target.shape[0])]
    Yval = all_target[train_num*500+test_num*500:min(train_num*500+test_num*500+val_num*500,all_target.shape[0])]
    
    if dim == 4:
        Xtrain = Xtrain[:,:4,:]
        Xtest = Xtest[:,:4,:]
        Xval = Xval[:,:4,:]
        
    return Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label
'''     
    
    


# starts training in CNN model
def run_model(lower_bound, upper_bound, max_len, dataset, **kwargs):
    
    pooling_size = kwargs['pooling_size'] #
    
    #pooling_size = int(kwargs['pooling_size']*kwargs['num_encoder']*2)
    print("pooling_size")
    print(pooling_size)
#     Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(kwargs['left'], kwargs['right'], dataset,padmod = kwargs['padmod'],pooling_size=pooling_size,onehot=False)
#     Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(kwargs['left'], kwargs['right'], dataset,padmod = kwargs['padmod'],pooling_size=pooling_size,onehot=True)
#     all_data = (Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors)
#     import pickle
#     pickle.dump(all_data,open("./onehot_true_data_V2.pickle","wb"))
#     Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label = load_saved_data(train_num=25,test_num=7,val_num=3,dim=4)
    
    all_data1 = pickle.load(open("./onehot_true_data_V2.pickle","rb"))
#     all_data2 = pickle.load(open("./onehot_true_data_V2_106_test.pickle","rb"))
#     Xtrain = all_data2[0]
#     Ytrain = all_data2[1]
#     Train_mask_label = all_data2[2]
#     Xtest = all_data2[3]
#     Ytest = all_data2[4]
#     Test_mask_label = all_data2[5]
#     Xval = all_data2[6]
#     Yval = all_data2[7]
#     Val_mask_label = all_data2[8]
    encoding_keys = all_data1[9]
    encoding_vectors = all_data1[10]
#     print("after transpose",Xtrain.shape)
    
    max_len = kwargs['left']+kwargs['right']
    
    
    # model mode maybe overridden by other parameter settings
    for i in range(1):#(kwargs['foldnum']):
#         print(Xtrain[i].shape)
#         print(Train_mask_label[i].shape)
        print('Evaluating KFolds {}/10'.format(i + 1))
        model = multihead_attention(max_len, kwargs['nb_classes'], OUTPATH, kfold_index=i)  # initialize
        model.build_model_multihead_attention_multiscaleCNN4_covermore(dim=4,
                                                 load_weights = kwargs['load_pretrain'],
                                                 weight_dir = kwargs['weights_dir'],
                                                 dim_attention=kwargs['dim_attention'],
                                                 headnum=kwargs['headnum'],
                                                 embedding_vec=encoding_vectors,
                                                 nb_filters=kwargs['nb_filters'],
                                                 filters_length1=kwargs['filters_length1'],
                                                 filters_length2=kwargs['filters_length2'],
                                                 filters_length3=kwargs['filters_length3'],
                                                 pooling_size=kwargs['pooling_size'],
                                                 drop_input=kwargs['drop_input'],
                                                 drop_cnn=kwargs['drop_cnn'],
                                                 drop_flat=kwargs['drop_flat'],
                                                 W1_regularizer=kwargs['W1_regularizer'],
                                                 W2_regularizer=kwargs['W2_regularizer'],
                                                 Att_regularizer_weight=kwargs['Att_regularizer_weight'],
                                                 BatchNorm=kwargs['BatchNorm'],
                                                 fc_dim = kwargs['fc_dim'],
                                                 fcnum = kwargs['fcnum'],
                                                 posembed=kwargs['posembed'],
                                                 pos_dmodel=kwargs['pos_dmodel'],
                                                 pos_nwaves = kwargs['pos_nwaves'],
                                                 posmod = kwargs['posmod'],
                                                 regularfun = kwargs['regularfun'],
                                                 huber_delta=kwargs['huber_delta'],
                                                 activation = kwargs['activation'],
                                                 activationlast = kwargs['activationlast'],
                                                 add_avgpooling = kwargs['add_avgpooling'],
                                                 poolingmod = kwargs['poolingmod'], #1 maxpooling 2 avgpooling
                                                 normalizeatt=kwargs['normalizeatt'],
                                                 attmod=kwargs['attmod'],
                                                 sharp_beta=kwargs['sharp_beta'],
                                                 lr = kwargs['lr']
                                                )
        
        if kwargs['nb_classes'] == 7:
           class_weights={0:1,1:1,2:7,3:1,4:3,5:5,6:8}
        
#         model.train(Xtrain[i], Ytrain[i],Train_mask_label[i], kwargs['batch_size'], kwargs['epochs'],Xval[i],Yval[i],Val_mask_label[i],loadFinal=kwargs['loadFinal'],classweight = kwargs['classweight'],class_weights=class_weights)

        #swaping the shape
#         Xtrain = np.swapaxes(Xtrain,1,2)
#         Xtest = np.swapaxes(Xtest,1,2)
#         Xval = np.swapaxes(Xval,1,2)
#         Xtrain[i] = Xtrain[i].astype("float32")
#         Xtest[i] = Xtest[i].astype("float32")
#         Xval[i] = Xval[i].astype("float32")
#         print(Xtrain[i])
#         print(Ytrain[i])
#         print("checking the nan values",np.isnan(Xtrain[i]).any())
#         print(Xtrain[i].shape,Xtest[i].shape,Xval[i].shape,Ytrain[i].shape)
#         model.train(Xtrain, Ytrain,Train_mask_label, kwargs['batch_size'], kwargs['epochs'],Xval,Yval,Val_mask_label,loadFinal=kwargs['loadFinal'],classweight = kwargs['classweight'],class_weights=class_weights)
        
        Xtrain, Train_mask_label, Ytrain = load_train(fold=0)
        Xval, Val_mask_label, Yval = load_val(fold=0)
#         model.train(train_data_generator(batchs=50,fold=0), val_data_generator(batchs=10,fold=0), kwargs['batch_size'], kwargs['epochs'],loadFinal=kwargs['loadFinal'],classweight = kwargs['classweight'],class_weights=class_weights)
        model.train(Xtrain, Ytrain,Train_mask_label, kwargs['batch_size'], kwargs['epochs'],Xval,Yval,Val_mask_label,loadFinal=kwargs['loadFinal'],classweight = kwargs['classweight'],class_weights=class_weights)
#         model.train(Xtrain[i], Ytrain[i], kwargs['batch_size'], kwargs['epochs'],Xval[i],Yval[i],loadFinal=kwargs['loadFinal'],classweight = kwargs['classweight'],class_weights=class_weights)
#         model.evaluate(Xtest[i], Ytest[i],Test_mask_label[i])
#         model.evaluate(Xtest, Ytest,Test_mask_label)
        
        K.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Model parameters'''
    parser.add_argument('--lower_bound', type=int, default=0, help='set lower bound on sample sequence length')
    parser.add_argument('--upper_bound', type=int, default=4000, help='set upper bound on sample sequence length')
    parser.add_argument('--max_len', type=int, default=4000,
                        help="pad or slice sequences to a fixed length in preprocessing")
    
    parser.add_argument('--left', type=int, default=4000, help='set left on sample sequence length')
    parser.add_argument('--right', type=int, default=4000, help='set left on sample sequence length')
    
    parser.add_argument('--dim_attention', type=int, default=80, help='dim_attention')
    parser.add_argument('--headnum', type=int, default=5, help='number of multiheads') #select one from 3
    parser.add_argument('--dim_capsule', type=int, default=4, help='capsule dimention')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--drop_input', type=float, default=0.06, help='dropout ratio')
    parser.add_argument('--drop_cnn', type=float, default=0.25, help='dropout ratio')
    parser.add_argument('--drop_flat', type=float, default=0.26, help='dropout ratio')
    
    parser.add_argument('--W1_regularizer', type=float, default=0.001, help='W1_regularizer')
    parser.add_argument('--W2_regularizer', type=float, default=0.001, help='W2_regularizer')
    parser.add_argument('--Att_regularizer_weight', type=float, default=0.001, help='Att_regularizer_weight')
    
    parser.add_argument('--dataset', type=str, default='../../mRNAsubloci_train.fasta', help='input sequence data')
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--nb_filters', type=int, default=64, help='number of CNN filters') 
    parser.add_argument('--filters_length1', type=int, default=9, help='kernel length for CNN filters1')
    parser.add_argument('--filters_length2', type=int, default=20, help='kernel length for CNN filters2') 
    parser.add_argument('--filters_length3', type=int, default=49, help='kernel length for CNN filters3') 
    parser.add_argument('--pooling_size', type=int, default=8, help='pooling_size') 
    parser.add_argument('--att_weight', type=float, default=1, help='number of att_weight') #select one from 3
    parser.add_argument("--BatchNorm", action="store_true",help="use BatchNorm")
    parser.add_argument("--loadFinal", action="store_true",help="whether loadFinal model")
    parser.add_argument('--fc_dim', type=int, default=100, help='fc_dim')
    parser.add_argument('--fcnum', type=int, default=1, help='fcnum')
    parser.add_argument('--sigmoidatt', type=int, default=0, help='whether sigmoidatt 0 no 1 yes') #select one from 3
    parser.add_argument("--message", type=str, default="", help="append to the dir name")
    parser.add_argument("--load_pretrain", action="store_true",
                        help="load pretrained CNN weights to the first convolutional layers")
    
    parser.add_argument("--weights_dir", type=str, default="",
                        help="Must specificy pretrained weights dir, if load_pretrain is set to true. Only enter the relative path respective to the root of this project.") 
    
    parser.add_argument("--randomization", type=int, default=None,
                        help="Running randomization test with three settings - {1,2,3}.") #use default none
    parser.add_argument("--posembed", action="store_true",help="use posembed")
    parser.add_argument("--pos_dmodel", type=int,default=40,help="pos_dmodel")
    parser.add_argument("--pos_nwaves", type=int,default=20,help="pos_nwaves")
    parser.add_argument("--posmod", type=str,default='concat',help="posmod")
    parser.add_argument("--regularfun",type=int,default=1,help = 'regularfun for l1 or l2 3 for huber_loss')
    parser.add_argument("--huber_delta",type=float,default=1.0,help = 'huber_delta')
    
    parser.add_argument("--activation",type=str,default='gelu',help = 'activation')
    parser.add_argument("--activationlast",type=str,default='gelu',help = 'activationlast')
    
    parser.add_argument("--add_avgpooling", action="store_true",help="add_avgpooling")
    parser.add_argument('--poolingmod',type=int,default=1,help = '1:maxpooling 2:avgpooling')
    parser.add_argument('--classweight', action="store_true", help='classweight')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument("--padmod", type=str,default='after',help="padmod: center, after")
    parser.add_argument("--normalizeatt", action="store_true",help="normalizeatt")
    parser.add_argument('--num_encoder', type=int, default=1, help='num_encoder')
    parser.add_argument('--lastCNN_length', type=int, default=1, help='lastCNN_length')
    parser.add_argument('--lastCNN_filter', type=int, default=128, help='lastCNN_filter')
    parser.add_argument("--attmod", type=str, default="smooth",help="attmod")
    parser.add_argument("--sharp_beta", type=int, default=1,help="sharp_beta")
    parser.add_argument("--lr",type=float,default=0.001,help = 'lr')
    parser.add_argument("--nb_classes",type=int,default=7,help = 'nb_classes')
    parser.add_argument('--foldnum', type=int, default=8, help='number of cross-validation folds') 
    
    args = parser.parse_args()
    OUTPATH = os.path.join(basedir,'Results/'+args.message + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)
    del args.message
    
    args.weights_dir = os.path.join(basedir, args.weights_dir)
    
    for k, v in vars(args).items():
        print(k, ':', v)
    
    run_model(**vars(args))



#use the remove data direct from fold
#python3 Multihead_train_4.py --normalizeatt --foldnum 5 --classweight  --epochs 300 --message train_V2_4 


