import datetime
import itertools
from collections import OrderedDict
import argparse
import os
import sys
import torch
device = "cuda"
basedir='/home/sxr280/DeepRBPLoc'
sys.path.append(basedir)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning import loggers as pl_loggers
from multihead_attention_model_torch_sequential_modifymuchpossible import *
from multihead_attention_model_torch_early_fusion_modifymuchpossible import *
from multihead_attention_model_torch_boost_modifymuchpossible import *
from Genedata import Gene_data
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold, StratifiedKFold
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import gin 
import wandb
import re
from torchinfo import summary
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef,f1_score



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
        print("spliting the data into:%s folds" % foldnum)
        print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
        #print(type(Train[i]))
        #print(Train[0][:foldnum])
        np.savetxt(datasetfolder+'/Train' + str(foldnum) +str(i)+'.txt', np.asarray(Train[i]),fmt="%s")
        np.savetxt(datasetfolder+'/Test' + str(foldnum) +str(i)+'.txt', np.asarray(Test[i]),fmt="%s")
        np.savetxt(datasetfolder+'/Val' + str(foldnum) +str(i)+'.txt', np.asarray(Val[i]),fmt="%s")
    
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


def preprocess_data(left=4000, right=4000, dataset='/home/sxr280/DeepRBPLoc/testdata/modified_multilabel_seq_nonredundent.fasta',padmod='center',pooling_size=8, foldnum=4, pooling=True):
    gene_data = Gene_data.load_sequence(dataset, left, right)
    id_label_seq_Dict = get_id_label_seq_Dict(gene_data)
    label_id_Dict = get_label_id_Dict(id_label_seq_Dict)
    Train=OrderedDict()
    Test=OrderedDict()
    Val=OrderedDict()
    datasetfolder=os.path.dirname(dataset)
    if os.path.exists(datasetfolder+'/Train5'+str(0)+'.txt'):
        for i in range(5):
            Train[i] = np.loadtxt(datasetfolder+'/Train5'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Train')[:]
            Test[i] = np.loadtxt(datasetfolder+'/Test5'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Test')[:]
            Val[i] = np.loadtxt(datasetfolder+'/Val5'+str(i)+'.txt',dtype='str')#HDF5Matrix(os.path.join('../mRNA_multi_data_keepnum_code/', 'datafold'+str(i)+'.h5'), 'Val')[:]
    else:
        [Train, Test,Val] = group_sample(label_id_Dict,datasetfolder,foldnum)
    
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
    
    for i in range(foldnum):
        #if i <2:
        #   continue
        
        print('padding and indexing data')
        encoding_keys = seq_encoding_keys
        encoding_vectors = seq_encoding_vectors
        #train
        #padd center
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Train[i]]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Train[i]]
        if padmod =='center':
            mask_label_left = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left-len(gene))]) for gene in X_left],dtype='float32')
            mask_label_right = np.array([np.concatenate([np.zeros(right-len(gene)),np.ones(len(gene))]) for gene in X_right],dtype='float32')
            mask_label = np.concatenate([mask_label_left,mask_label_right],axis=-1)
            Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            X_left = tf.keras.utils.pad_sequences(X_left,maxlen=left,
                               dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')  #padding after sequence
            
            X_right = tf.keras.utils.pad_sequences(X_right,maxlen=right,
                          dtype=np.int8, value=encoding_keys.index('UNK'),padding='pre')# padding before sequence
            
            Xtrain[i] = np.concatenate([X_left,X_right],axis = -1)
        else:
           #merge left and right and padding after sequence
           Xall = [np.concatenate([x,y],axis=-1) for x,y in zip(X_left,X_right)]
           Xtrain[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
           #mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
           #Train_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
           if pooling == False:
               maxpoolingmax=8000
               Train_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')
           else:
               Train_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')

               

        Ytrain[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Train[i]])
        print("training shapes"+str(Xtrain[i].shape)+" "+str(Ytrain[i].shape))
        
        #test
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Test[i]]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Test[i]]
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
            Xtest[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
            #mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
            #Test_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            # Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
            if pooling == False:
               maxpoolingmax=8000
               Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')
            else:
               Test_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')



        Ytest[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Test[i]])
        #validation
        X_left = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][0]] for id in Val[i]]
        X_right = [[encoding_keys.index(c) for c in list(id_label_seq_Dict[id].values())[0][1]] for id in Val[i]]
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
            Xval[i] = pad_sequences(Xall,maxlen=left+right,dtype=np.int8, value=encoding_keys.index('UNK'),padding='post')
            #mask_label = np.array([np.concatenate([np.ones(len(gene)),np.zeros(left+right-len(gene))]) for gene in Xall],dtype='float32')
            #Val_mask_label[i]=maxpooling_mask(mask_label,pool_length=pooling_size)
            # Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')
            if pooling == False:
               maxpoolingmax=8000
               Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene))),np.zeros(maxpoolingmax-int(len(gene)))]) for gene in Xall],dtype='float32')
            else:
               Val_mask_label[i]=np.array([np.concatenate([np.ones(int(len(gene)/pooling_size)),np.zeros(maxpoolingmax-int(len(gene)/pooling_size))]) for gene in Xall],dtype='float32')



        Yval[i] = np.array([label_dist(list(id_label_seq_Dict[id].keys())[0]) for id in Val[i]])
    
    return Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors




def gin_file_to_dict(file_path):
    with open(file_path, 'r') as f:
        gin_config = f.read()
    # Extract configuration lines from the gin file
    config_lines = re.findall(r'^\w[\w\.]+ = .+', gin_config, re.MULTILINE)
    # Convert the configuration lines into a dictionary
    config_dict = {}
    for line in config_lines:
        key, value = line.split(' = ')
        config_dict[key] = value
    return config_dict



def run_parnet(data , device, batch_size_parnet):
    encoding_seq = OrderedDict([
            ('UNK', [0, 0, 0, 0]),
            ('A', [1, 0, 0, 0]),
            ('C', [0, 1, 0, 0]),
            ('G', [0, 0, 1, 0]),
            ('T', [0, 0, 0, 1]),
            ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
        ])
    # device = "cpu"
    embedding_vec = np.array(list(encoding_seq.values()))

    parnet_model = torch.load("/home/sxr280/DeepRBPLoc/parnet_model/network.PanRBPNet.2023-03-13.ckpt")
    embedding_layer = nn.Embedding(num_embeddings=len(embedding_vec),embedding_dim=len(embedding_vec[0]),_weight=torch.tensor(embedding_vec))
    embedding_layer = embedding_layer.to(device=device)
    parnet_model = parnet_model.to(device=device)
    for param in parnet_model.parameters():
        param.requires_grad = False

    parnet_model.eval()
    parnet_out = np.zeros((data.shape[0], 1, 8000))
    # print("parnet_out ini:", parnet_out.shape)
    totalbatchs = int(np.ceil(float(data.shape[0]/batch_size_parnet)))
    
    for batch in range(totalbatchs):
        x = data[batch*batch_size_parnet: min((batch+1)*batch_size_parnet, data.shape[0])]
        # if device == "cpu":
        x = x.to(device)
        x = x.long()
        embedding_output = embedding_layer(x)
        embedding_output = embedding_output.transpose(1,2)
        
        # print("embedding_output:",embedding_output.shape)
        out = parnet_model.forward(embedding_output)
        # print("parnet output:", out, out.shape)
        out = out.detach().cpu().numpy()
        # print("batch:", batch)
        # print("batch*batch_size_parnet", batch*batch_size_parnet)
        # print("batch*(batch_size_parnet + 1)", batch*(batch_size_parnet + 1))
        # print("data.shape[0]", data.shape[0])
        # print("min(batch*(batch_size_parnet + 1), data.shape[0])", min(batch*(batch_size_parnet + 1), data.shape[0]))
        parnet_out[batch*batch_size_parnet: min((batch+1)*batch_size_parnet, data.shape[0])] = out
    return parnet_out


def evaluation(fold, batch_size, model):
    model = model.to(device)
    model.eval()
    roc_auc = dict()
    average_precision = dict()
    mcc_dict=dict()
    F1_score = dict()

    y_test = np.load("./testdata/Test_fold%s_y.npy" % fold)
    X_test = np.load("./testdata/Test_fold%s_X.npy" % fold)
    x_mask = np.load("./testdata/Test_fold%s_mask.npy" % fold)

    y_test = torch.from_numpy(y_test)#.to(device, torch.float)
    X_test = torch.from_numpy(X_test)#.to(device, torch.float)
    X_mask = torch.from_numpy(x_mask)#.to(device, torch.float)


    test_dataset = torch.utils.data.TensorDataset(X_test, X_mask, y_test)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    all_y_pred = []
    all_y_test = []

    
    for i, batch in enumerate(dataloader_test):
        print("doing evaluation:", i)
        X_test, X_mask, y_test = batch
        print("device check out")
        print(X_test.device)
        print(X_mask.device)
        print(model.device)
        y_pred = model.forward(X_test, X_mask)
        y_pred = y_pred.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)

    y_test = np.concatenate(all_y_test, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    print("y_test shape", y_test.shape)
    print("y_test shape", y_pred.shape)
    for i in range(7):#calculate one by one
        average_precision.setdefault(fold,{})[i+1] = average_precision_score(y_test[:, i], y_pred[:, i])
        roc_auc.setdefault(fold,{})[i+1] = roc_auc_score(y_test[:,i], y_pred[:,i])
        mcc_dict.setdefault(fold,{})[i+1] = matthews_corrcoef(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])
        F1_score.setdefault(fold,{})[i+1] = f1_score(y_test[:,i],[1 if x>0.5 else 0 for x in y_pred[:,i]])

    average_precision.setdefault(fold,{})["micro"] = average_precision_score(y_test, y_pred,average="micro")
    roc_auc.setdefault(fold,{})["micro"] = roc_auc_score(y_test,y_pred,average="micro")
    y_pred_bi = np.where(y_pred > 0.5, 1, 0)
    F1_score.setdefault(fold,{})["micro"] = f1_score(y_test, y_pred_bi, average='micro')
    # print("run", run)
    print("auprc:", average_precision)
    print("roauc:", roc_auc)
    print("F1 score:", F1_score)
    print("mcc score:", mcc_dict)

    
# starts training in CNN model

def run_model(dataset='/home/sxr280/DeepRBPLoc/testdata/modified_multilabel_seq_nonredundent.fasta',
              pooling_size = 8,
              pooling = True,
              left = 4000, 
              right = 4000, 
              padmod = "after", 
              foldnum = 5, 
              gpu_num = 1,
              run = 33,
              max_epochs = 500,
              message = "sequantial_fold2"):
    embedding_vec = seq_encoding_vectors
    OUTPATH = os.path.join(basedir,'Results/'+message + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    datasetfolder=os.path.dirname(dataset)
    wandb.login(key="57f4851d7943ea1dec3b10273876045d051b40f1")
    Xtrain,Ytrain,Train_mask_label,Xtest, Ytest,Test_mask_label,Xval,Yval,Val_mask_label, encoding_keys, encoding_vectors = preprocess_data(left = left, right = right, dataset = dataset, padmod = padmod, pooling_size=pooling_size,foldnum=foldnum, pooling=pooling)

    # print("saving the test dataset:...")
    i = int(message[-1])
    fold = i

    # #loading the datasets that are preprocessed
    Xtrain = torch.from_numpy(Xtrain[i])#.to(device, torch.int)
    Ytrain = torch.from_numpy(Ytrain[i])#.to(device, torch.float)
    Train_mask_label = torch.from_numpy(Train_mask_label[i])#.to(device, torch.int)

    Xval = torch.from_numpy(Xval[i])#.to(device, torch.int)
    Yval = torch.from_numpy(Yval[i])#.to(device, torch.float)
    Val_mask_label = torch.from_numpy(Val_mask_label[i])#.to(device, torch.int)


    Xtest = torch.from_numpy(Xtest[i])#.to(device, torch.int)
    Ytest = torch.from_numpy(Ytest[i])#.to(device, torch.float)
    Test_mask_label = torch.from_numpy(Test_mask_label[i])#.to(device, torch.int)
    


    train_dataset = torch.utils.data.TensorDataset(Xtrain, Train_mask_label, Ytrain)
    val_dataset = torch.utils.data.TensorDataset(Xval, Val_mask_label, Yval)
    #parameters that you can do the gridsearch
    hyperparams_1 = {
        'fc_dim': 100,
        'weight_decay': 1e-5,
        'attention': True,####need to be changed
        'lr':0.001,
        'drop_flat':0.4,
        'drop_cnn': 0.3,
        'drop_input': 0.3,
        'hidden':256,
        'pooling_opt':True,
        'filter_length1':3,
        'activation':"gelu",
        'optimizer':"torch.optim.Adam",
        'release_layers': 20,
        'prediction':False,
        'fc_layer' : True,
        'cnn_scaler': 1,
        'headnum': 3,
        'mode' : "full"
        }
    
    hyperparams_2 = {
        'filters_length1': 5,
        'filters_length2': 20,
        'filters_length3': 60,
        'headnum': 3,
        'nb_filters': 64,
        'hidden': 32,
        'dim_attention': 80,
        'mode' : "feature"
        }
    #fusion: use DM3Loc(get after attention layer) and parnet(get 256 layer) structure.
    #
    hyperparams_3 = {
        'hidden': 544,
        'fc_dim': 100,
        'drop_flat': 0.4,
        'batch_size': 32,
        'patience':20,
        'mode':"fusion"
        }
    
    hyperparams = {
        "param_1": hyperparams_1,
        "param_2": hyperparams_2,
        "param_3": hyperparams_3,
        }

    wandb_logger = WandbLogger(name = str(hyperparams_1), project = "5_folds_fusion_2_paper", log_model = "all", save_dir = OUTPATH + "/checkpoints_%s" % run)
    # for index, params  in enumerate(ParameterGrid(hyperparams)):
    index = 0
    print("running the fold: ", fold)
    print("Doing:", hyperparams)

    params_1 = hyperparams["param_1"]
    params_2 = hyperparams["param_2"]
    params_3 = hyperparams["param_3"]

    batch_size_params = {'batch_size': params_3.pop('batch_size')}
    patience_params = {'patience': params_3.pop('patience')}


    # model1 = myModel1(**params_1)
    # model2 = myModel2(**params_2)
    # model = myModel_fusion(model1, model2, **params_3)
    params_3["batch_size"] = batch_size_params['batch_size']
    params_3["patience"] = patience_params['patience']
    # model = model.to(device = device)

    model = myModel1(**params_1)
    
    if pooling:
        summary(model, input_size = [(2,8000),(2,1000)], device = device)
    else:
        summary(model, input_size = [(2,8000),(2,8000)], device = device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is not trainable")
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size_params['batch_size'], shuffle = True)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size_params['batch_size'], shuffle = True)

    # with wandb.init(reinit=True):
    
    trainer = Trainer(max_epochs = max_epochs, gpus = gpu_num, 
            logger = wandb_logger,
            log_every_n_steps = 1,
            callbacks = make_callback(OUTPATH, str(run), patience_params['patience']))

    trainer.fit(model, dataloader_train, dataloader_val)
    print("Saving the model not wrapped by pytorch-lighting")
    torch.save(model.network, OUTPATH + "/model%s_%s.pth" % (run, index))
    
    #Doing the prediction
    print("----------Doing the evaluation----------")
    
    evaluation(fold, 8, model)
 


    
class PredictionCallback(pl.Callback):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def on_batch_end(self, trainer, pl_module):
        x, mask, y = next(iter(self.dataloader))
        x = x.to(pl_module.device)
        y_pred = pl_module(x, mask)
        print('Batch', trainer.global_step, 'training predictions:', y_pred)
def make_callback(output_path, msg, patience):
    """
    save the parameters we trained during each epoch.
    Params:
    -------
    output_path: str,
        the prefix of the path to save the checkpoints.
    freeze: bool,
        whether to freeze the first section of the model. 
    """

    callbacks = [
        ModelCheckpoint(dirpath = output_path + "/checkpoints_%s" % str(msg), every_n_epochs = 1, save_last = True),
        EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = patience, verbose = True, mode = "min"),
        ]
    return callbacks

def _make_loggers(output_path, msg):
    loggers = [
        pl_loggers.TensorBoardLogger(output_path + '/logger', name=str(msg), version='', log_graph=True),
    ]
    return loggers



if __name__ == "__main__":
    gin.parse_config_file('/home/sxr280/DeepRBPLoc/Multihead_train_torch_sequential_modifymuchpossible.gin')
    run_model()

