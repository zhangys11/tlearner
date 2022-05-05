from tensorflow.keras.applications import efficientnet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, BatchNormalization, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow.lite
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot, plot_model
import tensorflow.compat.v1.lite

import numpy as np
import os
import pathlib
import time
import pickle
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import shutil
from tqdm import tqdm
from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc


class transfer_learner():

    FLAVOR = 'efficient_net' # class attribute

    def __init__(self, MODEL_NAME, W = 224): # constructor
        # set instance attributes
        self.MODEL_NAME = MODEL_NAME
        self.W = W
        self.BEST_MODEL = None # not trained or loaded
        self.MASK = None

    def load_best_model(self):
        '''
        Reload the model from disk. 
        Notice: This method will release the current model object.
        '''
        del self.BEST_MODEL
        self.BEST_MODEL = None
        return self.get_best_model()

    def load_dataset_from_pkl(self, PKL_PATH = None):
        # PKL_PATH = '../data/fundus/C5/C5_'+str(W)+'x'+str(W)+'_202108.pkl'

        if PKL_PATH:
            with open(PKL_PATH, 'rb') as f:
                dict = pickle.load(f)
            
            self.X_train = dict['X_train']
            self.X_val = dict['X_val']
            self.y_train = dict['y_train']
            self.y_val = dict['y_val']
            
            self.num_classes = dict['num_classes']

            # labels and class_names are the same.
            self.labels = dict['class_names']
            self.class_names = dict['class_names']

        else:
            print("Unable to locate the pkl file: ", PKL_PATH) 


    def load_dataset(self, DIR, use_mask = False, test_split_size = 0.2, PKL_PATH = None):
        '''
        The dataset should be organization in DIR. Each subfolder is one class.
        '''

        # data_path = '../../data/flowers/flower_photos'
        data_subdir_list = os.listdir(DIR)
        print(data_subdir_list)
        
        self.num_classes = len(data_subdir_list)
        self.labels = data_subdir_list
        self.class_names = data_subdir_list

        img_data_list=[]
        y_labels = []

        if (use_mask):
            self.load_mask()
        
        for i, dataset in enumerate(data_subdir_list):
            img_list=os.listdir(DIR+'/'+ dataset)
            print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            for img in img_list:
                img_path = DIR + '/'+ dataset + '/'+ img     

                try:
                    img = image.load_img(img_path, target_size=(self.W, self.W))
                except OSError:
                    print('Image load error: ' + img_path)
                    continue
                
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                if (use_mask):
                    kill_border(x, self.MASK)

                x = efficientnet.preprocess_input(x)
                #x = x/255
                # print('Input image shape:', x.shape)
                img_data_list.append(x)
                y_labels.append(i)
                
        self.num_of_samples = len(img_data_list)
        img_data = np.array(img_data_list)

        #img_data = img_data.astype('float32')
        print (img_data.shape)
        img_data=np.rollaxis(img_data,1,0)
        print (img_data.shape)
        img_data=img_data[0]
        print (img_data.shape)

        # convert class labels to on-hot encoding
        Y = to_categorical(y_labels, self.num_classes)
        
        x, y = shuffle(img_data, Y)
        # Split the dataset
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(x, y, test_size = test_split_size)

        if PKL_PATH:
            # PKL_PATH = '../data/fundus/C5/C5_202108.pkl' # '../data/fundus/C5/C5_202108.pkl' # the unmasked imgs
            dict = {}

            dict['X_train'] = np.array(self.X_train, dtype=np.int8)
            dict['X_val'] = np.array(self.X_val, dtype=np.int8)
            dict['y_train'] = np.array(self.y_train, dtype=np.int8)
            dict['y_val'] = np.array(self.y_val, dtype=np.int8)
            
            dict['num_classes'] = self.num_classes
            dict['labels'] = self.labels
            dict['class_names'] = self.class_names            

            import pickle
            with open(PKL_PATH, 'wb') as f:
                pickle.dump(dict, f)

            print("pickled in ", PKL_PATH)

    def load_mask(self, display = False):

        mask = Image.open(pathlib.Path(__file__).resolve().parent + '/mask_fundus.bmp')
        mask = mask.resize((self.W, self.W), Image.ANTIALIAS) # resize to target image size
        mask = np.asarray(mask)

        if display:
            plt.imshow(mask)
            plt.show()
            print("mask shape: ", mask.shape)

        self.MASK = mask

    def continue_train(self,
    model_subtype = "EfficientNetB1", 
    use_class_weights = False,
    batch = 8, epochs = [10,0], optimizer = "adam"):

        self.load_best_model()
        return self.train(
            model_subtype, 
            use_class_weights,
            batch, 
            epochs, 
            optimizer, 
            True)

    def train(self,
    model_subtype = "EfficientNetB1", 
    use_class_weights = False,
    batch = 8, epochs = [10,0], optimizer = "adam", use_existing_model = False):
        '''
        If epochs[1] = 0, perform one-stage training. 
        If epochs[1] > 0, perform two-stage training (i.e., unfreeze last N fc layers & unfreeze all layers)
        '''

        ###### PRETRAINED WEIGHTS ##########

        if use_existing_model: # continue training use the last best model
            custom_model = self.get_best_model()
        else:
            image_input = Input(shape=(self.W, self.W, 3))
            num_classes = self.y_train.shape[-1]

            f_createmodel = getattr(efficientnet, model_subtype)
            model = f_createmodel(input_tensor=image_input, 
            include_top=True,weights='imagenet')         
            
            last_layer = model.layers[-2].output # model.get_layer('global_average_pooling2d_3').output
            #x= Flatten(name='flatten')(last_layer)
            out = Dense(num_classes, activation='softmax', name='output')(last_layer)
            custom_model = Model(image_input, out)


        ####### STAGE I - Retrain the last N fc layers #########

        for layer in custom_model.layers[:-6]:
            layer.trainable = False
        assert(custom_model.layers[3].trainable == False)

        custom_model.compile(loss='categorical_crossentropy',
        optimizer=optimizer,metrics=['accuracy']) # rmsprop

        ####### TRAIN MODEL #########
        
        d_class_weights = None

        if use_class_weights:

            y_code = self.y_train.argmax(1).tolist()
            class_weights = class_weight.compute_class_weight('balanced',
                                                        classes = np.unique(y_code),
                                                        y = y_code)
            d_class_weights = dict(enumerate(class_weights))
            print('using class weights: ' + str(d_class_weights))

        t=time.time()

        earlyStopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(self.MODEL_NAME + '_best.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


        hist = custom_model.fit(self.X_train, self.y_train, 
                                batch_size=batch, 
                                class_weight = d_class_weights,
                                epochs=epochs[0], 
                                verbose=1, 
                                validation_data=(self.X_val, self.y_val),
                                callbacks=[earlyStopping, mcp_save]) # batch_size 32 epochs 12
        print('Training time: %s' % (t - time.time()))
        (loss, accuracy) = custom_model.evaluate(self.X_val, self.y_val, batch_size=16, 
        verbose=1) # batch_size 10

        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
        
        del custom_model
        K.clear_session()
        
        if epochs[1] > 0: # we cannot benefit from Phase II training. Retraining all layers causes acc drops to  around 0.56.
            
            ########## STAGE II - train all layers #############
            t=time.time()

            custom_model = load_model(self.MODEL_NAME + '_best.hdf5')    

            # use a default model name for Phase II to avoid overwriting Phase I
            mcp_save2 = ModelCheckpoint(self.MODEL_NAME + '_best2.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        
            for layer in custom_model.layers: # unlock all layers
                layer.trainable = True

            custom_model.compile(loss = 'categorical_crossentropy',
                                        optimizer = optimizer, # rmsprop
                                        metrics = ['accuracy'])


            hist2 = custom_model.fit(self.X_train, self.y_train, 
                                    class_weight=d_class_weights,
                                    batch_size= batch, #8, 
                                    epochs=epochs[1], 
                                    verbose=1,
                                    callbacks = [earlyStopping, mcp_save2],
                                    validation_data=(self.X_val, self.y_val)) # batch_size 32 epochs 12

            print('Phase II Training time: %s' % (t - time.time()))

            # (loss, accuracy) = custom_vgg_model.evaluate(X_val, y_val, batch_size=16, verbose=1) # batch_size 10
            # print("[INFO] val loss={:.4f}, val accuracy: {:.4f}%".format(loss,accuracy * 100))

            # return custom_vgg_model, hist
            
            del custom_model
            K.clear_session()

            print('-----------------')
            print('There are two training stages. You need to judge which one is the best by checking the training curve (plot_two_stage_hist()).')
            print('-----------------')

            return hist.history, hist2.history

        return hist.history


    def get_best_model(self):
        '''
        If self.MODEL_NAME + "_best.hdf5" exists in local folder, this will directly load the model without training.
        '''

        if (not self.BEST_MODEL):
            self.BEST_MODEL = load_model(self.MODEL_NAME + "_best.hdf5", 
            compile = True) # 使用自定义的loss或者metric时，应compile = False，并手动调用compile
        return self.BEST_MODEL

    def plot_best_model(self):
        if (self.BEST_MODEL):
            save_path = self.MODEL_NAME + '.png'
            plot_model(self.BEST_MODEL, show_shapes=True, 
            show_layer_names=True, to_file = save_path)
            SVG(model_to_dot(self.BEST_MODEL, show_shapes=True, 
            show_layer_names=True).create(prog='dot', format='svg'))
            print('The model architecture plot has been saved to ' + save_path)

    def convert_to_tflite(self, path = None, v1 = False):

        best_model_path = self.MODEL_NAME + "_best.hdf5"

        # Save the model.
        if (path == None):
            fname = self.MODEL_NAME + '.tflite'
        else:
            fname = path

        convert_keras_to_tflite(best_model_path, fname, v1) 

    def evaluate(self, N = 30):
        custom_model = self.get_best_model()

        N = min(N, len(self.X_val))

        fig = plt.figure(figsize=(20, 20*(N/10+1)))
        for i in range(N):    
            p = custom_model.predict(np.expand_dims(self.X_val[i], axis=0))
            ax = fig.add_subplot(int(N/3)+1, 3, i+1)
            x = restore_image(self.X_val[i])
            ax.imshow(x)
            # ax.imshow((X_val[i] - X_val[i].min())/(X_val[i].max()-X_val[i].min()))
            title = 'Predict: {}\n {} \n Actual: {}'.format(self.class_names[int(p.argmax(-1))], np.round(p[0],2), 
            self.class_names[int(self.y_val[i].argmax(-1))])
            ax.set_title(title)


    def predict_file(self, img_path, display = True, use_mask = False):

        img = image.load_img(img_path, target_size=(self.W, self.W))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if (use_mask):
            self.load_mask()
            kill_border(x, self.MASK)

        x = efficientnet.preprocess_input(x)

        p = self.get_best_model().predict(x)
        title = 'Probs: {}.\nPredict: {}'.format(np.round(p,2), 
        self.class_names[int(p.argmax(-1))])
        
        if display:
            plt.figure()
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            plt.show()

        return p

    def binary_cls_eval(self):
        '''
        Output confusion matrix, ROC, P-R curve, AUC, etc.
        Only available when this is a binary classification problem.
        '''

        N = len(self.X_val)

        fig = plt.figure(figsize=(20, 12*(N/10+1)))

        thresh = 0.5
        y_pred = []
        y_true = []
        probs = []    
        
        for i in range(N):
        
            # For test set, we already preprocessing by the mask. 
            p = self.get_best_model().predict(np.expand_dims(self.X_val[i], axis=0))
            ax = fig.add_subplot(N/3+1, 3, i+1)
        
            probs.append(1-p[0]) # treat C1 as one class, all others as the second
            y_pred.append(int(p.argmax(-1)))
            y_true.append(int(self.y_val[i].argmax(-1)))
        
            # ax.imshow((X_test[i] - X_test[i].min())/(X_test[i].max()-X_test[i].min()),aspect = 0.75)    
            img = restore_image( self.X_val[i] )
            ax.imshow(img)

            title = 'Predict: {}, Actual: {}\nProb: {}'.format(self.class_names[int(p.argmax(-1))], 
                                                            self.class_names[int(self.y_val[i].argmax(-1))], 
                                                            np.round(p,3))
            ax.set_title(title)
            ax.axis('off')

        plt.show()
        

        print('====================')

        matplotlib.rc("font", family = 'Microsoft Yahei')
        
        ### TEST FOLDERS ###
        
        # y_pred, y_true, probs = binary_test(dir1, dir2, thresh = thresh, labels = labels, display = display, copy = copy)
        
        ### DRAW CONFUSION MATRIX ### 
        
        sns.set()
        f,ax=plt.subplots()
        # y_true = [0,0,1,2,1,2,0,2,2,0,1,1]
        # y_pred = [1,0,1,2,1,0,0,2,2,0,1,1]
        C2= confusion_matrix(y_true, y_pred) # labels=labels
        print(C2) #打印出来看看
        sns.heatmap(C2, annot = True,ax=ax) #画热力图

        #### For multiclass, convert it to binary #######

        if (len(self.class_names) > 2):
            y_true = list(map(multiclass_to_binary, y_true))
            y_pred = list(map(multiclass_to_binary, y_pred))

        ax.set_title('confusion matrix') #标题
        ax.set_xlabel('predict') #x轴
        ax.set_ylabel('true') #y轴
        plt.show()
        
        ### DRAW ROC Curve ###
        
        plt.figure()
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        plot_roc_curve(fpr, tpr)

        print('ROC AUC = ', roc_auc_score(y_true, probs)) # ROC AUC
        
        ### DRAW P-R Curve ###
        
        from sklearn.metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(y_true, probs)

        plt.figure("P-R Curve")
        plt.title('Precision/Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall,precision)
        # plt.plot([0, 1], [1, 0], color='darkblue', linestyle='--')
        plt.show()
        
        print('P-R AUC = {}'.format(auc(recall, precision)))
        
        ### THRESHOLDS ###
        
        print('阈值\t 查准率\t 查全率')

        for k,_ in enumerate(thresholds):
            print(round(thresholds[k], 2),
                '\t', 
                round(precision[k], 2),
                '\t', 
                round(recall[k], 2))

        return y_pred, y_true, probs


    # move images from src to dest subdirs by prediction results
    def batch_classification(self, src_dir, target_dir, use_mask = False):
        if use_mask:
            self.load_mask()
        
        for l in self.class_names:
            os.makedirs(os.path.join(target_dir, l), exist_ok=True) # create the target sub folders if not exist

        for root, dirs, files in tqdm(os.walk(src_dir)):      
            for file in files:
                if file.endswith('.jpg') or file.endswith('.JPG') or file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    try: 
                        img = image.load_img(img_path, 
                        target_size=(self.W, self.W))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)

                        if (use_mask):
                            kill_border(x, self.MASK)

                        x = efficientnet.preprocess_input(x)
                        p = self.get_best_model().predict(x)                
                        cid = int(p.argmax(-1))
                        cname = self.class_names[cid]

                        shutil.move(img_path, os.path.join(target_dir, cname, file))
                    except Exception as err: 
                        print(err)



def convert_keras_to_tflite(source, target, v1 = False):

    if v1:
        converter = tensorflow.compat.v1.lite.TFLiteConverter.from_keras_model_file(source)
    else:
        converter = tensorflow.lite.TFLiteConverter.from_keras_model(source)
        # converter.experimental_new_converter = True
    
    tflite_model = converter.convert()

    with open(target, 'wb') as f:
        f.write(tflite_model)

    print('tflite model saved to: ' + target)


def plot_history(hist):

    # visualizing losses and accuracy
    train_loss=hist['loss']# + hist2['loss']
    val_loss=hist['val_loss']# + hist2['val_loss']
    train_acc=hist['accuracy']# + hist2['accuracy']
    val_acc=hist['val_accuracy']# + hist2['val_accuracy']
    xc=range(1, len(train_loss) + 1) # epochs #  + epochs[1]

    # plt.style.use(['classic'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.show()

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.show()


def plot_two_stage_hist(hist1, hist2, epochs):
    
    # visualizing losses and accuracy
    train_loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    train_acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    xc=range(1, epochs + 1) # epochs

    # plt.style.use(['classic'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.show()

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'])
    plt.show()

# add back the subtracted ImageNet mean
def restore_image(x):
    '''
    Unlike vgg models, efficient net doesn't nothign in preprocessing.
    '''
    assert len(x.shape) == 3
    img = x.copy().astype(np.uint8)
    img[:,:,0] = (img[:,:,0]).astype(np.uint8) #  + 103.939
    img[:,:,1] = (img[:,:,1]).astype(np.uint8) #  + 116.779
    img[:,:,2] = (img[:,:,2]).astype(np.uint8) #  + 123.68
    return img


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def multiclass_to_binary(c, t = 0):
    '''
    t: target class id
    '''
    if (c == t):
        return 1
    else:
        return 0


######### BELOW ARE FUNDUS IMAGE SPECIFIC FUNCTIONS ############

def kill_border(data, mask):
    '''
    使用 mask 去除周边区域（消除标注文字等影响）
    function to set to black everything outside the FOV, in a full image
    '''

    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[3]==1 or data.shape[3]==3)  #check the channel is 1 or 3
    assert (data.shape[1] == mask.shape[0])
    assert (data.shape[2] == mask.shape[1])
    
    height = data.shape[2]
    width = data.shape[1]
    for i in range(data.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV(x,y,mask)==False:
                    data[i,x,y,:]=0.0


def inside_FOV(x, y, mask):
    if (mask[x,y] > 0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False
