import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  tensorflow_addons as tfa
from keras import backend as K
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def get_sequence_tokens(sequence,pos):
    proteinseq_toks = {'L':4, 'A':5, 'G':6, 'V':7, 'S':8, 'E':9, 'R':10, 'T':11, 'I':12, 'D':13, 'P':14, 'K':15, 'Q':16, 'N':17, 'F':18, 'Y':19, 'M':20, 'H':21, 'W':22, 'C':23, 'X':24, 'B':25, 'U':26, 'Z':27, 'O':28,'_':0,'$':1,'#':2}
    length=10
    pos=pos-1
    if pos<=length:
        seq=sequence[:pos+length+1]
        seq=('_'*((length*2+1)-len(seq)))+'$'+seq+'#'
    elif pos>=len(sequence)-length-1:
        seq=sequence[pos-length:pos+length+1]
        seq='$'+seq+'#'+('_'*((length*2+1)-len(seq)))
    else:
        seq='$'+sequence[pos-length:pos+length+1]+'#'
    tokens=[proteinseq_toks[x] for x in seq]
    return tokens
def get_positions(seq):
    pos=[]
    for i,AA in enumerate(seq):
        if AA=='S':
            pos.append(i+1)
    return pos
def get_predictions(sequence):
    prediction_dic={}
    with tf.device('/cpu:0'):
        model=keras.models.load_model('model/Final',custom_objects={'f1_score':f1_score,'loss':tfa.losses.SigmoidFocalCrossEntropy()})
        for pos in get_positions(sequence):
            prediction_dic[pos]=model.predict([get_sequence_tokens(sequence,pos)])[0][0]
    return prediction_dic

