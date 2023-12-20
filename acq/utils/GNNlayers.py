"""
This script contains the classess of the layers that we have implemented for our model

"""

import tensorflow as tf


class AttGraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, outDim,activation=None,alphaLReLU=0.2,dropAtt=None, **kwargs):
        # constructor, which just calls super constructor
        # and turns requested activation into a callable function
        super(AttGraphConvLayer, self).__init__(**kwargs)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.leakyReLU = tf.keras.layers.LeakyReLU(alpha=alphaLReLU)
        if activation=='LReLU':
            self.activation = tf.keras.layers.LeakyReLU(0.2)
        else:
            self.activation = tf.keras.activations.get(activation)
        if dropAtt is not None:
            self.dropoutAtt = tf.keras.layers.Dropout(rate=dropAtt)
        else:
            self.dropoutAtt = None
        self.outDim = outDim #number of output features

    def build(self, input_shape):
        # create trainable weights dynamically i.e. senza sapere a priori la dimensione dell'input
        node_shape, Nmat_shape, Cmat_shape, mask_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], self.outDim),
                                 name="w",
                                 initializer="random_normal",
                                 trainable=True)



        self.attention = self.add_weight(
             shape=(2*self.outDim,1),
             initializer="random_normal",
             trainable=True)

    def call(self, inputs):
        # split input into nodes, adj
        nodes, Nmat, Cmat, mask = inputs
        Nnodes = nodes.shape[-2]
        #linear trasformation of features
        z = tf.matmul(nodes,self.w)
        if self.dropoutAtt is not None:
            z = self.dropoutAtt(z)
        z1 = tf.matmul(Cmat,z)
        z2 = tf.matmul(Nmat,z)
        couple_emb = tf.concat([z1,z2],axis=-1)
        attentionCoeff = self.leakyReLU(tf.matmul(couple_emb,self.attention))
        attentionCoeff = tf.transpose(attentionCoeff,perm=[0,2,1])
        Cmat_t = tf.transpose(Cmat,perm=[0,2,1])
        attentionCoeffM = tf.matmul((Cmat_t*attentionCoeff),Nmat)
        adj = tf.matmul(Cmat_t,Nmat)
        adj_norm = self.softmax(attentionCoeffM,mask=adj)
        #adj as a mask allows to compute softmax scores for each neighborood ignoring values to 0.
        # However, when there's a row with only 0s, they are not ignored and a softmax score is returned.
        # In order to neglect these constributes, an element wise mult with the original adj matrix is performed
        adj_norm = adj_norm*adj
        if self.dropoutAtt is not None:
            adj_norm = self.dropoutAtt(adj_norm)
        out = self.activation(tf.matmul(adj_norm,z))
        return out

class GlobalAttentionPooling(tf.keras.layers.Layer):
    def __init__(self,channels, **kwargs):
        super(GlobalAttentionPooling, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        node_shape, mask = input_shape
        self.features_layer = tf.keras.layers.Dense(
                            self.channels, name="features_layer")
        self.attention_layer = tf.keras.layers.Dense(
                            self.channels, activation="sigmoid", name="attn_layer")

    def call(self, inputs):
        # split input into nodes, adj
        nodes,mask= inputs

        att = self.attention_layer(nodes)
        linear = self.features_layer(nodes)

        #filtering

        att_f = tf.transpose( (tf.transpose(att, perm=[0, 2, 1]))*mask, perm=[0, 2, 1])
        linear_f = tf.transpose((tf.transpose(linear, perm=[0, 2, 1]))*mask, perm=[0, 2, 1])


        w_inputs = att_f *linear_f

        reduction = tf.reduce_sum(w_inputs,axis = 1, keepdims = True)
        return reduction



class MaskedSum(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaskedSum, self).__init__(**kwargs)

    def call(self,input):

        h1,h2,h3,h4,m1,m2,m3,m4 = input

        hm1 = tf.transpose((tf.transpose(h1,perm=[0,2,1]))*m1,perm=[0,2,1])
        hm2 = tf.transpose((tf.transpose(h2, perm=[0, 2, 1])) * m2, perm=[0, 2, 1])
        hm3 = tf.transpose((tf.transpose(h3, perm=[0, 2, 1])) * m3, perm=[0, 2, 1])
        hm4 = tf.transpose((tf.transpose(h4, perm=[0, 2, 1])) * m4, perm=[0, 2, 1])

        return hm1+hm2+hm3+hm4


def model_definition(NFEAT):
    ninput = tf.keras.Input((None, NFEAT))
    nminput1 = tf.keras.Input((None, None))
    nminput2 = tf.keras.Input((None, None))
    nminput3 = tf.keras.Input((None, None))
    nminput4 = tf.keras.Input((None, None))
    cminput1 = tf.keras.Input((None, None))
    cminput2 = tf.keras.Input((None, None))
    cminput3 = tf.keras.Input((None, None))
    cminput4 = tf.keras.Input((None, None))
    mkinput = tf.keras.Input((1, None))
    mkinput_1 = tf.keras.Input((1, None)) #one input mask for each bond type
    mkinput_2 = tf.keras.Input((1, None))
    mkinput_3 = tf.keras.Input((1, None))
    mkinput_4 = tf.keras.Input((1, None))
    nmall = tf.keras.Input((None, None))
    cmall = tf.keras.Input((None, None))

    # Attention Layer 1: 3 head for each bond type
    a11_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput1, cminput1, mkinput_1])
    a11_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput1, cminput1, mkinput_1])
    a11_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput1, cminput1, mkinput_1])

    a12_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput2, cminput2, mkinput_2])
    a12_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput2, cminput2, mkinput_2])
    a12_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput2, cminput2, mkinput_2])

    a13_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput3, cminput3, mkinput_3])
    a13_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput3, cminput3, mkinput_3])
    a13_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput3, cminput3, mkinput_3])

    a14_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput4, cminput4, mkinput_4])
    a14_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput4, cminput4, mkinput_4])
    a14_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput4, cminput4, mkinput_4])

    # Concatenate Node features for each boond type
    a11out = tf.keras.layers.Concatenate(axis=-1)([a11_1, a11_2, a11_3])
    a12out = tf.keras.layers.Concatenate(axis=-1)([a12_1, a12_2, a12_3])
    a13out = tf.keras.layers.Concatenate(axis=-1)([a13_1, a13_2, a13_3])
    a14out = tf.keras.layers.Concatenate(axis=-1)([a14_1, a14_2, a14_3])

    # Attention Layer 2: 3 head for each bond type
    a21_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a11out, nminput1, cminput1, mkinput_1])
    a21_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a11out, nminput1, cminput1, mkinput_1])
    a21_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a11out, nminput1, cminput1, mkinput_1])

    a22_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a12out, nminput2, cminput2, mkinput_2])
    a22_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a12out, nminput2, cminput2, mkinput_2])
    a22_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a12out, nminput2, cminput2, mkinput_2])

    a23_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a13out, nminput3, cminput3, mkinput_3])
    a23_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a13out, nminput3, cminput3, mkinput_3])
    a23_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a13out, nminput3, cminput3, mkinput_3])

    a24_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a14out, nminput4, cminput4, mkinput_4])
    a24_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a14out, nminput4, cminput4, mkinput_4])
    a24_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a14out, nminput4, cminput4, mkinput_4])

    # Concatenate Node features for each boond type
    a21out = tf.keras.layers.Average()([a21_1, a21_2, a21_3])
    a22out = tf.keras.layers.Average()([a22_1, a22_2, a22_3])
    a23out = tf.keras.layers.Average()([a23_1, a23_2, a23_3])
    a24out = tf.keras.layers.Average()([a24_1, a24_2, a24_3])

    masked_sum = MaskedSum()([a21out, a22out, a23out, a24out, mkinput_1, mkinput_2, mkinput_3, mkinput_4])

    h1 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])
    h2 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])
    h3 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])
    h4 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])

    pool_attention = tf.keras.layers.Concatenate(axis=-1)([h1, h2, h3,h4])
    pool_graph = GlobalAttentionPooling(512)([pool_attention, mkinput])
    x = tf.keras.layers.BatchNormalization(axis=-1)(pool_graph)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=(ninput,
                                   nminput1, cminput1,
                                   nminput2, cminput2,
                                   nminput3, cminput3,
                                   nminput4, cminput4,
                                   mkinput_1, mkinput_2,
                                   mkinput_3, mkinput_4,
                                   nmall,cmall,
                                   mkinput),
                           outputs=x)

    return model


def model_definition_PPB(NFEAT):
    ninput = tf.keras.Input((None, NFEAT))
    nminput1 = tf.keras.Input((None, None))
    nminput2 = tf.keras.Input((None, None))
    nminput3 = tf.keras.Input((None, None))
    nminput4 = tf.keras.Input((None, None))
    cminput1 = tf.keras.Input((None, None))
    cminput2 = tf.keras.Input((None, None))
    cminput3 = tf.keras.Input((None, None))
    cminput4 = tf.keras.Input((None, None))
    mkinput = tf.keras.Input((1, None))
    mkinput_1 = tf.keras.Input((1, None))  # one input mask for each bond type
    mkinput_2 = tf.keras.Input((1, None))
    mkinput_3 = tf.keras.Input((1, None))
    mkinput_4 = tf.keras.Input((1, None))
    nmall = tf.keras.Input((None, None))
    cmall = tf.keras.Input((None, None))

    # Attention Layer 1: 3 head for each bond type
    a11_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput1, cminput1, mkinput_1])
    a11_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput1, cminput1, mkinput_1])
    a11_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput1, cminput1, mkinput_1])

    a12_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput2, cminput2, mkinput_2])
    a12_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput2, cminput2, mkinput_2])
    a12_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput2, cminput2, mkinput_2])

    a13_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput3, cminput3, mkinput_3])
    a13_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput3, cminput3, mkinput_3])
    a13_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput3, cminput3, mkinput_3])

    a14_1 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput4, cminput4, mkinput_4])
    a14_2 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput4, cminput4, mkinput_4])
    a14_3 = AttGraphConvLayer(outDim=32, activation='LReLU', alphaLReLU=0.2)([ninput, nminput4, cminput4, mkinput_4])

    # Concatenate Node features for each boond type
    a11out = tf.keras.layers.Concatenate(axis=-1)([a11_1, a11_2, a11_3])
    a12out = tf.keras.layers.Concatenate(axis=-1)([a12_1, a12_2, a12_3])
    a13out = tf.keras.layers.Concatenate(axis=-1)([a13_1, a13_2, a13_3])
    a14out = tf.keras.layers.Concatenate(axis=-1)([a14_1, a14_2, a14_3])

    # Attention Layer 2: 3 head for each bond type
    a21_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a11out, nminput1, cminput1, mkinput_1])
    a21_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a11out, nminput1, cminput1, mkinput_1])
    a21_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a11out, nminput1, cminput1, mkinput_1])

    a22_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a12out, nminput2, cminput2, mkinput_2])
    a22_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a12out, nminput2, cminput2, mkinput_2])
    a22_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a12out, nminput2, cminput2, mkinput_2])

    a23_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a13out, nminput3, cminput3, mkinput_3])
    a23_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a13out, nminput3, cminput3, mkinput_3])
    a23_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a13out, nminput3, cminput3, mkinput_3])

    a24_1 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a14out, nminput4, cminput4, mkinput_4])
    a24_2 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a14out, nminput4, cminput4, mkinput_4])
    a24_3 = AttGraphConvLayer(outDim=64, activation='LReLU', alphaLReLU=0.2)([a14out, nminput4, cminput4, mkinput_4])

    # Concatenate Node features for each boond type
    a21out = tf.keras.layers.Average()([a21_1, a21_2, a21_3])
    a22out = tf.keras.layers.Average()([a22_1, a22_2, a22_3])
    a23out = tf.keras.layers.Average()([a23_1, a23_2, a23_3])
    a24out = tf.keras.layers.Average()([a24_1, a24_2, a24_3])

    masked_sum = MaskedSum()([a21out, a22out, a23out, a24out, mkinput_1, mkinput_2, mkinput_3, mkinput_4])

    h1 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])
    h2 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])
    h3 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])
    h4 = AttGraphConvLayer(outDim=128, activation='LReLU', alphaLReLU=0.2)([masked_sum, nmall, cmall, mkinput])

    pool_attention = tf.keras.layers.Concatenate(axis=-1)([h1, h2, h3,h4])
    pool_graph = GlobalAttentionPooling(512)([pool_attention, mkinput])
    x = tf.keras.layers.BatchNormalization(axis=-1)(pool_graph)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=(ninput,
                                   nminput1, cminput1,
                                   nminput2, cminput2,
                                   nminput3, cminput3,
                                   nminput4, cminput4,
                                   mkinput_1, mkinput_2,
                                   mkinput_3, mkinput_4,
                                   nmall, cmall,
                                   mkinput),
                           outputs=x)

    return model
