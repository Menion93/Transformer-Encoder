import tensorflow as tf

from .dense import Dense
from .multi_head_attention import MultiHeadAttention

class TransformerLayer:
    
    def __init__(self, heads, hidden_inp, dense_hidden, trainable=True):
        self.trainable = trainable
        self.heads = heads
        self.hidden_inp = hidden_inp
        self.hidden_output = hidden_inp // heads
        self.dense_hidden = dense_hidden
        self.build()
    
    def build(self):
        self.mha = MultiHeadAttention(self.heads, self.hidden_inp)
        self.dense1 = Dense(self.hidden_inp, self.dense_hidden, dropout=0.7)
        self.dense2 = Dense(self.dense_hidden, self.hidden_inp, dropout=0.7)
        self._trainable_variables = self.mha.get_trainable_weights() +    \
                                    self.dense1.get_trainable_weights() + \
                                    self.dense2.get_trainable_weights()
        
    #@tf.contrib.eager.defun
    def forward(self, Q, K, V):
        output_ma = self.mha.forward(Q, K, V)
        o_ma_norm = tf.contrib.layers.layer_norm(output_ma + K)
        o_dense1 = tf.nn.relu(self.dense1.forward(o_ma_norm))
        o_dense2 = self.dense2.forward(o_dense1)
        output = tf.contrib.layers.layer_norm(o_dense2 + o_ma_norm)
        return output
        
    def get_trainable_weights(self):
        return self._trainable_variables if self.trainable else False