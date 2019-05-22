# -*- coding: utf-8 -*-

from tensorflow.keras import models, layers


class FCBody(layers.Layer):
  def __init__(self,h1=64,h2=32, name='FCBody', **kwargs):
    super(FCBody, self).__init__(name=name, **kwargs)
    self.h1 = h1
    self.h2 = h2
    return

  def build(self, input_shape):
    self.hidden1 = layers.Dense(units=self.h1, name='FC_dense1')
    self.hidden1.build(input_shape)
    self.relu1 = layers.Activation('relu', name='FC_relu1')
    self.hidden2 = layers.Dense(self.h2, name='FC_dense2')
    self.hidden2.build((input_shape[0], self.h1))
    self.relu2 = layers.Activation('relu', name='FC_relu2')
    self._trainable_weights =  self.hidden1.trainable_weights + self.hidden2.trainable_weights
    super(FCBody, self).build(input_shape)

  
  def call(self, inputs):
    tf_x = self.hidden1(inputs)
    tf_x = self.relu1(tf_x)
    tf_x = self.hidden2(tf_x)
    tf_x = self.relu2(tf_x)
    return tf_x
    

class MLPActor(models.Model):
  def __init__(self, input_size=24, output_size=4, name='actor', 
               out_activation='tanh', loss='mse',
               metrics=['acc']):
    super(MLPActor, self).__init__(name='mlp')
    self.output_size = output_size
    self.input_size = input_size
    self.fcbody = FCBody(name='actor_fc_body')
    self.output_layer = layers.Dense(self.output_size, name='actor_output')    
    self.output_act = layers.Activation(out_activation, 
                                        name='actor_{}'.format(out_activation))
    self.compile(loss=loss, optimizer='adam',
                  metrics=metrics)
    self.build((None,input_size))
    
    
  def call(self, inputs):
    tf_x = inputs
    tf_x = self.fcbody(tf_x)
    tf_x = self.output_layer(tf_x)
    tf_x = self.output_act(tf_x)
    self.actor_outputs = tf_x
    return self.actor_outputs
  
  
  def soft_copy_from_model(self, model, tau):
    this_weights = self.get_weights()
    src_weights = model.get_weights()
    final_weights = [this_weights[i] * (1-tau) + 
                    src_weights[i] * tau 
                    for i in range(len(this_weights))]
    self.set_weights(final_weights)
    return
    
  def soft_copy_from_weights(self, src_weights, tau):
    this_weights = self.get_weights()
    final_weights = [this_weights[i] * (1-tau) + 
                    src_weights[i] * tau 
                    for i in range(len(this_weights))]
    self.set_weights(final_weights)
    return
  
  
if __name__ == '__main__':
  
  from tensorflow.keras.datasets import mnist
  (x,y), (xt,yt) = mnist.load_data()
  x = x.reshape((-1,28*28)) / 255.
  xt = xt.reshape((-1,28*28)) / 255.
  y = y.reshape((-1,1))
  yt = yt.reshape((-1,1))
  input_size = 28*28
  output_size = 10
  m_1 = MLPActor(input_size, output_size, out_activation='softmax', 
                 loss='sparse_categorical_crossentropy')
  m_2 = MLPActor(input_size, output_size, out_activation='softmax', 
                 loss='sparse_categorical_crossentropy')

  print("M_2 untrained eval: loss={:.2f}, acc={:.3f}".format(*m_2.evaluate(xt,yt)))
  m_1.fit(x=x,y=y, epochs=1, validation_data=(xt,yt))  
  print("M_1   trained eval: loss={:.2f}, acc={:.3f}".format(*m_1.evaluate(xt,yt)))
  m_2.soft_copy_from_model(m_1, tau=0.5)  
  print("M_2 soft-copy eval: loss={:.2f}, acc={:.3f}".format(*m_2.evaluate(xt,yt)))
  m_2.soft_copy_from_model(m_1, tau=0.5)  
  print("M_2 soft-copy eval: loss={:.2f}, acc={:.3f}".format(*m_2.evaluate(xt,yt)))
  
  