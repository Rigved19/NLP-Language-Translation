
import load_weights
import matplotlib.pyplot as plt
from layers import LSTM
import pickle
import numpy as np

def saveweight(model):
    #save the weights 
      params= model
      with open('weightRNN.dat','wb') as f:
          pickle.dump(params,f)
      print('weights saved successfully')
      pass


    def fit(X, Y, model):
    #Training the model
      costs = []
      for epoch in range(model.epoch):
          #Encoder Hidden State is captured after running Encoder Model
          enc_hidden, enc_memory, enc_ft, enc_it, enc_ot, enc_cct = model.encoder_forward() 
          #Last Encoder hidden state and Memory cell is passed as context vector
          dec_hidden, dec_memory, dec_output, dec_ft, dec_it, dec_ot, dec_cct = model.decoder_forward(enc_hidden[model.length_enc-1], enc_memory[model.length_enc-1] ) 
          #Compute Loss
          loss = model.loss(dec_output, Y)   
          #Print Loss
          if (epoch%5 ==0):
            costs.append(loss)
            print('loss: ', loss)
          #Backpropagation through time     
          model.decoder_backward(dec_output, dec_hidden, dec_memory, Y , dec_ft, dec_it, dec_ot, dec_cct)  

          model.encoder_backward(enc_hidden , enc_memory, enc_ft, enc_it, enc_ot, enc_cct)
          #Gradient Descent = updation of parameters
          model.update_params()

      #Plot the cost  
      plt.plot(np.squeeze(costs))
      plt.ylabel('Cost')
      plt.xlabel('Iterations (per 5)')
      plt.title("Trainingloss")
      plt.show()
      #Save Weights After Training
      model.saveweight()
      pass

    #Sampling Function to Predict Traanslation   
    def decoder_sampling (self, enc_hidden, enc_memory, idx_sos, idx_eos ) :
        
        def sigmoid(x):
          return 1 / (1 + np.exp(-x))

        def softmax(x):
          e_x = np.exp(x - np.max(x))
          return e_x / e_x.sum(axis=0)

        #Forward prop in Decoder
        dec_input = {}
        dec_hidden = {}
        dec_memory = {}

        y_pred = {}
        dec_output = {}

        indices = []
        
        a_next = enc_hidden
        c_next = enc_memory
        dec_hidden[-1] = enc_hidden
        dec_memory[-1] = enc_memory
        xt = np.zeros((self.ger_vocab , 1))      #Setting intial Input as <sos> token vector
        xt[idx_sos-1,0] = 1
        dec_input[-1] = xt

        idx = -1                                 #Initialize Index value

        t=0                                      #Counter

        while (idx+1 != idx_eos and t != 8) :    #Sample Until <eos> is reached or 8 words are sampled
          concat = np.concatenate((a_next , xt))
          ft = sigmoid(self.wf_dec @ concat + self.bf_dec )
          it = sigmoid(self.wi_dec @ concat + self.bi_dec )
          ot = sigmoid(self.wo_dec @ concat + self.bo_dec )
          cct = np.tanh(self.wc_dec @ concat + self.bc_dec )

          c_next = ((ft*c_next) + (it*cct))
          a_next = ot*(np.tanh(c_next))

          y_pred = softmax(self.wy_dec @a_next + self.by_dec)

          output = np.zeros((self.ger_vocab , 1)) 
          np.random.seed(t)
          idx = np.random.choice(list(range(self.ger_vocab)), p=y_pred[:,0].ravel())

          indices.append(idx+1)
          output[idx,0] = 1
          xt = output
          dec_input[t] = xt
          dec_output[t] = output

          t += 1

        return indices                            #Return Output Indices

    def predict(idx_sos, idx_eos, embed , eng_length, dummy):

      model = dummy.load_weights()  #load the weights

      model.embed_enc = embed
      model.length_enc = eng_length
      enc_hidden, enc_memory, enc_ft, enc_it, enc_ot, enc_cct = model.encoder_forward()

      indices = model.decoder_sampling(enc_hidden[model.length_enc-1], enc_memory[model.length_enc-1], idx_sos, idx_eos)

      return indices      