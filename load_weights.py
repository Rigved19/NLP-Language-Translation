import pickle
#load weights


    def load_weights(model):
      with open('weightRNN.dat','rb') as f:
          weights = pickle.load(f)
      return weights