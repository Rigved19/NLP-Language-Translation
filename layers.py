class LSTM:
   

    def __init__(self, embed_enc, onehot_dec, eng_vocab, ger_vocab, epoch, lr, hid_dim, embed_size, timesteps_enc , timesteps_dec):
        self.embed_enc=embed_enc  #Embedded input
        self.onehot_dec=onehot_dec #Decoder OneHot Input
        self.epoch=epoch  #number of epoch
        self.lr=lr   #learning rate

        self.embed_size=embed_size   #(n_x)_encoder
        self.ger_vocab=ger_vocab     #(n_x)_decoder and n_y
        self.eng_vocab=eng_vocab
        self.length_enc = timesteps_enc   #Encoder Timesteps
        self.length_dec = timesteps_dec   #Decoder Timesteps
        self.hidden = hid_dim        #(n_a)
        
        #Parameter Initialization

        np.random.seed(1)
        #Encoder Paramters
        self.wf_enc = np.random.randn(hid_dim , hid_dim + embed_size)*0.01
        self.wi_enc = np.random.randn(hid_dim , hid_dim + embed_size)*0.01
        self.wo_enc = np.random.randn(hid_dim , hid_dim + embed_size)*0.01
        self.wc_enc = np.random.randn(hid_dim , hid_dim + embed_size)*0.01

        self.bf_enc = np.zeros((hid_dim , 1))
        self.bi_enc = np.zeros((hid_dim , 1))
        self.bo_enc = np.zeros((hid_dim , 1))
        self.bc_enc = np.zeros((hid_dim , 1))

        #Decoder Paramters
        self.wf_dec = np.random.randn(hid_dim , hid_dim + ger_vocab)*0.01
        self.wi_dec = np.random.randn(hid_dim , hid_dim + ger_vocab)*0.01
        self.wo_dec = np.random.randn(hid_dim , hid_dim + ger_vocab)*0.01
        self.wc_dec = np.random.randn(hid_dim , hid_dim + ger_vocab)*0.01
        self.bf_dec = np.zeros((hid_dim , 1))
        self.bi_dec = np.zeros((hid_dim , 1))
        self.bo_dec = np.zeros((hid_dim , 1))
        self.bc_dec = np.zeros((hid_dim , 1))

        self.wy_dec = np.random.randn(ger_vocab , hid_dim )*0.01
        self.by_dec = np.zeros((ger_vocab , 1))

        #Encoder Gradients 
        self.dwf_enc = np.zeros_like(self.wf_enc)
        self.dwi_enc = np.zeros_like(self.wi_enc)
        self.dwo_enc = np.zeros_like(self.wo_enc)
        self.dwc_enc = np.zeros_like(self.wc_enc)
        
        self.dbf_enc = np.zeros_like(self.bf_enc)
        self.dbi_enc = np.zeros_like(self.bi_enc)
        self.dbo_enc = np.zeros_like(self.bo_enc)
        self.dbc_enc = np.zeros_like(self.bc_enc)
     
        #Decoder Gradients
        self.dwf_dec = np.zeros_like(self.wf_dec)
        self.dwi_dec = np.zeros_like(self.wi_dec)
        self.dwo_dec = np.zeros_like(self.wo_dec)
        self.dwc_dec = np.zeros_like(self.wc_dec)

        self.dbf_dec = np.zeros_like(self.bf_dec)
        self.dbi_dec = np.zeros_like(self.bi_dec)
        self.dbo_dec = np.zeros_like(self.bo_dec)
        self.dbc_dec = np.zeros_like(self.bc_dec)

        self.dwy_dec = np.zeros_like(self.wy_dec)
        self.dby_dec = np.zeros_like(self.by_dec)

        self.daa_dec=np.zeros((hid_dim, embed_enc.shape[1]))   ##
        self.daa_enc=np.zeros_like(self.daa_dec)              ##
        self.dcc_dec=np.zeros((hid_dim, embed_enc.shape[1]))   ##
        self.dcc_enc=np.zeros_like(self.dcc_dec)               ##
    

    def encoder_forward(self):

      def sigmoid(x):
        return 1 / (1 + np.exp(-x))

      #Forward prop in Encoder
      enc_hidden={}
      enc_memory = {}
      enc_ft = {}
      enc_it = {}
      enc_ot = {}
      enc_cct = {}

      a_next = np.zeros((self.wf_enc.shape[0], self.embed_enc.shape[1]))
      c_next = np.zeros((self.wf_enc.shape[0], self.embed_enc.shape[1]))
      enc_hidden[-1] = a_next
      enc_memory[-1] = c_next

      for t in range(self.length_enc):

        xt = self.embed_enc[:,:,t]
        concat = np.concatenate((a_next , xt))
        ft = sigmoid( self.wf_enc @ concat + self.bf_enc )
        it = sigmoid( self.wi_enc @ concat + self.bi_enc )
        ot = sigmoid( self.wo_enc @ concat + self.bo_enc)
        cct = np.tanh(self.wc_enc @ concat + self.bc_enc )

        enc_ft[t] = ft
        enc_it[t] = it 
        enc_ot[t] = ot
        enc_cct[t] = cct

        c_next = ((ft*c_next) + (it*cct))
        a_next = ot*(np.tanh(c_next))

        enc_hidden[t] = a_next
        enc_memory[t] = c_next


      return enc_hidden , enc_memory ,  enc_ft, enc_it, enc_ot, enc_cct

    def decoder_forward(self, enc_hidden, enc_memory):

        def sigmoid(x):
          return 1 / (1 + np.exp(-x))

        def softmax(x):
          e_x = np.exp(x - np.max(x))
          return e_x / e_x.sum(axis=0)


        #Forward prop in Decoder
        #dec_output = []
        #dec_input = []
        dec_hidden = {}
        dec_memory = {}
        dec_output = {}
        dec_ft = {}
        dec_it = {}
        dec_ot = {}
        dec_cct = {}
        
        a_next = enc_hidden
        c_next = enc_memory
        dec_hidden[-1] = enc_hidden
        dec_memory[-1] = enc_memory

        for t in range(self.length_dec):

          xt = self.onehot_dec[:,:,t]
          concat = np.concatenate((a_next , xt))
          ft = sigmoid(self.wf_dec @ concat + self.bf_dec )
          it = sigmoid(self.wi_dec @ concat + self.bi_dec )
          ot = sigmoid(self.wo_dec @ concat + self.bo_dec )
          cct = np.tanh(self.wc_dec @ concat + self.bc_dec )
          dec_ft[t] = ft
          dec_it[t] = it 
          dec_ot[t] = ot
          dec_cct[t] = cct

          c_next = ((ft*c_next) + (it*cct))
          a_next = ot*(np.tanh(c_next))

          y_pred = softmax( self.wy_dec @ a_next + self.by_dec )

          dec_hidden[t] = a_next
          dec_memory[t] = c_next
          dec_output[t] = y_pred


        return dec_hidden , dec_memory , dec_output, dec_ft, dec_it, dec_ot, dec_cct


    def loss(self, pred, orig):                                                   
        #compute loss
        loss=0
        for t in range(orig.shape[2]):
            yt_pred = pred[t]              #Since pred is dictionary with each key being a timestep have value of y pred - array(n_y , m)
            for m in range(orig.shape[1]):
              loss+= -np.sum((orig[:,m,t]*np.log(yt_pred[:,m])))


        return loss/orig.shape[1]

    def decoder_backward(self, dec_out, dec_hidden, dec_memory, orig, dec_ft, dec_it, dec_ot, dec_cct):
        #Backpropagation through time

        for t in reversed(range(self.length_dec)): 
            ft = dec_ft[t]
            it = dec_it[t] 
            ot = dec_ot[t]
            cct = dec_cct[t]
            xt = self.onehot_dec[:,:,t]
            c_next = dec_memory[t]
            c_prev = dec_memory[t-1]
            a_next = dec_hidden[t]
            a_prev = dec_hidden[t-1]
            n_a = self.hidden
   
            dy = np.copy(dec_out[t])                          #Backpropogation through Softmax
            Y = orig[:,:,t]
            for m in range(orig.shape[1]):
                dy[:,m] -= Y[:,m]

            self.dwy_dec += np.dot(dy, dec_hidden[t].T)       #Backpropogation through DenseLayer
            self.dby_dec += np.sum(dy, axis=1, keepdims=True)

            da = np.dot(self.wy_dec.T, dy) + self.daa_dec     #Backprop addidtion from DenseLayer and Reccurentlayer
            dot = da*np.tanh(c_next)*ot*(1 - ot)
            dcct = (self.dcc_dec*it + da*ot*(1- (np.tanh(c_next)**2))*it)*(1-cct**2)
            dit = (self.dcc_dec*cct + da*ot*(1- (np.tanh(c_next)**2))*cct)*it*(1 - it)
            dft = (self.dcc_dec*c_prev + da*ot*(1- (np.tanh(c_next)**2))*c_prev)*ft*(1 - ft)

            self.dwf_dec += dft@(np.concatenate((a_prev , xt))).T
            self.dwi_dec += dit@(np.concatenate((a_prev , xt))).T
            self.dwc_dec += dcct@(np.concatenate((a_prev , xt))).T
            self.dwo_dec += dot@(np.concatenate((a_prev , xt))).T
            self.dbf_dec += np.sum(dft , axis = 1 , keepdims = True)
            self.dbi_dec += np.sum(dit , axis = 1 , keepdims = True)
            self.dbc_dec += np.sum(dcct , axis = 1 , keepdims = True)
            self.dbo_dec += np.sum(dot , axis = 1 , keepdims = True)

            self.daa_dec = self.wf_dec[:,:n_a].T@dft + self.wi_dec[:,:n_a].T@dit + self.wc_dec[:,:n_a].T@dcct + self.wo_dec[:,:n_a].T@dot
            self.dcc_dec = self.dcc_dec*ft + da*ot*(1-(np.tanh(c_next)**2))*ft
        pass

    def encoder_backward(self, enc_hidden, enc_memory, enc_ft, enc_it, enc_ot, enc_cct):
        #Backpropagation through time
        self.daa_enc = self.daa_dec  #Passing Decoder to Encoder (Hidden State)
        self.dcc_enc = self.dcc_dec  #Passing Decoder to Encoder (Cell Memory)
        for t in reversed(range(self.length_enc)): 
            ft = enc_ft[t]
            it = enc_it[t] 
            ot = enc_ot[t]
            cct = enc_cct[t]
            xt = self.embed_enc[:,:,t]
            c_next = enc_memory[t]
            c_prev = enc_memory[t-1]
            a_next = enc_hidden[t]
            a_prev = enc_hidden[t-1]
            n_a = self.hidden
   
            
            da = self.daa_dec                
            dc = self.dcc_dec
            dot = da*np.tanh(c_next)*ot*(1 - ot)
            dcct = (dc*it + da*ot*(1- (np.tanh(c_next)**2))*it)*(1-cct**2)
            dit = (dc*cct + da*ot*(1- (np.tanh(c_next)**2))*cct)*it*(1 - it)
            dft = (dc*c_prev + da*ot*(1- (np.tanh(c_next)**2))*c_prev)*ft*(1 - ft)

            self.dwf_enc += dft@(np.concatenate((a_prev , xt))).T
            self.dwi_enc += dit@(np.concatenate((a_prev , xt))).T
            self.dwc_enc += dcct@(np.concatenate((a_prev , xt))).T
            self.dwo_enc += dot@(np.concatenate((a_prev , xt))).T
            self.dbf_enc += np.sum(dft , axis = 1 , keepdims = True)
            self.dbi_enc += np.sum(dit , axis = 1 , keepdims = True)
            self.dbc_enc += np.sum(dcct , axis = 1 , keepdims = True)
            self.dbo_enc += np.sum(dot , axis = 1 , keepdims = True)

            self.daa_enc = self.wf_enc[:,:n_a].T@dft + self.wi_enc[:,:n_a].T@dit + self.wc_enc[:,:n_a].T@dcct + self.wo_enc[:,:n_a].T@dot
            self.dcc_enc = dc*ft + da*ot*(1-(np.tanh(c_next)**2))*ft
        pass

    def update_params(self):

        for d in [self.dwf_dec, self.dwi_dec, self.dwo_dec, self.dwc_dec, self.dwy_dec, self.dbf_dec, self.dbi_dec, self.dbo_dec, self.dbc_dec, self.dby_dec, self.dwf_enc, self.dwi_enc, self.dwo_enc, self.dwc_enc, self.dbf_enc, self.dbi_enc, self.dbo_enc, self.dbc_enc ]:
            np.clip(d, -1, 1, out=d)                  #Gradient clipping to avoid exploding gradients
        
        #Updating Decoder Parameters
        self.wf_dec -= self.lr*self.dwf_dec
        self.wi_dec -= self.lr*self.dwi_dec
        self.wo_dec -= self.lr*self.dwo_dec
        self.wc_dec -= self.lr*self.dwc_dec
        self.wy_dec -= self.lr*self.dwy_dec

        self.bf_dec -= self.lr*self.dbf_dec
        self.bi_dec -= self.lr*self.dbi_dec
        self.bo_dec -= self.lr*self.dbo_dec
        self.bc_dec -= self.lr*self.dbc_dec
        self.by_dec -= self.lr*self.dby_dec


        #Updating Encoder Parameter
        self.wf_enc -= self.lr*self.dwf_enc
        self.wi_enc -= self.lr*self.dwi_enc
        self.wo_enc -= self.lr*self.dwo_enc
        self.wc_enc -= self.lr*self.dwc_enc
      
        self.bf_enc -= self.lr*self.dbf_enc
        self.bi_enc -= self.lr*self.dbi_enc
        self.bo_enc -= self.lr*self.dbo_enc
        self.bc_enc -= self.lr*self.dbc_enc
    

