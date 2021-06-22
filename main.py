r = LSTM(Xenc_train, Xdec_train, vocab_eng_size, vocab_german_size, 400 , 0.0005 , 32, 50, max_eng_length , max_german_length)

LSTM.fit(Xenc_train, Ydec_train, r)