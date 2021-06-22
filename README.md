# Natural Language Processing - Language Translation
Machine Translation Model which learns to translate English sentences into German Sentences. This is carried out using **seq2seq** Model ie **Encoder-Decoder** model with **LSTM Units**. 
#### Only ***Python Libraries*** are used to implement entire model for better understanding of the mathematics that goes behind working of Recurrent Neural Networks.

The dataset used can be found [here](http://www.manythings.org/anki/)

                       Given below is an example of **seq2seq** Model Architecture
![image](https://user-images.githubusercontent.com/63362412/122981439-60b6ca00-d3b7-11eb-8dc5-1bc2fb55747c.png)

# Technical Aspect

## The project is divided in 2 parts
### 1. Data Cleaning and Text Preprcossinhg
#### (i) Data Cleaning
 *  Adding additional tokens <sos> and <eos> for the German Sentences
 * Lowercasing
 * Removing Punctuations
 * Remove words with numbers
#### (ii) Text Preprcoessing
  * Creating Vocab Dictionaries for English and German Language (Word to Index, Index to Word)
  * Tokenization - Converting each input/output sentence into words based on info from the dictionaries
  * Padding - Converting each sentence to same length by pre and post padding
  * Creating seperate Word Embedding for Input
  * Creating 3D Tensors for English Input, German Input and German Output
  
 ### 2. Building Encoder-Decoder Model
  * Building model using LSTM Units
  * Following are the function being implemented :
    * Parameters Initialization
    * Encoder Forward Propogation
    * Decoder Forward Propogation
    * Loss - Cross Entropy Loss
    * Decoder Backward Propogation
    * Encoder Backward Propogation
    * Gradient Descent

