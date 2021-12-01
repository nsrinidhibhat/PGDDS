# HMM-and-Viterbi

- Using a TreeBank universal tag set corpus to build a Viterbi PoS tagger.
- Later to improve the performance to solve for unknown words.

#### Steps 
- Initially we have a tuple of (word, POS), which we split up into train and test datasets.
- Then we seperate the words and POS from the sentences formed above, and generate a vocabulary for words
- We calculate Emission probability (P(w/t)) and Transition probability (P(t2/t1))
- VaniLLa Viterbi model is created and sovled for higher accuracy
- For the unknown words, lexicon and rule based POS tagging is done manually, and seen if it is implemented atop the vanilla Viterbi, improves model performance and accuracy or not!
