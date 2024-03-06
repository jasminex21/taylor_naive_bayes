# taylor_naive_bayes

The aim of this project is essentially to train a Naive Bayes model that can play my Taylor Swift lyrics guessing game - as in, it can be fed a lyric from any Taylor Swift song, and guess from which song the lyric comes from. The model will be trained on a dataset of all lyrics from Taylor's (album) discography (as of March 2024).

Naive Bayes classifiers are a type of *probabilistic* classifers based on Bayes Theorem. As in, once trained, they can predict the probability of a given sequence of words being of a certain class (the song in this case), based on information learned during training ($P(c|W)$). This specific information includes the **likelihood** $P(W|c)$, which is the probability that the lyrics come up given the song, and also the **prior** $P(c)$, the probability of the song coming up itself.

Ultimately, what the model does is to compute the **posterior** $P(c|W)$ for *every class (song)*. The song whose posterior is the greatest is then the song that the lyric most likely comes from. Given the probabilistic nature of Naive Bayes models, it would be possible for the model to output a list of the songs that the lyric is most likely to come from, not just its overall prediction. 
