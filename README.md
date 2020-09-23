# recommender-collaborative

Using k-NN because it is similar to cosine similarity thematically. The k-NN is fit using the MovieLens 100k dataset which was built into the Surprise library. A few dummy users are created for a demo but a csv with the user data could work just as well.  Just call the predict() with the user name and movie ID to get a rating with which you can determine whether to recommend or not.

Using item-based filtering because it is scalable since number of items grow slower than number of users per unit time. Use grid-search to find optimal parameters to train on.
