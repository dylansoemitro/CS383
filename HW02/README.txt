Implementing TF-IDF
1. Soviet, Mao, Krushchev, Union, party, leader etc. Seems to be obituaries about communist leaders
2. The words with the highest tf-idf are she, her , how, working, british, think, sense. These words are relevant to the topic of obituaries, but do not seem indicative of hers, besides gender, because they are just general (stop)words and not specific. 

1. We can be sure the code has been implemented correctly. It also saves time development, and we can use other functions within the library seamlessly.
2. Cell 4: We initialize a TfidfVectorizer object to convert documents from a list of strings to tf-idf scores. We call the fit_transform function to convert the list of strings to a sparse matrix.
Cell 5: We convert the sparse matrices to an array. We print the length (compare to number of documents).
Cell 6: import pandas, make output folder if it doesn't exist
Cell 7: Create list of output filenames, which are csvs corresponding to input. Loops through each vector of tf-idf scores, and inserts score pair into dataframe. 
3. The score in our implentation is normalized between 0 and 1, whereas the sklearn implementation is not.
4. Use a log function. 

Feedback:
1. 3 hours.
2. Intricacies of tf-idf, and how it translates to real-world data.
3. Document classification with tf-idf, for example college theses in different fields, etc.