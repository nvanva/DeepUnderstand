-Run the script evalAllClassifiers.py to prepare data for leaderboards and to save all classifiers’ files fro analysis if possible
-Then run textClassifiers.py (it has parameters -h and -p for setting host and port, by default 127.0.0.1:5000)
	+firstly start from leaderboards by this address
	+then go for analysis for each classifier from the table if possible
	+on classifier analyzer page it’s possible to go for feature analysis by class and 	for error analysis
	+from error analysis page you can go for particular document analysis with weights 	of ngrams (document analysis)
	+from document analysis and feature analysis you can go to the page for particular 	ngram to see, where it happened to appear

-If you want to add single classifier run evalClassifier.py script with —clf parameter, where you should specify the name of .json file for this classifier

The structure of json files for classifiers, datasets and evaluations (you can look through examples in ./classifiers, ./datasets and ./evaluations directories):
	+ ’module’ - module from which you want to take classifier or load load function 		for dataset or evaluation fucntion
	+ ‘class’ or ‘load_function’ or ‘eval_function’ - name of python class for 			classifier or load function for dataset or evaluation funciton
	+ ‘boardname’ - name for table representation
	+ ’params’ - parameters of classifier or load function
	+ ‘dataset’ or ‘evaluation_file’ - dataset to use for classifier cards and 			evaluation json file to use for dataset cards
	+ ‘description’ - more complete description 
	+ ‘trainset’ - for classifier it’s parameter that show which part of trainset to 		use for training
	+ ‘enabled’ - parameter to on\off classifier evaluation

The structure of dataset’s load function (look through examples ./20newsgroups.py, ./imdb.py, ./imdb.py):
	It should have have parameter ‘parts’, it should be ‘dev’ for test set and ‘train’ 	or value of ‘trainset’ from classifier json file for train set. Other parameters 	are given in dataset’s json file
	
	P.S.: for IMDB dataset you need to download it: 
		wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
		tar -xvf aclImdb_v1.tar.gz

The structure of evaluation function (look through example in ./evalAllClassifiers.py function ‘score_df’):
	It should take y_pred, X, y_true as arguments
	It should return pandas dataframe with metrics

The structure (interface) of classifier class (look through examples in ./nbsvm.py, ./TfidfMultinomialNB.py):
	
	For representation on leaderboards:
	+ fit(X_train, y_train) - this method fits the model as standard sklearn fit()
	+ predict(X_test) - predicts labels for X_test 
	
	For error and feature analysis:
	+ decision_function() or predict_proba() - also as sklearn methods
	+ getDocumentsInRow(X) - if necessary realize method to represent test data in 		array format
	+ save(path) - method for saving classifier to path
	+ load(path) - method for loading classifier from path
	+ getDocumentHTML(test_X, doc_id) - test_X is a test data and doc_id is an id of 	document to text in HTML. This method return what to text in HTML and second value 	shows whether it’s a dataframe or not
	+ getClasses() - if labels are numbers not names of classes you should realize 		this method to get classes in classifier’s order
	+ feature_importances(test_X, doc_id, classes, real, pred) - test_X is a test data 	and doc_id is an id of document in set, classes is list of all classes and real, 	pred - ids of real and predicted classes. This method should return dictionary 		with list of weights for each real and predicted class name key and list of ngram 	names.
	+ feature_analysis_top(classes, i, top=25) - method for feature analysis without 	document, it takes list of classes, index of class to analyze and number of top 	elements to display. Returns the same weights dictionary and ngram list as 		feature_importances
	+ getExamplesForNgram(X, y, ngram, maxExampleNum) (not necessary) - method that 	allows to show examples from X set with particular ngram, maxExampleNum limits the 	number of displayed examples. It returns dataframe with values_count (how many 		examples from each class), dataframe with examples and title of second dataframe 	(Totally x example or Sampled x / y examples).
