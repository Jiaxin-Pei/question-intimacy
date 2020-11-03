# Question-Intimacy

## Intro
question-intimacy is a package used to estimate the intimacy of questions. It is released with
EMNLP 2020 paper `Quantifying Intimacy in Language`. 


## Install 

### Use pip
If `pip` is installed, question-intimacy could be installed directly from it:

    pip3 install question-intimacy

### Dependencies
	python>=3.6.0
	torch>=1.6.0
	transformers >= 3.1.0
	numpy
	math
	tqdm
	
	
## Usage and Example

### Notes: During your first usage, the package will download a model file automatically, which is about 500MB.

### `Construct the Predictor Object`
	>>> from question_intimacy.predict_intimacy import IntimacyEstimator
	>>> inti = IntimacyEstimator()
Cuda is disabled by default, to allow GPU calculation, please use

	>>> from question_intimacy.predict_intimacy import IntimacyEstimator
	>>> inti = IntimacyEstimator(cuda=True)

### `predict`
`predict` is the core method of this package, 
which takes a single text of a list of texts, and returns a list of raw values in `[-1,1]` (higher means more intimate, while lower means less).

	# Predict intimacy for one question
	>>> text = 'What is this movie ?''
	>>> inti.predict(text,type='list')
	-0.2737383
	
	# Predict intimacy for a list of questions (less than a batch)
	>>> text = ['What is this movie ?','Why do you hate me ?']
	>>> inti.predict(text,type='list')
	[-0.2737383, 0.3481976]
	
	# Predict intimacy for a long list of questions
	>>> text = [a long list of questions]
	>>> inti.predict(text,type='long_list',tqdm=tqdm)
    [-0.2737383, 0.3481976, ... ,-0.2737383, 0.3481976]



## Contact
Jiaxin Pei (pedropei@umich.edu)