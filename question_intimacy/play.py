from question_intimacy.predict_intimacy import IntimacyEstimator
from tqdm import tqdm


inti = IntimacyEstimator()

text = ['What is this movie ?','why do you hate me ?','What is this movie ?','why do you hate me ?','What is this movie ?','why do you hate me ?']
print(inti.predict(text,type='long_list',tqdm=tqdm))