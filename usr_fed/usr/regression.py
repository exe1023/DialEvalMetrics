import pickle as pkl
import numpy as np

from sklearn import linear_model

regr = pkl.loads(open('regr.pkl', 'rb').read())

def predict_scores(metrics):
  n_metrics = []

  # Normalization values, calculated from USR paper.
  vals = [(-1.012293440649907,  0.39053753419230813),
          (-1.012293440649907,  0.39053753419230813),
          (0.8604811806918587,  0.30259338627812254),
          (0.8604811806918587,  0.30259338627812254),
          (0.8472124995904354,  0.27987823011851076)]

  for m,(mean,std) in zip(metrics,vals):
    m = np.array(m)
    n_metrics.append((m - mean)/std)

  pred = np.stack(n_metrics, axis=1)
  p4 = regr.predict(pred)
  return p4.tolist()

def scores(mlm, drc, drf):
  return predict_scores([mlm, mlm, drc, drc, drf])


if __name__ == '__main__':
  score_names = ['undr/mlm_roberta', 'undr/mlm_roberta', 'both/dr', 'both/dr', 'fct/dr']
  score_set = []
  for n in score_names:
    score_set.append(eval(open(n+".scores").read()))
  scores = predict_scores(score_set)
  open("regression.scores", "w+").write(str(scores))
