import numpy as np
import ref

def getPreds(hm):
  assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
  res = hm.shape[2]
  hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
  idx = np.argmax(hm, axis = 2)
  #print(idx)
  preds = np.zeros((hm.shape[0], hm.shape[1], 2))
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res
  
  return preds

def calcDists(preds, gt, normalize):
  dists = np.zeros((preds.shape[1], preds.shape[0]))
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
        dists[j][i] = ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
      else:
        dists[j][i] = -1
  return dists

def distAccuracy(dist, thr = 0.5):
  dist = dist[dist != -1]
  if len(dist) > 0:
    return 1.0 * (dist < thr).sum() / len(dist)
  else:
    return -1

def Accuracy(output, target):
  aid=[]
  preds = getPreds(output)
  gt = getPreds(target)
  dists = calcDists(preds, gt, np.ones(preds.shape[0]) * ref.outputRes / 10)
  acc = np.zeros(len(aid))
  avgAcc = 0
  badIdxCount = 0


  for i in range(len(aid)):
    acc[i] = distAccuracy(dists[aid[i]])
    if acc[i] >= 0:
      avgAcc = avgAcc + acc[i]
    else:
      badIdxCount = badIdxCount + 1
  
  if badIdxCount == len(aid):
    return 0
  else:
    return avgAcc / (len(aid) - badIdxCount)

def MPJPE(output2D, output3D, meta):
  meta = meta.numpy()
  p = np.zeros((output2D.shape[0], 5, 3))
  p[:, :, :2] = getPreds(output2D).copy()
  
  hm = output2D.reshape(output2D.shape[0], output2D.shape[1], ref.outputRes, ref.outputRes)
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      pX, pY = int(p[i, j, 0]), int(p[i, j, 1])
      scores = hm[i, j, pX, pY]
      if pX > 0 and pX < ref.outputRes - 1 and pY > 0 and pY < ref.outputRes - 1:
        diffY = hm[i, j, pX, pY + 1] - hm[i, j, pX, pY - 1]
        diffX = hm[i, j, pX + 1, pY] - hm[i, j, pX - 1, pY]
        p[i, j, 0] = p[i, j, 0] + 0.25 * (1 if diffX >=0 else -1)
        p[i, j, 1] = p[i, j, 1] + 0.25 * (1 if diffY >=0 else -1)
  p = p + 0.5
  
  p[:, :, 2] = (output3D.copy() + 1) / 2 * ref.outputRes
  h36mSumLen = 4296.99233013/4
  root =0
  err = 0
  num3D = 0
  for i in range(p.shape[0]):
    s = meta[i].sum()
    if not (s > - ref.eps and s < ref.eps):
      num3D += 1
      lenPred = 0
      for e in [[3, 1],[1, 4],[0, 1], [1, 2]]:
        lenPred += ((p[i, e[0]] - p[i, e[1]]) ** 2).sum() ** 0.5 
      pRoot = p[i, root].copy()
      for j in range(5):
        p[i, j] = (p[i, j] - pRoot) / lenPred * h36mSumLen + meta[i, root]
     # p[i, 7] = (p[i, 6] + p[i, 8]) / 2
      for j in range(5):
        dis = ((p[i, j] - meta[i, j]) ** 2).sum() ** 0.5
        err += dis / 5
  if num3D > 0:
    return err / num3D, num3D
  else:
    return 0, 0
    

  
  

