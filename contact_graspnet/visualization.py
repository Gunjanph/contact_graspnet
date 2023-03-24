from visualization_utils import visualize_grasps, show_image
import numpy as np
from pprint import pprint
# from numpy import load
# import matplotlib.pyplot as plt
# from PIL import Image

data = np.load('results/predictions_data.npz', allow_pickle=True)
# print(data.files)
pred = data['pred_grasps_cam']
cp = data['contact_pts']
go = data['gripper_openings']
gw = data['gripper_width']
score = data['scores']
# print(gw)
pred = pred.item()
cp = cp.item()
go = go.item()
gw = gw.item()
score = score.item()
print(score[-1])
num = np.where(score[-1]==np.max(score[-1]))[0]
print(np.where(score[-1]==np.max(score[-1])))
# print(type(pred))
# print(pred.keys())
# print(pred[-1].shape)
# print(pred[-1][0], cp[-1][0], go[-1][0])

for i in num:
    x = pred[-1][i]
    a = x[0:3, 2]
    b = x[0:3, 0]
    t = x[0:3, 3]
    w = gw[-1]
    c = cp[-1][i]
    ad = (t-c-(b)*w/2)
    d = (t-c-(b)*w/2)/(a)
    pprint({
        "r": x,
        "c": c 
    })
# pred = dict(enumerate(pred.flatten()))

# pred_grasps_cam = pred[1]
# print(pred_grasps_cam)

# score = dta['scores']


# # print(score)
# score = dict(enumerate(score.flatten(), 1))

# scores = score[1]
# print(scores)

# cp = data['contact_pts']

# # print(score)
# cp = dict(enumerate(cp.flatten(), 1))

# cp = cp[1]
# print(cp)

pc_full = np.load('../sim_grasping/src/pc/data.npy') 
# print(pc_full)

# pc_new = np.copy(pc_full)
# pc_full[:, 0] = pc_new[:, 0]
# pc_full[:, 1] = -pc_new[:, 1]
# pc_full[:, 2] = -pc_new[:, 2]

# show_image(rgb, segmap)
visualize_grasps(pc_full, pred, score, plot_opencv_cam=True)