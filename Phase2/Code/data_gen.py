import numpy as np
import cv2
import os
import glob

# TRAIN_DIR = "../Data/Train/Train"
# NUM_ITERS=5

# print(os.path.join(TRAIN_DIR, "*.jpg"))
# img_paths = glob.glob(os.path.join(TRAIN_DIR, "*.jpg"))
# # print(img_paths)
# # img_paths = ["../Data/Train/1.jpg"]
# for img_path in img_paths:
#     img = cv2.imread(img_path)
    
#     for n in range(NUM_ITERS):
#         tlX = np.random.randint(0,img.shape[0])
#         tlY = np.random.randint(img.shape[1]//4,3*img.shape[1]//4)
#         patch_size = np.random.randint(50, min(51,img.shape[1]//4))

#         X = np.array([tlX, tlX, tlX+patch_size, tlX+patch_size])
#         Y = np.array([tlY, tlY+patch_size, tlY+patch_size, tlY])

#         kernel_size = 4
#         # X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

#         X_perturbed = np.random.randint(-kernel_size//2, kernel_size//2, 4)
#         Y_perturbed = np.random.randint(-kernel_size//2, kernel_size//2, 4)
        
#         X_ = X + X_perturbed
#         Y_ = Y + Y_perturbed

#         print(X, Y)

#         #get the patch
#         patch = img[X[0]:X[2], Y[0]:Y[1]]
#         #create random H matrix
#         print(X_,Y_)
#         arr1, arr2 = np.float32(np.column_stack((X,Y))),np.float32(np.column_stack((X_,Y_)))
#         print(arr1, arr2)
#         H = cv2.getPerspectiveTransform(arr1,arr2)
#         #transform the image with H
#         print(H)
#         img_warp = cv2.warpPerspective(img, (H), (img.shape[1],img.shape[0]))
#         #transform the patch to the original image again
#         patch_warp = img_warp[X[0]:X[2], Y[0]:Y[1]]
 
#         patch = cv2.resize(patch, (200,200))
#         patch_warp = cv2.resize(patch_warp, (200,200))

#         img = cv2.polylines(img,np.int32(arr1).reshape(-1,1,2),True,(0,0,255))
#         img = cv2.polylines(img,np.int32(arr2).reshape(-1,1,2),True,(0,255,0))
#         cv2.imshow("res",np.hstack([patch, patch_warp]))
#         cv2.imshow("res1",np.hstack([img, img_warp]))
        
#         cv2.waitKey()
#         #They showuld be the same

# # train_data = np.load("../Data/TrainData/training_data.npy")
# train_labels = np.load("../Data/TrainData/training_labels.npy")
# # train_labels_H = np.load("../Data/TrainData/training_labels_H.npy")

# # val_data = np.load("../Data/ValData/val_data.npy")
# # val_labels = np.load("../Data/ValData/val_labels.npy")
# # val_labels_H = np.load("../Data/ValData/val_labels_H.npy")

# # train_data = train_data.reshape((train_data.shape[0]*train_data.shape[1], train_data.shape[2], train_data.shape[3],train_data.shape[4],train_data.shape[5]))
# train_labels = train_labels.reshape((train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3],train_labels.shape[4]))

# # val_data = val_data.reshape((val_data.shape[0]*val_data.shape[1], val_data.shape[2], val_data.shape[3],val_data.shape[4],val_data.shape[5]))
# # val_labels = val_labels.reshape((val_labels.shape[0]*val_labels.shape[1], val_labels.shape[2], val_labels.shape[3],val_labels.shape[4]))

# # print(train_labels[0])
# print(train_labels.shape)
# train_labels = (train_labels[:,1]-train_labels[:,0]).reshape(-1,8)
# # print(train_labels[0])
# # print(val_data.shape,val_labels.shape, val_labels_H.shape)

# rand = np.random.rand(8)

# H = np.empty((1,8),np.float32)
# for i in range(len(train_labels)):
#     C_a, C_b_ = train_labels[i,:], train_labels[i,:]
#     # print(C_b_.shape, C_a.shape)

#     C_b_ = C_a + rand
#     A = []
#     b = []
#     for i in range(0,8,2): #since there are 4 corner pairs
#         Ai = [[0, 0, 0, -C_a[i], -C_a[i+1], -1, C_b_[i+1]*C_a[i], C_b_[i+1]*C_a[i+1]]]
#         Ai.append([C_a[i], C_a[i+1], 1, 0, 0, 0, -C_b_[i]*C_a[i], -C_b_[i]*C_a[i+1]])
#         A.extend(Ai)
    
#         bi = [-C_b_[i+1],-C_b_[i]]
#         b.extend(bi)    

#     A = np.array(A)
#     b = np.array(b)
#     # h = inv(A) dot b
#     h = np.linalg.inv(A) @ b
#     # print(h.shape)
#     H = np.append(H,h.reshape(1,-1), axis=0)
# print(H[1:,:].shape)



# A= A.squeeze(0)
# b=b.squeeze(0)

import torch

x = torch.ones((64,8))
y = torch.ones((64,))

out = torch.cat((x,torch.ones((x.size(dim=0),1))),dim=1)
out = out.reshape((out.size(dim=0),3,3))
print(out.size())
print(out[0])