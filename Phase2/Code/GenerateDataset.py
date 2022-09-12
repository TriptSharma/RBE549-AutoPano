import cv2
import numpy as np
import glob

paths = sorted(glob.glob("../Data/Train/*.jpg"))

# out = np.zeros((len(paths), 5, 2))
SAMPLE_PER_IMAGE = 5
input_data = np.zeros((len(paths), SAMPLE_PER_IMAGE, 2, 128, 128, 3), dtype=np.uint8)
label_data_H = np.zeros((len(paths), SAMPLE_PER_IMAGE, 3, 3), dtype=np.float32)
label_data = np.zeros((len(paths), SAMPLE_PER_IMAGE, 2, 4, 2), dtype=np.float32)
print("Input Size", input_data.__sizeof__())
print("Output Size", label_data.__sizeof__())

i = 0
while i < len(paths):
    img = cv2.imread(paths[i])
    rows = []
    j = 0
    while j < SAMPLE_PER_IMAGE:
        patch_size = 128
        x = np.random.randint(patch_size//2, img.shape[1]-patch_size//2)
        y = np.random.randint(patch_size//2, img.shape[0]-patch_size//2)

        # print(x, y)

        corners = np.array([(x - patch_size // 2, y - patch_size // 2),
                            (x - patch_size // 2, y + patch_size // 2),
                            (x + patch_size // 2, y - patch_size // 2),
                            (x + patch_size // 2, y + patch_size // 2)])

        patch = img[corners[0][1]:corners[1][1], corners[0][0]:corners[3][0]]

        corners_warp = corners + np.random.normal(0, 10, (4, 2))

        warp_H = cv2.findHomography(corners, corners_warp)[0]

        # TODO: check on the out of bounds
        img_warp = cv2.warpPerspective(img, warp_H, img.shape[:2])

        patch_warp = img_warp[corners[0][1]:corners[1][1], corners[0][0]:corners[3][0]]

        img_corners = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
        img_corners_warp = cv2.perspectiveTransform(img_corners.reshape((-1, 1, 2)).astype(float), warp_H)
        inside = np.all([cv2.pointPolygonTest(img_corners_warp.astype(int), corner.astype(float), False) >= 0 for corner in corners])

        if not inside:
            # print("NOT INSIDE", [cv2.pointPolygonTest(img_corners_warp.astype(int), corner.astype(float), False) >= 0 for corner in corners])
            # cv2.imshow('patch_warp', patch_warp)
            # cv2.waitKey(0)
            continue

        if patch.shape != patch_warp.shape:
            # print("mismatch size")
            continue

        input_data[i, j, 0] = patch
        input_data[i, j, 1] = patch_warp
        label_data_H[i, j] = warp_H
        label_data[i, j, 0] = corners
        label_data[i, j, 1] = corners_warp
        pair = np.hstack((patch, patch_warp))
        rows.append(pair)
        j += 1

    compare = np.vstack(rows)
    cv2.imshow('warp', compare)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

    i += 1

np.save("training_data.npy", input_data)
np.save("training_labels.npy", label_data)
np.save("training_labels_H.npy", label_data_H)
print("Data saved!")
