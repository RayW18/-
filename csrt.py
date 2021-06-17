import cv2
import sys
import os
import glob
import math

(major_version, minor_version, subminor_version) = cv2.__version__.split('.')

dir = 'trainval'
# data_path = os.path.join(img_dir, "*.jpg")
# img_files = glob.glob(data_path)
# gth_path = os.path.join(img_dir, 'groundtruth.txt')

if __name__ == '__main__':
    img_dirs = []
    for sub_dir in os.listdir(dir):
        img_dirs.append(sub_dir)

    for video in img_dirs:

        video_path = os.path.join(dir, video)
        pic_path = os.path.join(video_path, "*.jpg")
        img_files = glob.glob(pic_path)
        img_files.sort()
        gth_path = os.path.join(video_path, 'groundtruth.txt')

        txt_path = os.path.join("predict_val_csrt", video + ".txt")

        print(pic_path)
        frame = cv2.imread(img_files[0])

        if frame is None:
            print('cannot read image file')
            sys.exit()

        f = open(gth_path, 'r')
        fw = open(txt_path, "w")

        line = f.readline()
        cs = [float(t) for t in line.split(",")]
        x1 = cs[0]
        y1 = cs[1]
        x2 = cs[2]
        y2 = cs[3]
        x3 = cs[4]
        y3 = cs[5]
        x4 = cs[6]
        y4 = cs[7]
        xc = (x1 + x2 + x3 + x4) / 4
        yc = (y1 + y2 + y3 + y4) / 4

        width = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        height = math.sqrt(math.pow(x4 - x1, 2) + math.pow(y4 - y1, 2))

        x0 = round(xc - width / 2)
        y0 = round(yc - height / 2)
        width = round(width)
        height = round(height)

        bbox = [x0, y0, width, height]
        print(yc)
        print(bbox)

        # bbox = cv2.selectROI(frame, False)
        tracker = cv2.TrackerCSRT_create()

        initial = tracker.init(frame, bbox)

        for img_file in img_files:
            frame = cv2.imread(img_file)
            timer = cv2.getTickCount()

            initial, bbox = tracker.update(frame)

            fps = 30

            # if ok:
            p1 = bbox[0], bbox[1]
            p2 = bbox[0] + bbox[2], bbox[1] + bbox[3]

            px1 = max(bbox[0], 0)
            py1 = max(bbox[1], 0)
            px2 = max(px1 + bbox[2], 0)
            py2 = max(py1, 0)
            px3 = max(px2, 0)
            py3 = max(py2 + bbox[3], 0)
            px4 = max(px1, 0)
            py4 = max(py3, 0)

            fw.write(",".join(str(x) for x in [px1, py1, px2, py2, px3, py3, px4, py4]))
            fw.write("\n")

            # cv2.rectangle(frame, p1, p2, (225, 0, 0), 2, 1)
            # else:
            #     cv2.putText(frame, "tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # cv2.putText(frame, 'KCF Tracker', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # cv2.imshow("Tracking", frame)

        fw.close()
#
# import os
# for path,dir_list,file_list in os.walk("datasets/predict_csrt_3/"):
#   count = 0
#   for file in file_list:
#     count += 1
#     if count < 25:
#       continue
#     file1 = open("datasets/predict_csrt_3/"+file, 'r')
#     file2 = open("datasets/predict_csrt_4/" + file, 'w')
#     lines = file1.read().splitlines()
#     for line in lines:
#       cs = [float(t)+2.5 for t in line.split(",")]
#       file2.write(",".join(str(x) for x in cs))
#       file2.write("\n")
