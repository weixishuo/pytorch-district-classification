import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import dlib
import cv2

TRANS = transforms.ToTensor()

def district_classify(net, img_array):
    img = img_array[:, :, 0:3].copy()
    width = img.shape[1]
    height = img.shape[0]
    scale = max(1.0, max(width, height) / 1000.0)
    dst_width = round(width / scale)
    dst_height = round(height / scale)
    img = cv2.resize(img, (dst_width, dst_height), interpolation=cv2.INTER_AREA)
    detector = dlib.cnn_face_detection_model_v1(
        '/data/public/weixishuo/face_classification/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('/data/public/weixishuo/face_classification/shape_predictor_5_face_landmarks.dat')
    dets = detector(img, 1)
    faces = dlib.full_object_detections()
    for loc in dets:
        faces.append(sp(img, loc.rect))

    face_imgs = np.array([])
    if len(faces) > 0:
        face_imgs = np.array(dlib.get_face_chips(img, faces, size=64))

    net.cuda()
    net.eval()
    ret_list = []
    classes = ['asian', 'western']
    for ind, face_img in enumerate(face_imgs):
        tensor = torch.unsqueeze(TRANS(face_img), 0)
        inputs = Variable(tensor).cuda()
        probs = net(inputs)
        top, right, bot, left = faces[ind].rect.top(), faces[ind].rect.right(), faces[ind].rect.bottom(), faces[
            ind].rect.left()
        top = int(max(0, top) * scale)
        left = int(max(0, left) * scale)
        right = int(min(right, dst_width) * scale)
        bot = int(min(bot, dst_height) * scale)
        ret_list.append({'loc': (top, right, bot, left), 'probs':{classes[0]:probs.data[0][0], classes[1]:probs.data[0][1]}})

    return ret_list
