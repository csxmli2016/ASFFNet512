
import face_alignment
import cv2
import os.path as osp
import os
import torch
import numpy as np
import argparse

'''
Install for face landmark detection:

pip install face-alignment
or
conda install -c 1adrianb face_alignment

Please refer to https://github.com/1adrianb/face-alignment for more details
'''

# Path = '/home/lxm/ASFFNet512/TestSamples/OldVideo/LQVideo'
# ImgLists = os.listdir(Path)

# Id = 0
# fp = open('./Test2.txt', 'w')
# for name in ImgLists:
#     ImgPath = osp.join('./TestSamples/OldVideo/LQVideo/', name)
#     fp.writelines(ImgPath + '\t'+'./TestSamples/OldVideo/HQVideoRef/8888\t'+'./TestSamples/OldVideo/HQVideoLands/8888\n')

# fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--ids_path', type=str, default='./HQReferences', help='file path that contains many identities')
    parser.add_argument('--check', action='store_true', help='save the face images with landmarks shown on them to check the performance')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)

    ShowLandmarkFace = True # check the landmark detection performance
    IdPath = args.ids_path
    if args.ids_path.endswith('/'):
        SaveLandPath = args.ids_path[:-1]+'_'+'Landmarks'
    else:
        SaveLandPath = args.ids_path+'_'+'Landmarks'

    os.makedirs(SaveLandPath, exist_ok=True)
    IdLists = os.listdir(IdPath)
    for IdName in IdLists:
        CuIdPath = osp.join(IdPath, IdName)
        ImgNames = os.listdir(CuIdPath)
        for ImgName in ImgNames:
            print('Detecting Id {}, ImgName {}...'.format(IdName, ImgName))
            ImgPath = osp.join(CuIdPath, ImgName) 
            Img = cv2.imread(ImgPath, cv2.IMREAD_UNCHANGED)  # BGR or G
            if Img.ndim == 2:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)  # GGG
            else:
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # RGB
            if Img.shape[0] < 512 or Img.shape[1] < 512:
                Img = cv2.resize(Img, (512,512), interpolation = cv2.INTER_AREA)
            
            try:
                PredsAll = FaceDetection.get_landmarks(Img)
            except:
                print('Error in detecting this face {}. Continue...'.format(ImgPath))
                continue
            if PredsAll is None:
                print('Warning: No face is detected in {}. Continue...'.format(ImgPath))
                continue
            ins = 0
            if len(PredsAll)!=1:
                hights = []
                for l in PredsAll:
                    hights.append(l[8,1] - l[19,1])
                ins = hights.index(max(hights))
                print('Warning: Too many faces are detected, only handle the largest one...')
            SelectPred = PredsAll[ins]
            os.makedirs(osp.join(SaveLandPath, IdName), exist_ok=True)
            np.savetxt(os.path.join(SaveLandPath, IdName, ImgName+'.txt'), SelectPred[:,0:2],fmt='%.1f')

            if args.check:
                ImgShow = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR) #RGB to BGR
                SaveTmpPath = osp.join(SaveLandPath, IdName+'_CheckLandmarks')
                os.makedirs(SaveTmpPath, exist_ok=True)
                for point in SelectPred[17:,0:2]:
                    cv2.circle(ImgShow, (int(point[0]), int(point[1])), 1, (0,255,0), 4)
                cv2.imwrite(osp.join(SaveTmpPath, ImgName), ImgShow)
