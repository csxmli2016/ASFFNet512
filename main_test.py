
import torch
import cv2 
import numpy as np
import os.path as osp
import time
from models.ASFFNet import ASFFNet
import face_alignment # pip install face-alignment or conda install -c 1adrianb face_alignment
import argparse
import os
import math
from torchvision.transforms.functional import normalize
FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda' if torch.cuda.is_available() else 'cpu')

def cal_sim(lq_landmarks, ref_landmark):
    weight_eye = 1.01
    weight_mouth = 1.49

    # lq landmark process    (68,2)
    lq_landmarks_eye = np.concatenate([lq_landmarks[17:27, :], lq_landmarks[36:48, :]], axis=0)  # eye+eyebrow landmark

    # ref landmark process
    Ref_Ab = np.insert(ref_landmark, 2, 1, -1)
    Ref_Ab_eye = np.concatenate([Ref_Ab[17:27, :], Ref_Ab[36:48, :]], axis=0)  # eyebrow+eye (22, 3)
    Ref_Ab_mouth = Ref_Ab[48:, :]  # mouth #(20,2)
    

    # eye
    result_Ab_eye = np.dot(np.dot(np.linalg.inv(np.dot(Ref_Ab_eye.T, Ref_Ab_eye)), Ref_Ab_eye.T), lq_landmarks_eye) #(3, 2)
    # mouth
    result_Ab_mouth = np.dot(np.dot(np.linalg.inv(np.dot(Ref_Ab_mouth.T, Ref_Ab_mouth)), Ref_Ab_mouth.T), lq_landmarks[48:, :]) #(3, 2)

    ref_landmark_align_eye = np.dot(Ref_Ab_eye, result_Ab_eye.reshape([3, 2]))  # transposed eye landmark (22, 2)
    ref_landmark_align_mouth = np.dot(Ref_Ab_mouth, result_Ab_mouth.reshape([3, 2]))  # transposed mouth landmark (20, 2)
    Sim = weight_eye * np.linalg.norm(ref_landmark_align_eye - lq_landmarks_eye) + weight_mouth * np.linalg.norm(ref_landmark_align_mouth - lq_landmarks[48:, :])
    return Sim


def read_img_tensor(img_path=None, return_landmark=True): #rgb -1~1 
    Img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR or G
    if Img.ndim == 2:
        Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)  # GGG
    else:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # RGB
    
    if Img.shape[0] < 512 or Img.shape[1] < 512:
        Img = cv2.resize(Img, (512,512), interpolation = cv2.INTER_AREA)

    ImgForLands = Img.copy()
    Img = Img.transpose((2, 0, 1))/255.0
    Img = torch.from_numpy(Img).float()
    normalize(Img, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True)
    Img = Img.unsqueeze(0)
    SelectPred = None
    if return_landmark:
        try:
            PredsAll = FaceDetection.get_landmarks(ImgForLands)
        except:
            print('Error in detecting this face {}. Continue...'.format(img_path))
        if PredsAll is None:
            print('Warning: No face is detected in {}. Continue...'.format(img_path))
        ins = 0
        if len(PredsAll)!=1:
            hights = []
            for l in PredsAll:
                hights.append(l[8,1] - l[19,1])
            ins = hights.index(max(hights))
            print('Warning: Too many faces are detected, only handle the largest one...')
        SelectPred = PredsAll[ins]
    return Img, SelectPred

def optimal_reference_selection(lq_landmarks, hq_landmark_paths):
    # read the landmarks of all reference images
    RefPaths = os.listdir(hq_landmark_paths)
    RefsLands = []
    for path in RefPaths:
        Landmarks = []
        with open(osp.join(hq_landmark_paths, path),'r') as f:
            for line in f:
                tmp = [np.float(i) for i in line.split(' ') if i != '\n']
                Landmarks.append(tmp)
        Landmarks = np.array(Landmarks) #
        RefsLands.append(np.reshape(Landmarks, [-1, 2]))
    # read the landmarks of lq image
    LQLands = np.reshape(lq_landmarks, [-1, 2])

    # compute similarity
    RefSim = []
    for RefLands in RefsLands:
        sim = cal_sim(LQLands, RefLands)
        RefSim.append(sim)
    Optimal_index = RefSim.index(min(RefSim))
    return RefsLands[Optimal_index], RefPaths[Optimal_index]

def generate_lq_point_mask(lq_landmarks):
    PointMask = torch.zeros((1, 512, 512))
    for i in range(17, len(lq_landmarks)):
        point_x = lq_landmarks[i][0]
        point_y = lq_landmarks[i][1]
        if point_x > 1 and point_y > 1 and point_x < 512 - 2 and point_y < 512 - 2:
            PointMask[0,int(math.floor(point_y))-3:int(math.ceil(point_y))+3,int(math.floor(point_x))-3:int(math.ceil(point_x))+3] = 1
    return PointMask.unsqueeze(0)



if __name__ == '__main__':
    '''
    img_list should have the following format:
    {}\t{}\t{}\n
    that is
    low-quality image path  high-quality image path high-quality facial landmarks
    see the given example
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--img_list', type=str, default='./TestExamples/TestLists.txt', help='input path of lq image')
    parser.add_argument('-d', '--out_path', type=str, default='./TestExamples/TestResults', help='save path of restoration result')
    args = parser.parse_args()

    c_time = time.strftime("%m-%d_%H-%M", time.localtime()) 
    save_path = osp.join(args.out_path+'_'+c_time)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ASFFNet512 = ASFFNet().to(device)#
    weights = torch.load('./checkpoints/ASFFNet512.pth') 
    ASFFNet512.load_state_dict(weights['params'], strict=True)
    ASFFNet512.eval()
    num_params = 0
    for param in ASFFNet512.parameters():
        num_params += param.numel()

    print('{:>8s} : {}'.format('Using device', device))
    print('{:>8s} : {:.2f}M'.format('Model params', num_params/1e6))
    torch.cuda.empty_cache()

    if not osp.exists(args.img_list):
        exit('Error in test image list')

    fp = open(args.img_list, 'r')
    lines = fp.read().split("\n")
    lines = [line.strip() for line in lines if len(line)]
    for line in lines:
        lq_path, id_path, land_path = line.split('\t')
        if not osp.exists(lq_path):
            print('{} does not exist. Continue...'.format(lq_path))
            continue
        if len(os.listdir(id_path)) < 2:
            print('{} does not have enough high-quality references (>=2). Continue...'.format(id_path))
            continue
        if len(os.listdir(land_path)) < 2:
            print('{} does not have enough corresponding landmarks. You man run the ./TestSamples/FaceLandmarkDetection.py to obtain its landmarks...'.format(id_path))
            continue
        
        lq, lq_landmarks = read_img_tensor(lq_path, return_landmark=True)
        if lq_landmarks is None:
            print('Error in detecting landmarks of {}. Maybe its quality is very low. Continue...'.format(lq_path))
            continue
        print('Restoring {}'.format(osp.basename(lq_path)))
        optimal_ref_landmarks, optimal_ref_path = optimal_reference_selection(lq_landmarks, land_path)

        optimal_img_name = osp.basename(optimal_ref_path)[:-4]

        ref, _ = read_img_tensor(osp.join(id_path, optimal_img_name), return_landmark=False) 
        landmark_mask = generate_lq_point_mask(lq_landmarks)

        with torch.no_grad():
            output = ASFFNet512(lq.to(device), ref.to(device), landmark_mask.to(device), torch.from_numpy(lq_landmarks).unsqueeze(0), torch.from_numpy(optimal_ref_landmarks).unsqueeze(0))

        save_out = output * 0.5 + 0.5
        save_out = save_out.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        save_out = np.clip(save_out.float().cpu().numpy(), 0, 1) * 255.0

        check_lq = lq * 0.5 + 0.5
        check_lq = check_lq.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        check_lq = np.clip(check_lq.float().cpu().numpy(), 0, 1) * 255.0
        check_ref = ref * 0.5 + 0.5
        check_ref = check_ref.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        check_ref = np.clip(check_ref.float().cpu().numpy(), 0, 1) * 255.0
        
        cv2.imwrite(osp.join(save_path, osp.basename(lq_path)), np.hstack((check_lq, check_ref,save_out)))
    
