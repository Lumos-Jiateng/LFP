# generate images for certain range of data.  -- for constructing dataset

from torchvision.datasets.utils import download_url
from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
import cv2
import torchvision


class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

class Get_images():

    def __init__(self, video_root, ann_root, filename, target_dir ,num_frm=9, frm_sampling_strategy="uniform", max_img_width=384,max_image_height=384,video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''     
        self.target_dir = target_dir   
        with open(os.path.join(ann_root,filename),'r',encoding='utf-8') as file:
            self.annotation = json.load(file)
        
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        #self.max_img_size = max_img_size
        self.max_height_size=max_image_height
        self.max_width_size=max_img_width
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        #contains video_name,section_number,start_frame,end_frame,caption
        self.captions=self.annotation['database']  
       
        
    def get_images(self):  
        annotation_list = []
        for youtube_id in self.captions:
            video_info = self.captions[youtube_id]
            video_path = self.video_root + str('/') + str(video_info['recipe_type']) +'/' + str(youtube_id) + self.video_fmt
            #video_path = self.video_root +'/' + str(youtube_id) + self.video_fmt
            fps = self.get_fps(video_path)
            annotation=video_info['annotation']
            if not os.path.exists(self.target_dir + str(video_info['recipe_type']) + '/'):
                os.makedirs(self.target_dir + str(video_info['recipe_type']) + '/')
            if not os.path.exists(self.target_dir + str(video_info['recipe_type']) + '/'+ str(youtube_id) + '/'):
                os.makedirs(self.target_dir + str(video_info['recipe_type']) + '/'+ str(youtube_id) + '/')

            for i in range(len(annotation)):
                storage_dict = {}
                current_section = i
                path_s_start = self.target_dir + str(video_info['recipe_type']) + '/' + str(youtube_id) + '/' + 'image_s_' + str(current_section) +'.jpg'
                path_m = self.target_dir + str(video_info['recipe_type']) + '/' + str(youtube_id) + '/' + 'image_m_' + str(current_section) +'.jpg'
                path_s_last = self.target_dir + str(video_info['recipe_type']) + '/' + str(youtube_id) + '/' + 'image_s0_' + str(current_section) +'.jpg'
                piece_infor=annotation[i]
                start_time=piece_infor['segment'][0]
                end_time=piece_infor['segment'][1]
                #print(video_path,path_s,path_m,start_time,end_time)
                vid_frm_array = self._load_video_from_path_decord(video_path, 
                                                                  height=self.max_height_size, 
                                                                  width=self.max_width_size,
                                                                  start_time=start_time,
                                                                  end_time=end_time,
                                                                  fps=fps)
                if(vid_frm_array == None):
                    continue
                torchvision.utils.save_image(vid_frm_array[0,:,:,:],path_s_start)
                torchvision.utils.save_image(vid_frm_array[8,:,:,:],path_s_last)
                torchvision.utils.save_image(self.image_combine(vid_frm_array),path_m)
                storage_dict['single_image'] = path_s_start
                storage_dict['single_image_lf'] = path_s_last
                storage_dict['multi_image'] = path_m
                storage_dict['caption'] = piece_infor['label']
                annotation_list.append(storage_dict)
        
        with open(self.target_dir + 'annotation.json' ,'w') as f:
            json.dump(annotation_list,f)


    def image_combine(self,img_frms):
        assert(img_frms.shape[0] == 9)
        img_frm_line1=torch.cat((img_frms[0,:,:,:],img_frms[1,:,:,:],img_frms[2,:,:,:]),1)
        img_frm_line2=torch.cat((img_frms[3,:,:,:],img_frms[4,:,:,:],img_frms[5,:,:,:]),1)
        img_frm_line3=torch.cat((img_frms[6,:,:,:],img_frms[7,:,:,:],img_frms[8,:,:,:]),1)
        return torch.cat((img_frm_line1,img_frm_line2,img_frm_line3),2)     

    def get_fps(self,video_path):       
        video = cv2.VideoCapture(video_path) 
        fps = video.get(cv2.CAP_PROP_FPS) 
        return fps       

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            #Get the total length of the video frames.
            vlen = len(vr)
            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
                duration = end_idx - start_idx

            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx+duration/(self.num_frm-1), duration/(self.num_frm-1), dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))
            raw_sample_frms = vr.get_batch(frame_indices).float()
        except Exception as e:
            print(f'error detected for video_path{video_path}')
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # torch tensor
        # raw_sample_frms = np.transpose(raw_sample_frms, (0, 3, 1, 2)) # numpy

        return raw_sample_frms/255.0

if __name__ == '__main__':
    decord.bridge.set_bridge("torch")

    video_root_train = '/nfs4-p1/ljt/Code/BRE/coin_dataset'
    video_root_val = '/nfs4-p1/ljt/Code/BRE/coin_dataset'
    video_root_test = '/nfs4-p1/ljt/Code/BRE/coin_dataset'
    
    ann_root = YOUR_JSON_FILE
    
    filename_train = 'train_coin.json'
    filename_val = 'val_coin.json'
    filename_test = 'test_coin.json' 
    
    Target_Train_image_dir = YOUR_TARGET_IMAGE_DIRECTORY

    Get_train_images = Get_images(video_root_train, ann_root, filename_train,Target_Train_image_dir)
    Get_train_images.get_images()
    
    Target_Test_image_dir = YOUR_TARGET_IMAGE_DIRECTORY

    Get_test_images = Get_images(video_root_val, ann_root, filename_val,Target_Test_image_dir)
    Get_test_images.get_images()
    

    Target_Test_image_dir = YOUR_TARGET_IMAGE_DIRECTORY

    Get_test_images = Get_images(video_root_test, ann_root, filename_test,Target_Test_image_dir)
    Get_test_images.get_images()