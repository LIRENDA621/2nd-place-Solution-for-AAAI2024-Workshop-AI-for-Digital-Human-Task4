import os
from options.test_feature2face_options import TestOptions
from datasets import create_dataset
from models import create_model
from util import html
import time
from util.visualizer import Visualizer
from util.visualizer import save_images
import tqdm
import glob
from PIL import Image
import moviepy
import moviepy.editor
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip, AudioFileClip

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print(len(dataset))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.load_epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)


    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.load_epoch))
    model.eval()

    if opt.n_prev_frames > 0:
        name = opt.name
        n_prev_frames = opt.n_prev_frames

        opt.name = opt.single_name     #load model trained with n_prev_frames=0
        opt.n_prev_frames = 4
        single_model = create_model(opt)
        single_model.setup(opt)
        opt.name = name
        opt.n_prev_frames = n_prev_frames
        single_model.eval()

    generated_frames = []
    gt_frames = []

    # # for quick debug
    # dataset.dataset.gt_feature_path=dataset.dataset.gt_feature_path[:100]
    
    for i, data in tqdm.tqdm(enumerate(dataset)):
        # if i>1000:
        #     break
        if opt.n_prev_frames > 0:
            if i < opt.n_prev_frames:
                single_model.set_input(data)  # unpack data from data loader
                single_model.test()           # run inference
                visuals = single_model.get_current_visuals()  # get image results
                generated_frames.append(visuals['pred_fake'])
                gt_frames.append(visuals['pred_real'])
                img_path = single_model.get_image_paths()     # get image paths
            else:
                # data['cand_image'] = torch.cat(generated_frames[-opt.n_prev_frames:], dim=1)
                data['cand_image'] = torch.cat(gt_frames[-opt.n_prev_frames:], dim=1)
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                visuals = model.get_current_visuals()  # get image results
                generated_frames.append(visuals['pred_fake'])
                gt_frames.append(visuals['pred_real'])
                img_path = model.get_image_paths()

        else:
            # if i < 2 or i+2 > len(dataset)-1:
            #     continue
            torch.cuda.synchronize()
            t = time.time()
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals(mode='test_demo')  # get image results
            torch.cuda.synchronize()
            # print(time.time()-t)
            
            generated_frames.append(visuals['pred_fake'])
            img_path = model.get_image_paths()
            # print(visuals['pred_fake'].shape)
        if i % 5 == 0:  # save images to an HTML file
            # print('processing (%04d)-th image... %s' % (i, img_path))
            pass
        save_images(webpage, visuals, i)
    webpage.save()  # save the HTML

    print('done!')

    img_list = []
    feat_list = []
    # gt_img_list = []
    # for i in range(len(dataset)-1):
    for i in tqdm.tqdm(range(len(generated_frames))):
        img_path = web_dir + '/images/%06d_pred_fake.png'%i
        img_list.append(img_path)
        feat_path = web_dir + '/images/%06d_feature_map.png'%i
        feat_list.append(feat_path)
        # gt_img_path = web_dir + '/images/%06d_pred_real.png'%i
        # gt_img_list.append(gt_img_path)
    outVid = []
    for i in tqdm.tqdm(range(len(img_list))):
        img = img_list[i]
        img_ = Image.open(img).convert("RGB")
        img_ = np.array(img_)

        img_ = img_[:, :400, ...]
        img_ = img_[:, 112:, ...]

        # gt_img = gt_img_list[i]
        # gt_img_ = Image.open(gt_img).convert("RGB")
        # gt_img_ = np.array(gt_img_)
        feat = feat_list[i]
        feat_ = Image.open(feat).convert("RGB")
        feat_ = np.array(feat_)

        out = img_
        outVid.append(img_)
        # out = np.concatenate((feat_, img_), axis=1)
        # outVid.append(out)

    moviepy.editor.ImageSequenceClip(sequence=[(npyFrame).clip(0.0, 255.0).round().astype(np.uint8) for npyFrame in outVid], fps=30).write_videofile(web_dir+'/generated_video.mp4')

    # 创建视频剪辑
    video_clip = ImageSequenceClip(sequence=[(npyFrame).clip(0.0, 255.0).round().astype(np.uint8) for npyFrame in outVid], fps=30)

    # 加载音频剪辑
    # audio_clip = AudioFileClip("/nws/user/lirenda621/Data/tmp/PersonB_test.mp3")
    # audio_clip = AudioFileClip("/nws/user/lirenda621/Data/tmp/PersonA_test.mp3")
    audio_clip = AudioFileClip("personA/train_val.wav")



    # 将音频与视频合并
    final_clip = video_clip.set_audio(audio_clip)

    # 写入最终的视频文件
    final_clip.write_videofile(web_dir+'/generated_video.mp4', codec='libx264', audio_codec='aac')