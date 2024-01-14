### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import os
import time
from . import util
from . import html
import ntpath
import scipy.misc 
import wandb
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0
        self.wandb_project_name = opt.wandb_project_name
        if opt.isTrain:
            if self.tf_log:
                from torch.utils.tensorboard import SummaryWriter
    #            import tensorflow as tf      
    #            self.tf = tf
                self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
    #            self.writer = tf.summary.FileWriter(self.log_dir)
                self.writer = SummaryWriter(self.log_dir, flush_secs=1)   
    
            if self.use_html:
                self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
                self.img_dir = os.path.join(self.web_dir, 'images')
                print('create web directory %s...' % self.web_dir)
                util.mkdirs([self.web_dir, self.img_dir])
    
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt, entity="cmu-gil") if not wandb.run else wandb.run
            self.wandb_run._label(repo=self.wandb_project_name)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, iter, save_result):
#        if self.tf_log: # show images in tensorboard output
#            img_summaries = []
#            for label, image_numpy in visuals.items():
#                # Write the image to a string
#                try:
#                    s = StringIO()
#                except:
#                    s = BytesIO()
#                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
#                # Create an Image object
#                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
#                # Create a Summary value
#                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))
#
#            # Create and write Summary
#            summary = self.tf.Summary(value=img_summaries)
#            self.writer.add_summary(summary, step)

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

        # if self.use_html: # save images to a html file
        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # print('saving images!')
            for label, image_numpy in visuals.items():
                image_numpy = util.tensor2im(image_numpy)
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_%d.jpg' % (epoch, iter, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s.jpg' % (epoch, iter, label))
                    # print(img_path)
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 5:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):

        if self.use_wandb:
            self.wandb_run.log(errors)

        if self.tf_log:
            for tag, value in errors.items():            
#                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
#                self.writer.add_summary(summary, step)
                self.writer.add_scalar(tag, value, step)
                

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        # print(errors)
        for k, v in sorted(errors.items()):
            # if v != 0:
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, image_dir, visuals, image_path, webpage=None):        
        dirname = os.path.basename(os.path.dirname(image_path[0]))
        image_dir = os.path.join(image_dir, dirname)
        util.mkdir(image_dir)
        name = image_path
#        name = os.path.basename(image_path[0])
#        name = os.path.splitext(name)[0]        

        if webpage is not None:
            webpage.add_header(name)
            ims, txts, links = [], [], []         

        for label, image_numpy in visuals.items():
            save_ext = 'jpg'
            image_name = '%s_%s.%s' % (label, name, save_ext)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            if webpage is not None:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        if webpage is not None:
            webpage.add_images(ims, txts, links, width=self.win_size)

    def vis_print(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

def save_images(webpage, visuals, i, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    # short_path = ntpath.basename(image_path[0])
    # name = os.path.splitext(short_path)[0]
    name = '%06d'%i

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)

