import os

import Constants
from ModelUtilities import list_all_files_sorted, list_all_folders, crop_center, range_compressor, psnr
from ComputeTrainingExamples import PrepareInputFeatures
from TrainersDeepHDR import DirectTrainerDeepHDR, WeTrainerDeepHDR, WieTrainerDeepHDR
from ImagePreprocessing import ReadExpoTimes, ReadTrainingData
import torch
import argparse
import numpy as np
import imageio
import subprocess
import sys

def generate_hdr_img(scene_path, trainer, name):
    #Read Expo times in scene
    expoTimes = ReadExpoTimes(os.path.join(scene_path, 'exposure.txt'))

    #Read Image in scene
    fileNames = list_all_files_sorted(scene_path, '.tif')
    original_imgs, label = ReadTrainingData(fileNames)

    imgs, label = PrepareInputFeatures(original_imgs, expoTimes, label, False)

    imgs = np.rollaxis(imgs, 2)

    imgs = np.expand_dims(imgs, 0).astype('float32')

    output = trainer.evaluate_imgs(torch.from_numpy(imgs), expos=expoTimes, tone_mapping=False)

    output = torch.squeeze(output)

    output = output.cpu().numpy()

    output = np.rollaxis(output, 0, start=3)

    output_hdr_path = os.path.join(scene_path, '{}.hdr'.format(name))
    imageio.imsave(output_hdr_path, output[:, :, [2, 1, 0]], format='hdr')

def generate_hdr_imgs(trainer, name, path_to_scenes=os.path.join(Constants.scenes_root, Constants.test_directory)):
    test_scenes = list_all_folders(path_to_scenes)

    for scene in test_scenes:
        generate_hdr_img(os.path.join(path_to_scenes, scene), trainer, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate HDR image from a trained model.',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", dest="do_all", action="store_true")
    group.add_argument('--checkpoint', type=str)

    scene_group = parser.add_mutually_exclusive_group(required='--checkpoint' in sys.argv)
    scene_group.add_argument('--scenes', type=str)
    scene_group.add_argument('--scene', type=str)

    model_group = parser.add_mutually_exclusive_group(required='--checkpoint' in sys.argv)
    model_group.add_argument('--Direct', dest='model', action='store_const', const='Direct')
    model_group.add_argument('--WE', dest='model', action='store_const', const='WE')
    model_group.add_argument('--WIE', dest='model', action='store_const', const='WIE')

    args = parser.parse_args()

    if args.do_all:
        generate_hdr_img_for_architectures()
    else:
        if args.model == 'Direct':
            trainer = DirectTrainerDeepHDR(args.checkpoint)
        elif args.model == 'WE':
            trainer = WeTrainerDeepHDR(args.checkpoint)
        elif args.model == 'WIE':
            trainer = WieTrainerDeepHDR(args.checkpoint)

        if args.scenes:
            generate_hdr_imgs(trainer, args.model, args.scenes)
        elif args.scene:
            generate_hdr_img(args.scene, trainer, args.model)