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
import ModelsConstants

def track_psnrs(output, label, scene_path, name):
    label = np.expand_dims(np.rollaxis(label, 2), 0)
    cropped_label = crop_center(torch.from_numpy(label).cuda(), ModelsConstants.cnn_crop_size)
    cropped_label = torch.squeeze(cropped_label)

    psnr_l = str(psnr(output, cropped_label))

    tonemapped_label = range_compressor(cropped_label)
    tonemapped_output = range_compressor(output)

    psnr_t = str(psnr(tonemapped_output, tonemapped_label))

    with open(os.path.join(scene_path, '{}.psnrs'.format(name)), "w") as f:
        f.write(psnr_l + "\n")
        f.write(psnr_t + "\n")

    psnrs_l_path = os.path.join("../", "{}.psnrs_l".format(name))
    if os.path.exists(psnrs_l_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(psnrs_l_path, append_write) as f:
        f.write(psnr_l + "\n")

    psnrs_t_path = os.path.join("../", "{}.psnrs_t".format(name))
    if os.path.exists(psnrs_t_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(psnrs_t_path, append_write) as f:
        f.write(psnr_t + "\n")

def generate_hdr_img(scene_path, trainer, name):
    #Read Expo times in scene
    expoTimes = ReadExpoTimes(os.path.join(scene_path, 'exposure.txt'))

    #Read Image in scene
    fileNames = list_all_files_sorted(scene_path, '.tif')
    original_imgs, label = ReadTrainingData(fileNames)

    imgs, label = PrepareInputFeatures(original_imgs, expoTimes, label, False)

    imgs = np.rollaxis(imgs, 2)

    imgs = np.expand_dims(imgs, 0).astype('float32')

    output, imgs, weights = trainer.evaluate_imgs(torch.from_numpy(imgs), expos=expoTimes, tone_mapping=False)

    weights_path = os.path.join(scene_path, "{}.weights_and_refined".format(name))
    os.makedirs(weights_path, exist_ok=True)
    exponame = ["low", "medium", "high"]
    weights = weights.data.cpu().numpy()
    weights = np.squeeze(weights)
    weights = np.rollaxis(weights, 0, start=3)
    imgs = imgs.data.cpu().numpy()
    imgs = np.squeeze(imgs)
    imgs = np.rollaxis(imgs, 0, start=3)
    for i in range(3):
        weight_path = os.path.join(weights_path, '{}expo.{}.weight.png'.format(exponame[i], name))
        weight = weights[:, :, i * 3:(i + 1) * 3]
        weight = weight[:, :, [2, 1, 0]]
        imageio.imsave(weight_path, weight)

        img_path = os.path.join(weights_path, '{}expo.{}.refined.png'.format(exponame[i], name))
        img = imgs[:, :, i * 3:(i + 1) * 3]
        img = img[:, :, [2, 1, 0]]
        imageio.imsave(img_path, img)

    output = torch.squeeze(output)

    #track_psnrs(output, label, scene_path, name)

    output = output.cpu().numpy()

    output = np.rollaxis(output, 0, start=3)

    output_hdr_path = os.path.join(scene_path, '{}.hdr'.format(name))
    imageio.imsave(output_hdr_path, output[:, :, [2, 1, 0]], format='hdr')

    #output_tonemapped_path = os.path.join(scene_path, "{}.png".format(name))

    #subprocess.call(["luminance-hdr-cli", "-t", "mantiuk06", "-p", "contrast=0.1:saturation=2.0:detail=1.0:equalization=false", "-o", output_tonemapped_path, "-l", output_hdr_path])

def generate_hdr_imgs(trainer, name, path_to_scenes=os.path.join(Constants.scenes_root, Constants.test_directory)):
    test_scenes = list_all_folders(path_to_scenes)

    for scene in test_scenes:
        generate_hdr_img(os.path.join(path_to_scenes, scene), trainer, name)

def generate_hdr_img_for_architectures():
    #trainer = DirectTrainerDeepHDR(checkpoint="../checkpoints/Direct/model_best.pth")

    #generate_hdr_img_for_tests(trainer, "Direct")

    trainer = WeTrainerDeepHDR(checkpoint="../checkpoints/WE/model_best.pth")

    generate_hdr_imgs(trainer, "WE")

    #trainer = WieTrainerDeepHDR(checkpoint="../checkpoints/WIE/model_best.pth", phase=2)

    #generate_hdr_imgs(trainer, "WIE")

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

'''
    wegihts_path = os.path.join(scene_path, "{}.weights".format(name))
    os.makedirs(wegihts_path, exist_ok=True)
    exponame = ["low", "medium", "high"]
    weights = weights.data.cpu().numpy()
    weights = np.squeeze(weights)
    weights = np.rollaxis(weights, 0, start=3)
    for i in range(3):
        wegiht_path = os.path.join(wegihts_path, '{}expo.{}.weight.png'.format(exponame[i],name))
        weight = weights[:, :, i * 3:(i + 1) * 3]
        weight = weight[:, :, [2, 1, 0]]
        imageio.imsave(wegiht_path, weight)
'''