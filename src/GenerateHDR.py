from ModelUtilities import CropBoundaries, extract_patches_from_image, LDR_to_HDR, LDR_to_LDR
from ComputeTrainingExamples import PrepareInputFeatures
from DeepHDRTrainers import DirectDeepHDRTrainer, WeDeepHDRTrainer, WieDeepHDRTrainer
from ImagePreprocessing import select_subset
from ImagePreprocessing import get_num_patches
from ModelUtilities import range_compressor
import torch
import scipy.misc

def generate_hdr_img(path, model)
#Read Expo times in scene
expoTimes = ReadExpoTimes(os.path.join(path, 'exposure.txt'))

#Read Image in scene
fileNames = list_all_files_sorted(path, '.tif')
original_imgs, label = ReadTrainingData(fileNames)

imgs, _ = PrepareInputFeatures(original_imgs, expoTimes, label, False)

imgs = CropBoundaries(imgs, Constants.crop)

num_patches = get_num_patches(imgs.shape[0], imgs.shape[1], Constants.patchSize, Constants.stride)

imgs_patches = np.zeros((Constants.patchSize, Constants.patchSize, imgs.shape[-1], num_patches), dtype='float32')

transformed_imgs, transformed_labels = imgAugmentation.augment(imgs, label)

cur_imgs_patches = extract_patches_from_image(transformed_imgs, Constants.patchSize, Constants.stride)
imgs_patches[:, :, :, i * num_patches:(i + 1) * num_patches] = cur_imgs_patches.astype('float32')

indexes = select_subset(imgs_patches[:, :, 3:6])

cnn = model(checkpoint='./checkpoints/Direct/model_best.pth')

output = cnn(imgs_patches)

hdr_ref = LDR_to_HDR(original_imgs[1], expoTimes[1])

hdr = np.zeros(hdr_ref.shape)

hdr_tonemapped = range_compressor(torch.from_numpy(hdr)).numpy()

for index in indexes:
    col = index % 69
    row = (col // 44)

    row += 6
    col += 6

    hdr_tonemapped[row:row+28, col:col+28, :] = output[index]

scipy.misc.imsave(os.path.join(path, 'outfile.png'), hdr_tonemapped)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate HDR image from a trained model.',
    )

    checkpoint_group.add_argument('checkpoint', type=str)

    parser.add_argument('path', type=str)

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--Direct', dest='model', action='store_const', const='Direct')
    model_group.add_argument('--WE', dest='model', action='store_const', const='WE')
    model_group.add_argument('--WIE', dest='model', action='store_const', const='WIE')

    args = parser.parse_args()

    if args.model == 'Direct':
        trainer = DirectDeepHDRTrainer(args.checkpoint)
    elif args.model == 'WE':
        trainer = WeDeepHDRTrainer(args.checkpoint)
    elif args.model == 'WIE':
        trainer = WieDeepHDRTrainer(args.checkpoint)

    trainer.evaluate