from skimage.util.shape import view_as_windows
import torch
    
def extract_patches_from_image(img, patch_size, stride):
    return view_as_windows(img, (self.patch_size, self.patch_size), self.stride)

def weighted_average(weights, imgs, num_channels):
    assert weights.size() == imgs.size()
    assert weights.size()[1] % num_channels == 0
    
    mat_size = weights.size()
    num_imgs = mat_size[1] / num_channels

    average_denominator = torch.zeros(mat_size[0], 3, mat_size[2], mat_size[3])
    average_numerator = torch.zeros(mat_size[0], 3, mat_size[2], mat_size[3])
    for index in range(num_imgs):
        weight_mat = weights[:, index * num_channels:(index + 1) * num_channels]
        img = imgs[:, index * num_channels:(index + 1) * num_channels]

        average_denominator += weight_mat

        average_numerator += weight_mat.matmul(img)
    
    return average_numerator / average_denominator

def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo

def l2_distance(result, target):
        assert result.size() == target.size()
        return (target - result).pow(2).sum()

def tone_map(x):
    return torch.log(x.mul(mu).add(1)) / log(1 + mu)

def crop(imgs):
    
