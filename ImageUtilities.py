from skimage.util.shape import view_as_windows
    
def extract_patches_from_image(self, img, patch_size, stride):
    return view_as_windows(img, (self.patch_size, self.patch_size), self.stride)

