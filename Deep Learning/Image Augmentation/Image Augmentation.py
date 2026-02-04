#Dataset
import torchvision.datasets as datasets

train_dataset = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=None
)

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=None
)



#Transform
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),])



#Model
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)



#util
import torchvision.utils as utils

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
dataiter = iter(dataloader)
images, labels = next(dataiter)
img_grid = utils.make_grid(images)



#Torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

image_tensor = transform(image_color_open)

def imshow(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    plt.show()



#Albumentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(224, 224), 
    ToTensorV2()
])

transformed = transform(image=image_color)
image_tensor = transformed['image']

def imshow(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()



#Torchvision Flip
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

flipped_horizontal_image = horizontal_flip(image)



#Albumentation Flip
transform_horizontal_flip_100 = A.HorizontalFlip(p=1.0)
transform_vertical_flip_100 = A.VerticalFlip(p=1.0)

flipped_horizontal_100 = transform_horizontal_flip_100(image=image_np)['image']
flipped_vertical_100 = transform_vertical_flip_100(image=image_np)['image']



#Albumentation Crop
transform_random_crop = A.RandomCrop(height=200, width=200, p=1.0)
transform_center_crop = A.CenterCrop(height=200, width=200, p=1.0)

random_cropped = transform_random_crop(image=image_np)['image']
center_cropped = transform_center_crop(image=image_np)['image']



#Albumentation Shift, Scale, Rotateshift_transform = A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0, rotate_limit=0, p=1.0)
scale_transform = A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=0, p=1.0)
rotate_transform = A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, p=1.0)
combined_transform = A.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.4, rotate_limit=45, p=1.0)

shifted_image = shift_transform(image=image_np)['image']
scaled_image = scale_transform(image=image_np)['image']
rotated_image = rotate_transform(image=image_np)['image']
combined_image = combined_transform(image=image_np)['image']



#Albumentation ColorJitter
color_brightness_transform = A.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0, p=1.0)
color_contrast_transform = A.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0, p=1.0)
color_saturation_transform = A.ColorJitter(brightness=0, contrast=0, saturation=0.5, hue=0, p=1.0)
color_hue_transform = A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5, p=1.0)
color_all_transform = A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1.0)

color_brightness_image = color_brightness_transform(image=image_np)['image']
color_contrast_image = color_contrast_transform(image=image_np)['image']
color_saturation_image = color_saturation_transform(image=image_np)['image']
color_hue_image = color_hue_transform(image=image_np)['image']
color_all_image = color_all_transform(image=image_np)['image']



#Noise
gauss_noise_transform = A.GaussNoise(var_limit=(100.00, 200.0), p=1.0)

gauss_noised_image = gauss_noise_transform(image=image_np)['image']



#GlassBlur
transform = A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=True)

augmented_image = transform(image=image_np)['image']



#CLAHE
transform = A.CoarseDropout(max_holes=20, max_height=8, max_width=8, min_holes=2, min_height=4, min_width=4, fill_value=0, always_apply=True)

augmented_image = transform(image=image_np)['image']



#CoarseDropout
transform = A.CoarseDropout(max_holes=20, max_height=8, max_width=8, min_holes=2, min_height=4, min_width=4, fill_value=0, always_apply=True)
    
augmented_image = transform(image=image_np)['image']