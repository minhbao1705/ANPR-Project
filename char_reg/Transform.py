import torchvision.transforms as transforms
class DataTransform():
    def __init__(self, inputsize, degree=45, shear=2):
        self.transform = {"train":
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(1),
                transforms.Resize((inputsize, inputsize),interpolation= transforms.InterpolationMode.NEAREST),
                transforms.RandomAffine(degrees=degree, shear= shear)
                ]),
            "val":
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(1),
                transforms.Resize((inputsize, inputsize),interpolation= transforms.InterpolationMode.NEAREST),
            ])
        }

    def __call__(self, img, phase):
        return self.transform[phase](img)