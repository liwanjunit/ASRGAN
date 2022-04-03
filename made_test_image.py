
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':

    crop_size = 256
    UPSCALE_FACTOR = 4
    TARGET_PATH = 'data/test/target'
    DATA_PATH = 'data/test/data'


    target_set = DatasetFromFolder(TARGET_PATH, upscale_factor=UPSCALE_FACTOR)
    target_loader = DataLoader(dataset=target_set, num_workers=4, batch_size=1, shuffle=False)
    target_bar = tqdm(target_loader, desc='[testing benchmark datasets]')

    for data, target in target_bar:
        data = ToPILImage()(data)
        data.save(DATA_PATH + '')











