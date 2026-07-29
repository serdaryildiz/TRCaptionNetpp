import io
import json

import lmdb
import warnings
from PIL import Image
from torch.utils.data import Dataset

from Datasets.utils.text_utils import clear_text

warnings.filterwarnings('ignore')


class TurkCap8M(Dataset):
    """
        TurkCap8M train dataset class
    """

    def __init__(self, dataset_root: str, json_path: str, tokenizer=None, transforms=None):
        self.dataset_root = dataset_root
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.transforms = transforms

        # init env
        self.env = lmdb.open(dataset_root, readonly=True, lock=False, readahead=False, meminit=False)

        # load captions
        self.labels = json.load(open(json_path, 'r'))
        self.prompt = ""
        self.num_images = len(self.labels)
        self.num_samples = self.num_images * 4

        self.target_keys = ["OFA-Caption-translated", "BLIP-Caption-translated",
                            "FUSE-Caption-translated", "BLIP2-Caption-translated"]

        print(f"Train Sample Size : {len(self.labels)}")
        return

    def __getitem__(self, index):
        FLAG = True
        sample = None
        while FLAG:
            try:
                sample = self.get_sample(index)
                FLAG = False
            except KeyError as e:  # other caption
                index += 1
            except Exception as e:  # other image
                index += 4

            if index == self.num_samples:
                index = 0

        return sample

    def get_sample(self, index):
        caption_index = index % 4
        # caption_index = 0
        img_id = index // 4
        # img_id = index
        sample = self.labels[img_id]

        with self.env.begin(write=False) as txn:
            key = sample["image-key"].encode()

            img_buffer = txn.get(key)
            image = Image.open(io.BytesIO(img_buffer)).convert('RGB')

        # get caption
        caption = sample[self.target_keys[caption_index]]
        caption = self.prompt + clear_text(caption, 30)

        # transform
        if self.transforms is not None:
            image = self.transforms(image)

        # tokenize
        if self.tokenizer is not None:
            caption = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt")

        return image, caption, img_id

    def __len__(self):
        return self.num_samples
        # return self.num_images
