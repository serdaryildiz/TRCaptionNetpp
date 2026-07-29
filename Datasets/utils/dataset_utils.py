import torch.utils.data
from torchvision import transforms

from Datasets import TurkCap8M
from Datasets.coco import COCOTrain, COCOTest
from Datasets.flickr import FlickrTrain, FlickrTest
from Datasets.tasviret import TasvirEtTrain, TasvirEtTest

from Model import clip


def getTestTransforms(vision_model=None):
    """
        returns clip image encoder transforms
    :param vision_model:
    :return:
    """
    if vision_model is None:
        _, preprocess = clip.load("ViT-B/32", jit=False)
    else:
        if "clip" in vision_model:
            _, preprocess = clip.load(vision_model["clip"], jit=False)
        elif "dino2" in vision_model:
            preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        else:
            raise Exception("")
    return preprocess


def getTrainDataset(dataset_name: str, dataset_root: str, train_json_path: str,
                    vision_model: str = None) -> torch.utils.data.Dataset:
    """
        gets train dataset class
    :param dataset_name: dataset name
    :param dataset_root: dataset root path
    :param train_json_path: annotation file path
    :param vision_model: vision model name for image transforms
    :return: dataset class
    """
    test_transforms = getTestTransforms(vision_model)
    train_transforms = test_transforms  # TODO : maybe train transforms can be added.

    # get torch dataset class
    if dataset_name.lower() == 'coco-karphaty':
        train_dataset = COCOTrain(dataset_root=dataset_root,
                                  json_path=train_json_path,
                                  tokenizer=None,
                                  transforms=train_transforms)
    elif dataset_name.lower() == 'tasvir-et':
        train_dataset = TasvirEtTrain(dataset_root=dataset_root,
                                      json_path=train_json_path,
                                      transforms=train_transforms)
    elif dataset_name.lower() == 'flickr30k':
        train_dataset = FlickrTrain(dataset_root=dataset_root,
                                    json_path=train_json_path,
                                    transforms=train_transforms)
    elif dataset_name.lower() == 'turkcap8m':
        train_dataset = TurkCap8M(dataset_root=dataset_root,
                                  json_path=train_json_path,
                                  transforms=train_transforms)
    else:
        raise Exception(f"Unknown dataset : {dataset_name}")
    return train_dataset


def getTestDataset(dataset_name, dataset_root, test_json_path, vision_model=None):
    """
        gets test dataset class
    :param dataset_name: dataset name
    :param dataset_root: dataset root path
    :param test_json_path: annotation file path
    :param vision_model: vision model name for image transforms
    :return: dataset class
    """
    test_transforms = getTestTransforms(vision_model)

    if dataset_name.lower() == 'coco-karphaty':
        test_dataset = COCOTest(dataset_root=dataset_root,
                                json_path=test_json_path,
                                transforms=test_transforms)
    elif dataset_name.lower() == 'tasvir-et':
        test_dataset = TasvirEtTest(dataset_root=dataset_root,
                                    json_path=test_json_path,
                                    transforms=test_transforms)
    elif dataset_name.lower() == 'flickr30k':
        test_dataset = FlickrTest(dataset_root=dataset_root,
                                  json_path=test_json_path,
                                  transforms=test_transforms)
    else:
        raise Exception(f"Unknown dataset : {dataset_name}")
    return test_dataset


def getCocoDataset(dataset_root, train_json_path, test_json_path, vision_model=None):
    """ returns coco dataset train and test classes """
    test_transforms = getTestTransforms(vision_model)
    train_dataset = COCOTrain(dataset_root=dataset_root,
                              json_path=train_json_path,
                              tokenizer=None,
                              transforms=test_transforms)

    test_dataset = COCOTest(dataset_root=dataset_root,
                            json_path=test_json_path,
                            transforms=test_transforms)
    return train_dataset, test_dataset


def getTasvirEtDataset(dataset_root, train_json_path, test_json_path, vision_model=None):
    """ returns tasvir-et dataset train and test classes """

    test_transforms = getTestTransforms(vision_model)
    train_dataset = TasvirEtTrain(dataset_root=dataset_root,
                                  json_path=train_json_path,
                                  transforms=test_transforms)

    test_dataset = TasvirEtTest(dataset_root=dataset_root,
                                json_path=test_json_path,
                                transforms=test_transforms)
    return train_dataset, test_dataset
