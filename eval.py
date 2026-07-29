import json
import os

import pandas
import torch
import argparse

import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torch.utils.data import DataLoader

from Datasets import COCOTest, FlickrTest, TasvirEtTest, getTestTransforms
from Model.TRCaptionNet import TRCaptionNet
from utils import over_write_args

RESULT = {
    "config": [],
    "weight": [],
    "Data": [],
    "Bleu_1": [],
    "Bleu_2": [],
    "Bleu_3": [],
    "Bleu_4": [],
    "METEOR": [],
    "ROUGE_L": [],
    "CIDEr": [],
    "SPICE": [],
}


@torch.no_grad()
def predict(model, data_loader, device, max_length=None, min_length=12, num_beams=3, repetition_penalty=1.1):
    # evaluate
    model.eval()
    result = []

    counter = 0
    for image, img_ids in tqdm.tqdm(data_loader):
        image = image.to(device)
        preds = model.generate(image, max_length=max_length, min_length=min_length, num_beams=num_beams, repetition_penalty=repetition_penalty)
        for pred, img_id in zip(preds, img_ids):
            result.append({"image_id": int(img_id), "caption": pred})
            counter += 1
    return result


def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


def test(opt):
    print(opt)

    # initialize model
    model = TRCaptionNet(args.model)
    model = model.to(opt.device)

    checkpoint = torch.load(opt.weights)
    msg = model.load_state_dict(checkpoint['model'])
    print("Model Load : ", msg)
    model.eval()

    test_transforms = getTestTransforms(vision_model=opt.model)

    if opt.dataset.lower() == 'coco' or opt.dataset.lower() == 'turkishmscoco':
        test_dataset = COCOTest(dataset_root=opt.test_data,
                                json_path=opt.test_json,
                                transforms=test_transforms)
    elif opt.dataset.lower() == 'tasviret':
        test_dataset = TasvirEtTest(dataset_root=opt.test_data,
                                    json_path=opt.test_json,
                                    transforms=test_transforms)
    elif opt.dataset.lower() == 'flickr':
        test_dataset = FlickrTest(dataset_root=opt.test_data,
                                  json_path=opt.test_json,
                                  transforms=test_transforms)
    else:
        raise Exception()

    test_loader = DataLoader(test_dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers,
                             pin_memory=True,
                             shuffle=False)

    test_result = predict(model, test_loader, opt.device, max_length=None, min_length=6, num_beams=3, repetition_penalty=1.2)

    result_file = ".tmp.json"
    json.dump(test_result, open(result_file, 'w'))

    result = evaluate_on_coco_caption(result_file, opt.test_json)
    os.remove(result_file)

    RESULT["config"].append(opt.config)
    RESULT["weight"].append(opt.weights)
    RESULT["Data"].append(opt.dataset)
    RESULT["Bleu_1"].append(result["Bleu_1"])
    RESULT["Bleu_2"].append(result["Bleu_2"])
    RESULT["Bleu_3"].append(result["Bleu_3"])
    RESULT["Bleu_4"].append(result["Bleu_4"])
    RESULT["METEOR"].append(result["METEOR"])
    RESULT["ROUGE_L"].append(result["ROUGE_L"])
    RESULT["CIDEr"].append(result["CIDEr"])
    RESULT["SPICE"].append(result["SPICE"])

    file_path = "results.xlsx"
    df_new = pandas.DataFrame(RESULT)

    if os.path.isfile(file_path):
        df_existing = pandas.read_excel(file_path)
        df_all = pandas.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(file_path, index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TR-CLIP-Captioning!')
    parser.add_argument('--config', type=str, default='configs/turkcap8m/turkcap8m_exp1_base16_electra.yaml')
    parser.add_argument('--weights', type=str, default='turkcap8m_experiments/turkcap8m_exp1_base16_electra/model_last.pth')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--test-data', type=str, default='data/flickr')
    parser.add_argument('--test-json', type=str, default='data/flickr/flickr30k_test_trV2.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-worker', type=int, default=8)
    args = parser.parse_args()
    over_write_args(args, args.config)
    test(args)
