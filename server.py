import os, logging, argparse
import urllib, base64
from functools import lru_cache
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import numpy as np
import PIL
import torch
from tqdm import tqdm
from glob import glob
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
try:
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List
    api_env = True
except:
    warn = """Not exists api ENV, if you want to use api please setup by fellow
    pip install fastapi[all] uvicorn pydantic
    """
    logging.warning(warn)
    api_env = False


CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "SAM")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 1024
TOP_K_OBJ = 100
THRESHOLD = 0.85
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def load_mask_generator() -> SamAutomaticMaskGenerator:
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


@lru_cache
def load_clip(
    name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess


def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str]) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)
    similarity = logits_per_image.softmax(-1).cpu()
    return similarity[0, 0]


def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> PIL.Image.Image:
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y : y + h, x : x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = PIL.Image.fromarray(crop)
    return crop


def get_texts(query: str) -> List[str]:
    return [f"a picture of {query}", "a picture of background"]


def filter_masks(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    query: str,
    clip_threshold: float,
) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
            or image.shape[:2] != mask["segmentation"].shape[:2]
            or query
            and get_score(crop_image(image, mask), get_texts(query)) < clip_threshold
        ):
            continue

        filtered_masks.append(mask)

    return filtered_masks

def mask2box(mask):
    def mask_find_bboxs(mask):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
        stats = stats[stats[:,4].argsort()]
        return stats[:-1]

    bboxs = mask_find_bboxs(mask)
    for i, box in enumerate(bboxs):
        bboxs[i][2] += box[0]
        bboxs[i][3] += box[1]
    
    return bboxs

def draw_masks(
    image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.7
) -> np.ndarray:
    total_boxes = []
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]

        # mask2box
        mm = np.where(mask["segmentation"], 255, 0)
        mm = mm.astype(np.uint8)
        bboxes = mask2box(mm)
        
        # draw mask overlay
        # colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        # colored_mask = np.moveaxis(colored_mask, 0, -1)
        # masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        # image_overlay = masked.filled()
        # image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
        
        # draw box
        for i, box in enumerate(bboxes):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
            
            # h, w = image.shape[:-1]
            # if 0.4 * h * w > (box[2] - box[0]) * (box[3] - box[1]):
            total_boxes.append(box)
        
    return image, total_boxes


def segment(
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    clip_threshold: float,
    image_path: str,
    query: str,
):
    mask_generator = load_mask_generator()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print("image shape: ", image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reduce the size to save gpu memory
    image = adjust_image_size(image)
    masks = mask_generator.generate(image)
    masks = filter_masks(
        image,
        masks,
        predicted_iou_threshold,
        stability_score_threshold,
        query,
        clip_threshold,
    )
    image, boxes = draw_masks(image, masks)
    image = PIL.Image.fromarray(image.astype(np.uint8))
    return image, boxes

class Input(BaseModel):
    image: str              # base64
    iou_thresh: float       = 0.9
    stability_thresh: float = 0.8
    clip_thresh: float      = 0.8
    query: str              = ""
    
class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Output(BaseModel):
    bbox: List[Box]

# global app
app = FastAPI(debug=True, description="SAM with CLIP API")

@app.post("/sam")
async def model_detection(data: Input):
    # base64 to image
    # img_uri = data.image.split(",")[1]
    # imgdata = base64.b64decode(img_uri)
    # tmp_path = "~/.cache/tmp.jpg"
    tmp_path = "tmp.jpg"
    with open(tmp_path, mode="wb") as f:
        f.write(base64.b64decode(data.image))
    
    # segment
    image, boxes = segment(data.iou_thresh, data.stability_thresh, 
                        data.clip_thresh, tmp_path, data.query)
    image.save("api_result.png")
    
    res = Output(bbox=[])
    for box in boxes:
        res.bbox.append(Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default=None, help='image folder')
    parser.add_argument('--iou-thres', type=float, default=0.9, help='predicted iou threshold')
    parser.add_argument('--stability-thres', type=float, default=0.8, help='stability score threshold')
    parser.add_argument('--clip-thres', type=float, default=0.9, help='clip threshold')
    parser.add_argument('--query', type=str, default="", help='query text')
    parser.add_argument('--save-path', type=str, default="./results", help='save path')
    
    parser.add_argument('--api', action='store_true', default=False, help='api only mode')
    opt = parser.parse_args()
    
    if opt.image_path is not None:
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        image_ls = glob(opt.image_path + "/*g")
        for imagename in tqdm(image_ls):
            image, boxes = segment(opt.iou_thres, opt.stability_thres, 
                                opt.clip_thres, imagename, opt.query)
            path = os.path.join(opt.save_path, imagename.split("/")[-1])
            image.save(path)
            
            res = []
            for box in boxes:
                res.append(",".join(list(map(str, box))))
            with open(os.path.splitext(path)[0] + ".txt", 'w') as f:
                f.write("\n".join(res))
    
    elif opt.api and api_env:
        uvicorn.run('server:app', reload=True, host='0.0.0.0', port=4000)

    else:
        print(opt)
        logging.error("please check input param, and retry.")
