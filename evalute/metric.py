import torch
import torchvision
import numpy as np
import os
from PIL import Image

class MetricEvaluation(object):
    
    def __init__(self, model, transform, img_size=(640, 640)):
        """
        Parameters:
            - model: model in nn.Modules() to analyze
            - transform: transform function used in model
            - img_size: input image size (default for YOLOX-l is 640x640)
        """
        self.model = model.eval()
        self.transform = transform
        self.img_size = img_size
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pg = 0.
        self.ebpg = 0.
        self.drop_conf_score = 0.
        self.drop_info_score = 0.


    def pointing_game(self, saliency_map, bbox, tiny=False):
        """
        Calculate pointing game for one saliency map given target bounding box
        Parameters:
            - saliency_map: 2D-array in range [0, 1]
            - bbox: Target bounding box to analyze Tensor([xmin, ymin, xmax, ymax, p_obj, p_cls, cls])
            - tiny: Analyze for tiny object (default = False)
        Returns: PG score for the saliency map
        """
        x_min, y_min, x_max, y_max = bbox[:4]
        point_x, point_y = np.where(saliency_map == np.max(saliency_map))
        hit = 0
        miss = 0
        for i, val in enumerate(point_x):
            if x_min < point_x[i] < x_max and y_min < point_y[i] < y_max:
                hit += 1
            else:
                miss += 1            
            if not tiny:
                break
        self.pg = hit / (hit + miss)
        return self.pg

    def energy_based_pointing_game(self, saliency_map, bbox):
        """
        Calcualte the energy-based pointing game score of one saliency map given target bounding box
        Parameters:
            - saliency_map: 2D-array in range [0, 1]
            - bbox: Target bounding box to analyze Tensor([xmin, ymin, xmax, ymax, p_obj, p_cls, cls])
        Returns: EBPG score for the saliency map
        """
        x_min, y_min, x_max, y_max = bbox[:4]
        x_min = max(int(x_min), 0)
        y_min = max(int(y_min), 0)
        x_max = max(int(x_max), 0)
        y_max = max(int(y_max), 0)
        empty = np.zeros(self.img_size)
        empty[y_min:y_max, x_min:x_max] = 1

        if saliency_map.max() > 1.:
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        mask_bbox = saliency_map * empty
        energy_bbox = np.sum(mask_bbox)
        energy_whole = np.sum(saliency_map)
        if energy_whole == 0:
            self.ebpg = 0
        else:
            self.ebpg = energy_bbox / energy_whole
        
        return self.ebpg

    def drop_conf(self, img, saliency_map, bbox, org_size=None):
        """
        Calculate the information drop and confidence drop after masking with saliency map
        Parameters:
            - img: 3D-array like [H, W, 3]
            - saliency_map: 2D-array in range [0, 1]
            - bbox: Target bounding box to analyze Tensor([xmin, ymin, xmax, ymax, p_obj, p_cls, cls])
            - org_size: The original size of image file got by function os.path.getsize(file_name)
                    If not None, return the information drop score
        Returns: confidence drop score and information drop score.
        """
        mu = np.mean(img)
        h, w, c = img.shape

        ratio = min(self.img_size[0] / h, self.img_size[1] / w)
        resized_img, _ = self.transform(img, None, self.img_size)
        img = torch.from_numpy(resized_img).unsqueeze(0)
        img = img.float()
        resized_img = resized_img.transpose(1, 2, 0)
        
        invert = ((resized_img / 255) * np.dstack([1 - saliency_map]*3)) * 255
        bias = mu * saliency_map
        masked = (invert + bias[:, :, np.newaxis]).astype(np.uint8)
        
        if org_size is not None:
            # Save image in webp format
            save_path = './test_webp'
            im = Image.fromarray(masked)
            im.save(save_path, 'webp')
            save_size = os.path.getsize(save_path)
            self.drop_info_score = (1 - save_size / org_size)

        masked = torch.from_numpy(masked.transpose(2, 0, 1)).unsqueeze(0).float()
        sample_out = self.model(masked.to(self.device))

        sample_box, _ = postprocess(sample_out, num_classes=80, conf_thre=0.25, nms_thre=0.45, class_agnostic=True)

        if sample_box[0] is None:
            return 1.
        sample_box = sample_box[0]

        target_box = bbox[id]
        target_cls = int(bbox[id][6].item())
        target_score = (bbox[id][4] * bbox[id][5]).cpu().item()

        max_score = 0.
        for i, sp in enumerate(sample_box):
            sp_cls = int(sp[6].item())
            sp_score = sp[4] * sp[5]
            if sp_cls != target_cls:
                continue

            iou_score = torchvision.ops.box_iou(sp[:4].unsqueeze(0), target_box[:4].unsqueeze(0)) * sp_score
            iou_score = iou_score.cpu().item()
            if iou_score > max_score:
                max_score = iou_score

        self.drop_conf_score = max(0, target_score - max_score) / target_score
        return self.drop_conf_score, self.drop_info_score


