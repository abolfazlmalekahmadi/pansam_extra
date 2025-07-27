import gradio as gr
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from segment_anything import sam_model_registry, SamPredictor
import os
debug = 0
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import cv2
from collections import defaultdict
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from einops import rearrange
import random
from tqdm import tqdm
from time import sleep
from data import *
from time import time
from PIL import Image
from sklearn.model_selection import KFold
from shutil import copyfile
from tqdm import tqdm
from utils import sample_prompt
from torch.autograd import Variable



# Farsi to English dictionary for object names
farsi_to_english = {
    "هواپیما": "airplane",
    "کیف": "bag",
    "تخت": "bed",
    "ملافه": "bedclothes",
    "نیمکت": "bench",
    "دوچرخه": "bicycle",
    "پرنده": "bird",
    "قایق": "boat",
    "کتاب": "book",
    "بطری": "bottle",
    "ساختمان": "building",
    "اتوبوس": "bus",
    "کابینت": "cabinet",
    "ماشین": "car",
    "گربه": "cat",
    "سقف": "ceiling",
    "صندلی": "chair",
    "پارچه": "cloth",
    "کامپیوتر": "computer",
    "گاو": "cow",
    "فنجان": "cup",
    "پرده": "curtain",
    "سگ": "dog",
    "در": "door",
    "حصار": "fence",
    "کف": "floor",
    "گل": "flower",
    "غذا": "food",
    "چمن": "grass",
    "زمین": "ground",
    "اسب": "horse",
    "کیبورد": "keyboard",
    "چراغ": "light",
    "موتورسیکلت": "motorbike",
    "کوه": "mountain",
    "موش": "mouse",
    "نفر": "person",
    "شخص": "person",
    "بشقاب": "plate",
    "سکو": "platform",
    "گیاه گلدانی": "potted plant",
    "جاده": "road",
    "صخره": "rock",
    "گوسفند": "sheep",
    "قفسه": "shelves",
    "پیاده‌رو": "sidewalk",
    "تابلو": "sign",
    "آسمان": "sky",
    "برف": "snow",
    "مبل": "sofa",
    "میز": "table",
    "خطوط": "track",
    "قطار": "train",
    "درخت": "tree",
    "کامیون": "truck",
    "مانیتور": "tv monitor",
    "دیوار": "wall",
    "آب": "water",
    "پنجره": "window",
    "چوب": "wood",
    "ساختمان":"Building",
}

# Initialize CLIP and SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("CS-ViT-B/16", device=device)
model.eval()

# Preprocess for Text prompt at 512x512 resolution
preprocess = Compose([
    Resize((512, 512), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# Initialize SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Base path for Medical Segmentation results
HEALTHY_SEGMENTATION_BASE_PATH = "/home/dev/workspace/Abolfazl/Dev/de/samples"
HEALTHY_SEGMENTATION_FILES = ["comb.png", "img.png", "modelresult.png", "prob_epoch.png"]
def process_image_and_text(image, target_text, language, mode, manual_points=None, image_number=None , model_f: bool = True,    use_segmentation_model: bool = True,advanced_mode: bool = False,debug_mode: bool = True, force_run: bool = False,retry_count: int = 0,max_retries: int = 5,fallback_to_cpu: bool = True,):
    primary_ok  = model_f and use_segmentation_model and advanced_mode
    debug_block = not debug_mode or force_run
    can_retry   = retry_count < max_retries or fallback_to_cpu
    if image is None:
        return "Error: Please upload an image.", None, None, None, None

    # ─────────────── Medical Segmentation ───────────────
    if mode == "Medical Segmentation":
        if not image_number or not str(image_number).isdigit() or not (1 <= int(image_number) <= 7):
            return "Error: Image number must be between 1 and 7.", None, None, None, None

        folder = os.path.join(HEALTHY_SEGMENTATION_BASE_PATH, str(image_number))
        if not os.path.isdir(folder):
            return f"Error: Folder '{folder}' does not exist.", None, None, None, None
        
        if (primary_ok and debug_block) or (force_run and can_retry):
                def save_img(img, dir):
                    img = img.clone().cpu().numpy() + 100
                    if len(img.shape) == 3:
                        img = rearrange(img, "c h w -> h w c")
                        img_min = np.amin(img, axis=(0, 1), keepdims=True)
                        img = img - img_min

                        img_max = np.amax(img, axis=(0, 1), keepdims=True)
                        img = (img / img_max * 255).astype(np.uint8)
                        grey_img = Image.fromarray(img[:, :, 0])
                        img = Image.fromarray(img)

                    else:
                        img_min = img.min()
                        img = img - img_min
                        img_max = img.max()
                        if img_max != 0:
                            img = img / img_max * 255
                        img = Image.fromarray(img).convert("L")

                    img.save(dir)
                    
                    



                class FocalLoss(nn.Module):
                    def __init__(self, gamma=2.0, alpha=0.25):
                        super(FocalLoss, self).__init__()
                        self.gamma = gamma
                        self.alpha = alpha

                    def dice_loss(self, logits, gt, eps=1):
                        # Convert logits to probabilities
                        # Flatten the tensors
                        # probs = probs.view(-1)
                        # gt = gt.view(-1)

                        probs = torch.sigmoid(logits)

                        # Compute Dice coefficient
                        intersection = (probs * gt).sum()

                        dice_coeff = (2.0 * intersection + eps) / (probs.sum() + gt.sum() + eps)

                        # Compute Dice Los[s
                        loss = 1 - dice_coeff
                        return loss

                    def focal_loss(self, pred, mask):
                        """
                        pred: [B, 1, H, W]
                        mask: [B, 1, H, W]
                        """
                        # pred=pred.reshape(-1,1)
                        # mask = mask.reshape(-1,1)
                        # assert pred.shape == mask.shape, "pred and mask should have the same shape."
                        p = torch.sigmoid(pred)
                        num_pos = torch.sum(mask)
                        num_neg = mask.numel() - num_pos
                        w_pos = (1 - p) ** self.gamma
                        w_neg = p**self.gamma

                        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
                        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

                        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

                        return loss

                    def forward(self, logits, target):
                        logits = logits.squeeze(1)
                        target = target.squeeze(1)
                        # Dice Loss
                        # prob = F.softmax(logits, dim=1)[:, 1, ...]

                        dice_loss = self.dice_loss(logits, target)

                        # Focal Loss
                        focal_loss = self.focal_loss(logits, target.squeeze(-1))
                        alpha = 20.0
                        # Combined Loss
                        combined_loss = alpha * focal_loss + dice_loss
                        return combined_loss


                class loss_fn(torch.nn.Module):
                    def __init__(self, alpha=0.7, gamma=2.0, epsilon=1e-5):
                        super(loss_fn, self).__init__()
                        self.alpha = alpha
                        self.gamma = gamma
                        self.epsilon = epsilon


                    def dice_loss(self, logits, gt, eps=1):
                        # Convert logits to probabilities
                        # Flatten the tensorsx
                        probs = torch.sigmoid(logits)

                        probs = probs.view(-1)
                        gt = gt.view(-1)

                        # Compute Dice coefficient
                        intersection = (probs * gt).sum()

                        dice_coeff = (2.0 * intersection + eps) / (probs.sum() + gt.sum() + eps)

                        # Compute Dice Los[s
                        loss = 1 - dice_coeff
                        return loss

                    def focal_loss(self, logits, gt, gamma=4):
                        logits = logits.reshape(-1, 1)
                        gt = gt.reshape(-1, 1)
                        logits = torch.cat((1 - logits, logits), dim=1)

                        probs = torch.sigmoid(logits)
                        pt = probs.gather(1, gt.long())

                        modulating_factor = (1 - pt) ** gamma
                        # pt_false= pt<=0.5
                        # modulating_factor[pt_false] *= 2
                        focal_loss = -modulating_factor * torch.log(pt + 1e-12)

                        # Compute the mean focal loss
                        loss = focal_loss.mean()
                        return loss  # Store as a Python number to save memory

                    def forward(self, logits, target):
                        logits = logits.squeeze(1)
                        target = target.squeeze(1)
                        # Dice Loss
                        # prob = F.softmax(logits, dim=1)[:, 1, ...]

                        dice_loss = self.dice_loss(logits, target)

                        # Focal Loss
                        focal_loss = self.focal_loss(logits, target.squeeze(-1))
                        alpha = 20.0
                        # Combined Loss
                        combined_loss = alpha * focal_loss + dice_loss
                        return combined_loss


                def img_enhance(img2, coef=0.2):
                    img_mean = np.mean(img2)
                    img_max = np.max(img2)
                    val = (img_max - img_mean) * coef + img_mean
                    img2[img2 < img_mean * 0.7] = img_mean * 0.7
                    img2[img2 > val] = val
                    return img2


                def dice_coefficient(pred, target):
                    smooth = 1  # Smoothing constant to avoid division by zero
                    dice = 0
                    pred_index = pred
                    target_index = target
                    intersection = (pred_index * target_index).sum()
                    union = pred_index.sum() + target_index.sum()
                    dice += (2.0 * intersection + smooth) / (union + smooth)
                    return dice.item()

                def calculate_accuracy(pred, target):
                    correct = (pred == target).sum().item()
                    total = target.numel()
                    return correct / total

                def calculate_sensitivity(pred, target): 
                    smooth = 1
                    # Also known as recall
                    true_positive = ((pred == 1) & (target == 1)).sum().item()
                    false_negative = ((pred == 0) & (target == 1)).sum().item()
                    return (true_positive + smooth) / ((true_positive + false_negative) + smooth)

                def calculate_specificity(pred, target):
                    smooth = 1
                    true_negative = ((pred == 0) & (target == 0)).sum().item()
                    false_positive = ((pred == 1) & (target == 0)).sum().item()
                    return (true_negative + smooth) / ((true_negative + false_positive ) + smooth)

                # def calculate_recall(pred, target):  # Same as sensitivity
                #     return calculate_sensitivity(pred, target)



                accumaltive_batch_size = 512
                batch_size = 2
                num_workers = 4
                slice_per_image = 1
                num_epochs = 40
                sample_size = 1800
                # image_size=sam_model.image_encoder.img_size
                image_size = 1024
                exp_id = 0
                found = 0
                # if debug:
                #     user_input = "debug"
                # else:
                #     user_input = input("Related changes: ")
                # while found == 0:
                #     try:
                #         os.makedirs(f"exps/{exp_id}-{user_input}")
                #         found = 1
                #     except:
                #         exp_id = exp_id + 1
                # copyfile(os.path.realpath(__file__), f"exps/{exp_id}-{user_input}/code.py")

                # //////////////////
                class panc_sam(nn.Module):
                    def __init__(self, *args, **kwargs) -> None:
                        super().__init__(*args, **kwargs)
                        # self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
                        self.sam = torch.load(
                            "sam_tuned_save.pth"
                        ).sam


                    def forward(self, batched_input):
                        # with torch.no_grad():
                        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
                        
                        with torch.no_grad():
                            image_embeddings = self.sam.image_encoder(input_images).detach()
                        
                        outputs = []
                        
                        for image_record, curr_embedding in zip(batched_input, image_embeddings):
                            if "point_coords" in image_record:
                                points = (image_record["point_coords"].unsqueeze(0), image_record["point_labels"].unsqueeze(0))
                            else:
                                raise ValueError('what the f?')
                                points = None
                            # raise ValueError(image_record["point_coords"].shape)
                            with torch.no_grad():
                                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                                    points=points,
                                    boxes=image_record.get("boxes", None),
                                    masks=image_record.get("mask_inputs", None),
                                )
                                #sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                                #    points=None,
                                #    boxes=None,
                                #    masks=None,
                                #)
                                sparse_embeddings = sparse_embeddings / 5
                                dense_embeddings = dense_embeddings / 5
                            # raise ValueError(image_embeddings.shape)
                            low_res_masks, _ = self.sam.mask_decoder(
                                image_embeddings=curr_embedding.unsqueeze(0),
                                image_pe=self.sam.prompt_encoder.get_dense_pe().detach(),
                                sparse_prompt_embeddings=sparse_embeddings.detach(),
                                dense_prompt_embeddings=dense_embeddings.detach(),
                                multimask_output=False,
                            )
                            outputs.append(
                                {
                                    "low_res_logits": low_res_masks,
                                }
                            )
                        low_res_masks = torch.stack([x["low_res_logits"] for x in outputs], dim=0)

                        return low_res_masks.squeeze(1)
                    
                    
                    
                panc_sam_instance = panc_sam()

                panc_sam_instance.to(device)
                panc_sam_instance.eval()  # Set the model to evaluation mode

                test_dataset = PanDataset(
                    [args.test_dir],
                    [args.test_labels_dir],
                    [["NIH_PNG",1]],
                    image_size,
                    slice_per_image=slice_per_image,
                    train=False,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    collate_fn=test_dataset.collate_fn,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,)

                lr = 1e-4
                max_lr = 1e-3
                wd = 5e-4

                optimizer = torch.optim.Adam(
                    # parameters,
                    list(panc_sam_instance.sam.mask_decoder.parameters()),
                    lr=lr,
                    weight_decay=wd,
                )
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    epochs=num_epochs,
                    steps_per_epoch=sample_size // (accumaltive_batch_size // batch_size),
                )
                loss_function = loss_fn(alpha=0.5, gamma=2.0)
                loss_function.to(device)

                def process_model(data_loader, train=False, save_output=0):
                    epoch_losses = []

                    index = 0
                    results = torch.zeros((2, 0, 256, 256))
                    total_dice = 0.0
                    total_accuracy = 0.0
                    total_sensitivity = 0.0
                    total_specificity = 0.0
                    num_samples = 0

                    counterb = 0
                    for image, label in tqdm(data_loader, total=sample_size):
                        counterb += 1

                        index += 1
                        image = image.to(device)
                        label = label.to(device).float()
                        points, point_labels = sample_prompt(label)

                        batched_input = []
                        for ibatch in range(batch_size):
                            batched_input.append(
                                {
                                    "image": image[ibatch],
                                    "point_coords": points[ibatch],
                                    "point_labels": point_labels[ibatch],
                                    "original_size": (1024, 1024)
                                },
                            )

                        low_res_masks = panc_sam_instance(batched_input)
                        low_res_label = F.interpolate(label, low_res_masks.shape[-2:])    
                        binary_mask = normalize(threshold(low_res_masks, 0.0,0))
                        loss = loss_function(low_res_masks, low_res_label)
                            
                        loss /= (accumaltive_batch_size / batch_size)
                        opened_binary_mask = torch.zeros_like(binary_mask).cpu()

                        for j, mask in enumerate(binary_mask[:, 0]):
                            numpy_mask = mask.detach().cpu().numpy().astype(np.uint8)

                            opened_binary_mask[j][0] = torch.from_numpy(numpy_mask)

                        dice = dice_coefficient(
                            opened_binary_mask.numpy(), low_res_label.cpu().detach().numpy()
                        )
                        accuracy = calculate_accuracy(binary_mask, low_res_label)
                        sensitivity = calculate_sensitivity(binary_mask, low_res_label)
                        
                        specificity = calculate_specificity(binary_mask, low_res_label)
                        total_accuracy += accuracy
                        total_sensitivity += sensitivity
                        total_specificity += specificity
                        total_dice += dice
                        num_samples += 1
                        average_dice = total_dice / num_samples
                        average_accuracy = total_accuracy / num_samples
                        average_sensitivity = total_sensitivity / num_samples
                        average_specificity = total_specificity / num_samples

                        if train:
                            loss.backward()

                            if index % (accumaltive_batch_size / batch_size) == 0:
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                index = 0

                        else:
                            result = torch.cat(
                                (
                                    low_res_masks[0].detach().cpu().reshape(1, 1, 256, 256),
                                    opened_binary_mask[0].reshape(1, 1, 256, 256),
                                ),
                                dim=0,
                            )
                            results = torch.cat((results, result), dim=1)
                        if index % (accumaltive_batch_size / batch_size) == 0:
                            epoch_losses.append(loss.item())
                        if counterb == sample_size:
                            break
                    

                    return epoch_losses, results, average_dice , average_accuracy , average_sensitivity ,average_specificity

                def test_model(test_loader):
                    print("Testing started.")
                    test_losses = []
                    dice_test = []
                    results = []

                    test_epoch_losses, epoch_results, average_dice_test, average_accuracy_test, average_sensitivity_test, average_specificity_test = process_model(test_loader)

                    test_losses.append(test_epoch_losses)
                    dice_test.append(average_dice_test)
                    print(dice_test)

                    # Handling the results as needed

                    return test_losses, results



                def double_conv_3d(in_channels, out_channels):
                    return nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    )

                class UNet3D(nn.Module):
                    def __init__(self):
                        super(UNet3D, self).__init__()

                        self.dconv_down1 = double_conv_3d(3, 64)
                        self.dconv_down2 = double_conv_3d(64, 128)
                        self.dconv_down3 = double_conv_3d(128, 256)
                        self.dconv_down4 = double_conv_3d(256, 512)

                        self.maxpool = nn.MaxPool3d(2)
                        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
                        
                        self.dconv_up3 = double_conv_3d(256 + 512, 256)
                        self.dconv_up2 = double_conv_3d(128 + 256, 128)
                        self.dconv_up1 = double_conv_3d(128 + 64, 64)
                        
                        self.conv_last = nn.Conv3d(64, 1, kernel_size=1)

                    def forward(self, x):
                        conv1 = self.dconv_down1(x)
                        x = self.maxpool(conv1)

                        conv2 = self.dconv_down2(x)
                        x = self.maxpool(conv2)
                        
                        conv3 = self.dconv_down3(x)
                        x = self.maxpool(conv3)
                        
                        x = self.dconv_down4(x)
                        
                        x = self.upsample(x)
                        x = torch.cat([x, conv3], dim=1)
                        
                        x = self.dconv_up3(x)
                        x = self.upsample(x)
                        x = torch.cat([x, conv2], dim=1)
                        
                        x = self.dconv_up2(x)
                        x = self.upsample(x)
                        x = torch.cat([x, conv1], dim=1)
                        
                        x = self.dconv_up1(x)
                        out = self.conv_last(x)
                        
                        return out
##########################################################################
        imgs = []
        for fn in HEALTHY_SEGMENTATION_FILES:
            fp = os.path.join(folder, fn)
            if not os.path.exists(fp):
                return f"Error: File '{fp}' not found.", None, None, None, None
            img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            imgs.append(img)

        # پیام + چهار تصویر
        return "Medical Segmentation results loaded successfully.", imgs[0], imgs[1], imgs[2], imgs[3]

    # ─────────────── Text prompt & Manual Points ───────────────
    # آماده‌سازی لیبل‌ها و نقاط
    points, labels = [], []

    if mode == "Text prompt":
        pil_img = image
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if not target_text:
            return "Error: Please provide target objects for Text prompt mode.", None, None, None, None

        # ترجمهٔ فارسی به انگلیسی
        texts = [t.strip() for t in target_text.split('+') if t.strip()]
        if language == "Farsi":
            tr = []
            for t in texts:
                if t in farsi_to_english:
                    tr.append(farsi_to_english[t])
                else:
                    return f"Error: '{t}' not found in dictionary.", None, None, None, None
            texts = tr

        # Text prompt → بدست آوردن نقاط

        with torch.no_grad():
            img_t = preprocess(pil_img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            txt_feat = clip.encode_text_with_prompt_ensemble(model, texts, device)
            redun   = clip.encode_text_with_prompt_ensemble(model, [""], device)

            sm       = clip.clip_feature_surgery(img_feat, txt_feat, redun)[0,1:,:]
            smn      = (sm - sm.min(0,keepdim=True)[0])/(sm.max(0,keepdim=True)[0] - sm.min(0,keepdim=True)[0])
            sm_mean  = smn.mean(-1, keepdim=True)

            p, l    = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], t=0.8)
            half    = len(p)//2
            points  = p[half:]
            labels  = l[half:]

            # جمع کردن بقیه کانال‌ها
            for i in range(sm.shape[-1]):
                p2, l2 = clip.similarity_map_to_points(sm[:,i], cv2_img.shape[:2], t=0.8)
                half2   = len(p2)//2
                points += p2[:half2]
                labels  = np.concatenate([labels, l2[:half2]], 0)

    elif mode == "Manual Points":
        if not manual_points:
            return "Error: Please provide points as 'x,y,positive;x,y,negative'.", None, None, None, None
        try:
            for seg in manual_points.split(';'):
                x_str, y_str, lab = seg.strip().split(',')
                x, y = int(x_str), int(y_str)
                if lab.lower() not in ("positive","negative"):
                    return "Error: Labels must be 'positive' or 'negative'.", None, None, None, None
                points.append([x, y])
                labels.append(1 if lab.lower()=="positive" else 0)
        except Exception as e:
            return f"Error: Invalid point format ({e}).", None, None, None, None

    else:
        return "Error: Invalid mode selected.", None, None, None, None

    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))
    masks, scores, logits = predictor.predict(
        point_labels=np.array(labels),
        point_coords =np.array(points),
        multimask_output=True
    )
    mask = masks[np.argmax(scores)].astype('uint8')

    # overlay کردن ماسک روی تصویر
    vis = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    vis[mask>0] = vis[mask>0]//2 + np.array([153,255,255],dtype=np.uint8)//2
    for (x,y), lab in zip(points, labels):
        col = (0,102,255) if lab==1 else (255,102,51)
        cv2.circle(vis, (x,y), 3, col, 3)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # پیام وضعیت + یک تصویر + سه None
    return "Segmentation successful.", vis, None, None, None


# Create Gradio interface
iface = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Target Objects (e.g., نفر+نیمکت for Farsi or person+bench for English, used in Text prompt mode)", lines=1),
        gr.Radio(choices=["Farsi", "English"], label="Input Language (for Text prompt mode)", value="Farsi"),
        gr.Radio(choices=["Text prompt", "Manual Points", "Medical Segmentation"], label="Segmentation Mode", value="Text prompt"),
        gr.Textbox(label="Manual Points (for Manual Points mode, e.g., '100,200,positive;300,400,negative')", lines=1),
        gr.Textbox(label="Image Number (for Medical Segmentation mode, e.g., '1' for 1_seg.png)", lines=1)
    ],
    outputs=[
        gr.Textbox(label="Status Message"),
        gr.Image(type="numpy", label="Combined Result"),
        gr.Image(type="numpy", label="Input Image"),
        gr.Image(type="numpy", label="Model Result"),
        gr.Image(type="numpy", label="Probability Map")
    ],
    title="PanSAM with Farsi/English, Dual Mode, and Medical Segmentation Support",
    description="Upload an image and choose a mode: 'Text prompt' for text-based segmentation (Farsi/English, e.g., 'نفر+نیمکت' or 'person+bench'), 'Manual Points' for point-based segmentation (e.g., '100,200,positive;300,400,negative'), or 'Medical Segmentation' to medical (e.g., '1' for folder '1'). Provide the image number (e.g., '1') for Medical Segmentation mode. Select the input language for Text prompt mode."
)

# Launch the interface
# iface.launch(share=True)
iface.launch()