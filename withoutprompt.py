from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# import cv2
from collections import defaultdict
import torchvision.transforms as transforms
import torch
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

# import wandb_handler


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

    def focal_loss(self, logits, gt, gamma=2):
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

def dice_coefficient(logits, gt):
    eps=1
    binary_mask = logits>0
    intersection = (binary_mask * gt).sum(dim=(-2,-1))
    dice_scores = (2.0 * intersection + eps) / (binary_mask.sum(dim=(-2,-1)) + gt.sum(dim=(-2,-1)) + eps)
    
    return dice_scores.mean()

def what_the_f(low_res_masks,label):
            
    low_res_label = F.interpolate(label, low_res_masks.shape[-2:])
    dice = dice_coefficient(
        low_res_masks, low_res_label
    )
    return dice




accumaltive_batch_size = 8
batch_size = 1
num_workers = 2
slice_per_image = 1
num_epochs = 100
sample_size = 2000
# image_size=sam_model.image_encoder.img_size
image_size = 1024
exp_id = 0
found=0
debug = 0

if debug:
    user_input='debug'
else:    
    user_input = input("Related changes: ")
while found == 0:
    try:
        os.makedirs(f"exps/{exp_id}-{user_input}/")
        found = 1
    except:
        exp_id = exp_id + 1
copyfile(os.path.realpath(__file__), f"exps/{exp_id}-{user_input}/code.py")



model_type = "vit_h"
checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
device = "cuda:0"


from segment_anything import SamPredictor, sam_model_registry


# //////////////////
class panc_sam(nn.Module):
    def __init__(self, *args, **kwargs) -> None: 
        super().__init__(*args, **kwargs)
          
        self.sam=sam_model_registry[model_type](checkpoint=checkpoint)
        # self.sam  = torch.load('/mnt/new_drive/PanCanAid/PanCanAid-segmentation/exps/Finetune_on_NIH/sam_tuned_save.pth').sam
        self.prompt_encoder =  self.sam.prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False


        
    def forward(self, image ,box):
        with torch.no_grad():
            image_embedding = self.sam.image_encoder(image).detach()
            
        outputs_prompt = []

        for curr_embedding in image_embedding:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
            
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe().detach(),
                sparse_prompt_embeddings=sparse_embeddings.detach(),
                dense_prompt_embeddings=dense_embeddings.detach(),
                multimask_output=False,
            )
            outputs_prompt.append(low_res_masks)

        low_res_masks_promtp = torch.cat(outputs_prompt, dim=0)
        # raise ValueError(low_res_masks_promtp)

        return low_res_masks_promtp

# ///////////////


augmentation = A.Compose(
    [
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
        A.RandomResizedCrop(1024, 1024, scale=(0.9, 1.0), p=1),
        # A.HorizontalFlip(p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, fill_value=0, p=0.5),
        A.RandomScale(scale_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GridDistortion(p=0.5),
        
    ]
)
panc_sam_instance=panc_sam()


panc_sam_instance.to(device)
panc_sam_instance.train()

train_dataset = PanDataset(
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/train/images"],
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/train/labels"],
    # ["/mnt/new_drive/PanCanAid/Data/NIH_PNG/train/images"],
    # ["/mnt/new_drive/PanCanAid/Data/NIH_PNG/train/labels"],
    [["NIH_PNG",1]],
    
    image_size,
    
    slice_per_image=slice_per_image,
    train=True,
    augmentation=augmentation,
)
test_dataset = PanDataset(
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/test/images"],
    [
     "/media/external_2T/malekahmadi/PanCanAid/Data/NIH_PNG/test/labels"],
        
    [["NIH_PNG",1]],

    image_size,
    
    slice_per_image=slice_per_image,
    train=False,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=train_dataset.collate_fn,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=test_dataset.collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


# Set up the optimizer, hyperparameter tuning will improve performance here
#3e-4
lr = 1e-4
#1e-3
max_lr = 5e-5 #3e-4✅/7e-4✅/1.5e-3/1e-4/✅5e-5
wd = 5e-4#5e-4✅/1e-4✅/1e-3✅/5e-5✅



optimizer = torch.optim.Adam(
    # parameters,
    list(panc_sam_instance.sam.mask_decoder.parameters()),
    # list(panc_sam_instance.mask_decoder.parameters()),
    lr=lr, weight_decay=wd
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=num_epochs,
    steps_per_epoch=sample_size // (accumaltive_batch_size // batch_size),
)



from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

loss_function = loss_fn(alpha=0.5, gamma=2.0)
loss_function.to(device)

from time import time
import time as s_time

log_file = open(f"exps/{exp_id}-{user_input}/log.txt", "a")


def process_model(data_loader, train=0, save_output=0):
    epoch_losses = []

    index = 0
    results = torch.zeros((2, 0, 256, 256))
    total_dice = 0.0
    num_samples = 0

    counterb = 0
    for image, label in tqdm(data_loader):
        counterb += 1
        num_samples += 1
        index += 1
        image = image.to(device)
        label = label.to(device).float()

        input_size = (1024, 1024)

        box = torch.tensor([[200, 200, 750, 800]]).to(device)         
        low_res_masks = panc_sam_instance(image,box)
        low_res_label = F.interpolate(label, low_res_masks.shape[-2:])
        dice = what_the_f(low_res_masks,low_res_label)


        binary_mask = normalize(threshold(low_res_masks, 0.0, 0))

        
        total_dice += dice
        average_dice = total_dice / num_samples
        log_file.write(str(average_dice) + "\n")
        log_file.flush()
        loss = loss_function.forward(low_res_masks, low_res_label)

        loss /= accumaltive_batch_size / batch_size
        if train:
            
            loss.backward()

            if index % (accumaltive_batch_size / batch_size) == 0:
                # print(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                index = 0

        else:
            pass
            # result = torch.cat(
            #     (
            #         low_res_masks[0].detach().cpu().reshape(1, 1, 256, 256),
            #         binary_mask[0].reshape(1, 1, 256, 256),
            #     ),
            #     dim=0,
            # )
            # results = torch.cat((results, result), dim=1)
        if index % (accumaltive_batch_size / batch_size) == 0:
            epoch_losses.append(loss.item())
        if counterb == sample_size and train:
            break
        elif counterb == sample_size / 10  and not train:
            break

    return epoch_losses, results, average_dice

def train_model(train_loader, test_loader, K_fold=False, N_fold=7, epoch_num_start=7):
    print("Train model started.")

    train_losses = []
    train_epochs = []
    test_losses = []
    test_epochs = []
    dice = []
    dice_test = []
    results = []

    index = 0
    # for image, label in tqdm(test_loader):
    #     if index < 100:
    #         if not os.path.exists(f"ims/batch_{index}"):
    #             os.mkdir(f"ims/batch_{index}")

    #         save_img(
    #             image[0],
    #             f"ims/batch_{index}/img_0.png",
    #         )
    #         save_img(0.2*image[0][0] + label[0][0], f"ims/batch_{index}/gt_0.png")

    #     index += 1
    #     if index == 100:
    #         break

    # In each epoch we will train the model and the test it
    # training without k_fold cross validation:
    last_best_dice = 0

    for epoch in range(num_epochs):
        print(f"=====================EPOCH: {epoch + 1}=====================")
        log_file.write(
            f"=====================EPOCH: {epoch + 1}===================\n"
        )
        print("Training:")
        train_epoch_losses, epoch_results, average_dice = process_model(
            train_loader, train=1
        )
        
        dice.append(average_dice)
        train_losses.append(train_epoch_losses)
        if (average_dice) > 0.5:
            print("Testing:")
            test_epoch_losses, epoch_results, average_dice_test = process_model(
                test_loader
            )

            test_losses.append(test_epoch_losses)
            for i in tqdm(range(len(epoch_results[0]))):
                if not os.path.exists(f"ims/batch_{i}"):
                    os.mkdir(f"ims/batch_{i}")

                save_img(epoch_results[0, i].clone(), f"ims/batch_{i}/prob_epoch_{epoch}.png")
                save_img(epoch_results[1, i].clone(), f"ims/batch_{i}/pred_epoch_{epoch}.png")

        train_mean_losses = [mean(x) for x in train_losses]
        # raise ValueError(average_dice)
        test_mean_losses = [mean(x) for x in test_losses]
        np.save("train_losses.npy", train_mean_losses)
        np.save("test_losses.npy", test_mean_losses)

        print(f"Train Dice: {average_dice}")
        print(f"Mean train loss: {mean(train_epoch_losses)}")

        try:
            dice_test.append(average_dice_test)
            print(f"Test Dice : {average_dice_test}")
            print(f"Mean test loss: {mean(test_epoch_losses)}")

            results.append(epoch_results)
            test_epochs.append(epoch)
            train_epochs.append(epoch)
            plt.plot(test_epochs, test_mean_losses, train_epochs, train_mean_losses)
            print("********last_best_dice********")
            print(last_best_dice)

            if average_dice_test > last_best_dice:
                torch.save(panc_sam_instance, f"exps/{exp_id}-{user_input}/sam_tuned_save.pth")

                last_best_dice = average_dice_test
            del epoch_results
            del average_dice_test
        except:
            train_epochs.append(epoch)
            plt.plot(train_epochs, train_mean_losses)
            print(f"=================End of EPOCH: {epoch}==================\n")

        plt.yscale("log")
        plt.title("Mean epoch loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.savefig("result")

    

    return train_losses, test_losses, results


train_losses, test_losses, results = train_model(train_loader, test_loader)
log_file.close()

# train and also test the model
