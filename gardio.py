
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
    "چوب": "wood"
}

# Initialize CLIP and SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("CS-ViT-B/16", device=device)
model.eval()

# Preprocess for CLIP Surgery at 512x512 resolution
preprocess = Compose([
    Resize((512, 512), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# Initialize SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure this file is in the working directory
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

def process_image_and_text(image, target_text, language, mode, manual_points=None):
    # Convert Gradio image input (PIL Image) to OpenCV format
    if image is None:
        return "Error: Please upload an image."
    
    pil_img = image
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Initialize SAM predictor with the input image
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(pil_img))
    
    if mode == "CLIP Surgery":
        # Process target text (e.g., "نفر+نیمکت" for Farsi or "person+bench" for English)
        if not target_text:
            return "Error: Please provide target objects for CLIP Surgery mode."
        texts = target_text.split('+')
        
        # Translate Farsi to English if language is Farsi
        if language == "Farsi":
            translated_texts = []
            for text in texts:
                text = text.strip()
                if text in farsi_to_english:
                    translated_texts.append(farsi_to_english[text])
                else:
                    return f"Error: '{text}' not found in Farsi-to-English dictionary. Please use supported Farsi terms."
        else:  # English
            translated_texts = [text.strip() for text in texts]
        
        if not translated_texts:
            return "Error: No valid target objects provided."
        
        # CLIP Surgery + SAM processing
        with torch.no_grad():
            # Preprocess image for CLIP
            image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            
            # Encode image features
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode text features
            text_features = clip.encode_text_with_prompt_ensemble(model, translated_texts, device)
            
            # Extract redundant features from an empty string
            redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)
            
            # Apply CLIP feature surgery
            sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]
            sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
            sm_mean = sm_norm.mean(-1, keepdim=True)
            
            # Get points and labels for SAM
            p, l = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], t=0.8)
            num = len(p) // 2
            points = p[num:]  # Negatives in the second half
            labels = [l[num:]]
            for i in range(sm.shape[-1]):
                p, l = clip.similarity_map_to_points(sm[:, i], cv2_img.shape[:2], t=0.8)
                num = len(p) // 2
                points = points + p[:num]  # Positives in first half
                labels.append(l[:num])
            labels = np.concatenate(labels, 0)
    
    elif mode == "Manual Points":
        if not manual_points:
            return "Error: Please provide points in the format 'x,y,positive;x,y,negative' (e.g., '100,200,positive;300,400,negative')."
        
        # Parse manual points (format: x,y,positive;x,y,negative)
        points = []
        labels = []
        try:
            for point in manual_points.split(';'):
                x, y, label = point.strip().split(',')
                x, y = int(x), int(y)
                if label.lower() not in ["positive", "negative"]:
                    return "Error: Labels must be 'positive' or 'negative'."
                points.append([x, y])
                labels.append(1 if label.lower() == "positive" else 0)
        except Exception as e:
            return f"Error: Invalid point format. Use 'x,y,positive;x,y,negative'. Details: {str(e)}"
        
        if not points:
            return "Error: No valid points provided."
    
    else:
        return "Error: Invalid mode selected."
    
    # Inference SAM with points
    masks, scores, logits = predictor.predict(
        point_labels=np.array(labels),
        point_coords=np.array(points),
        multimask_output=True
    )
    mask = masks[np.argmax(scores)]
    mask = mask.astype('uint8')
    
    # Visualize the results
    vis = cv2_img.copy()
    vis[mask > 0] = vis[mask > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
    for i, [x, y] in enumerate(points):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)
    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    
    return vis

# Create Gradio interface
iface = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Target Objects (e.g., نفر+نیمکت for Farsi or person+bench for English, used in CLIP Surgery mode)", lines=1),
        gr.Radio(choices=["Farsi", "English"], label="Input Language (for CLIP Surgery mode)", value="Farsi"),
        gr.Radio(choices=["CLIP Surgery", "Manual Points"], label="Segmentation Mode", value="CLIP Surgery"),
        gr.Textbox(label="Manual Points (for Manual Points mode, e.g., '100,200,positive;300,400,negative')", lines=1)
    ],
    outputs=gr.Image(type="numpy", label="Segmented Output"),
    title="PanSAM with Farsi/English and Dual Mode Support",
    description="Upload an image and choose a mode: 'CLIP Surgery' to segment objects using text (Farsi or English, e.g., 'نفر+نیمکت' or 'person+bench'), or 'Manual Points' to specify points in the format 'x,y,positive;x,y,negative' (e.g., '100,200,positive;300,400,negative'). Select the input language for CLIP Surgery mode."
)

# Launch the interface
# To create a public link, uncomment the following line:
# iface.launch(share=True)
iface.launch()



