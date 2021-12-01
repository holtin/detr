import torch, torchvision
import torchvision.transforms as T
import math
import requests
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from util.plot_utils import plot_logs
from pathlib import Path
import datetime
import time

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs,threshold=0.7):
  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  probas_to_keep = probas[keep]
  
  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  return probas_to_keep, bboxes_scaled

def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def run_worflow(my_image, my_model,cls=0.8):
  # mean-std normalize the input image (batch-size: 1)
  
  img = transform(my_image).unsqueeze(0)
    
  start_time = time.time()
  outputs = my_model(img)
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  print('Model running time {}'.format(total_time_str))
  
  for threshold in [cls]:
    
    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                              threshold=threshold)    
    plot_finetuned_results(my_image,probas_to_keep,bboxes_scaled)
    
def plot_finetuned_results2(pil_img, prob=None, boxes=None):
    ratio=2
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    width, height = pil_img.size
    ax = plt.gca()
    colors = COLORS * 100
    i=0;
    if prob is not None and boxes is not None:
      for x in range(1,5):        
        for p, (xmin, ymin, xmax, ymax), c in zip(prob[x], boxes[x].tolist(), colors):
            c=colors[i]
            i=i+1
            xmin = xmin if x%2 == 1 else xmin+width
            xmax = xmax if x%2 == 1 else xmax+width
            ymin = ymin if x <= 2 else ymin+height
            ymax = ymax if x <= 2 else ymax+height

            ax.add_patch(plt.Rectangle((xmin/ratio, ymin/ratio), (xmax - xmin)/ratio, (ymax - ymin)/ratio,
                                      fill=False, color=c, linewidth=3))
            cl = p.argmax()
            
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin/ratio, ymin/ratio, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def run_worflow2(my_image, my_model,cls=0.8):
  outputs={}
  probas_to_keep={}
  bboxes_scaled={}
  width, height = my_image.size
  start_time = time.time()
  for x in range(1,5):
    # Setting the points for cropped image
    left = 0 if x%2 == 1 else width/2
    right = width/2 if x%2 == 1 else width
    top = 0 if x <= 2 else height/2
    bottom = height/2 if x <= 2 else height

    im1 = my_image.crop((left, top, right, bottom))  
    img = transform(im1).unsqueeze(0)
    outputs[x] = my_model(img)
  
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  print('Enhanced Model running time {}'.format(total_time_str))

  for threshold in [cls]:
    for x in range(1,5):
      probas_to_keep[x], bboxes_scaled[x] = filter_bboxes_from_outputs(outputs[x],threshold=threshold)    
    
    
    plot_finetuned_results2(my_image,probas_to_keep,bboxes_scaled)

torch.set_grad_enabled(False);


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
          
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

first_class_index=0
assert(first_class_index in [0, 1])

if first_class_index == 0:

  num_classes = 1
  finetuned_classes = [
      'PolyU',
  ]
else:
  num_classes = 2
  finetuned_classes = [
      'N/A', 'PolyU',
  ]
    
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print("Training log with 300 ecophs")
log_directory=[Path('/content/drive/MyDrive/Learning/Machine Learning/DETR/log300')]

fields_of_interest = (
    'loss',
    'mAP',
    )

plot_logs(log_directory, fields_of_interest)

fields_of_interest = (
    'loss_ce',
    'loss_bbox',
    'loss_giou',
    )

plot_logs(log_directory, fields_of_interest)

fields_of_interest = (
    'class_error',
    'cardinality_error_unscaled',
    )

plot_logs(log_directory,fields_of_interest) 

#======================================load the custom trained model

model = torch.hub.load('facebookresearch/detr','detr_resnet50',pretrained=False,num_classes=1)


checkpoint = torch.load('/content/drive/MyDrive/Learning/Machine Learning/DETR/checkpoint0299.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)

model.eval();

#=====================================
print("Test the trained model with random online images")
url={}
url[0]='http://lh3.ggpht.com/_ju_qrF93NiU/SuUfYNDl6hI/AAAAAAAAAjw/NjUVuoJr56w/s1600/SNV31753.JPG'
url[1]='https://www.polyu.edu.hk/cpa/75thanniversary/catalog/view/theme/default/image/en/common/logo_poly75.jpg'
url[2]='https://www.polyu.edu.hk/-/media/department/home/content/about-polyu/president_2_1080x524.jpg'
url[3]='https://cdn-images-1.medium.com/max/1024/1*pE_y_IrCiPZuHXRnaCp6bw.jpeg' 
url[4]='https://www.polyu.edu.hk/combatcovid19/-/media/department/combatcovid19/media-release/polyu-logo_1200x630.jpg'
url[5]='https://www.polyu.edu.hk/fce/-/media/department/fce/events/2021/polyu-info-day-2021_kv_v2_opt-2.jpg' 
url[6]='https://www40.polyu.edu.hk/hrtd/images/login_building3.jpg'
url[7]='https://sites.google.com/site/historyofhkpolyupaulchan/_/rsrc/1475933135055/history-of-hk-polyu/hkpu-001.jpg'
url[8]='https://blogs.discovery.edu.hk/heo/wp-content/uploads/sites/6/2019/03/university_logo_1-1024x375.jpg'
url[9]='https://www.polyu.edu.hk/ife/corp/cntimgs/gallery/1434/5f8a5c02-23ce-11e2-8985-0018f31181e9_l.jpg'
url[10]='https://fastly.4sqi.net/img/general/2000x1080/37872072_EbrlsEU5a8bfh2q3sSnE6AbKUVcHQWpvCig_zI7lobw.jpg'
url[11]='https://hongkongfp.com/wp-content/uploads/2021/05/polyu-car-1050x591.jpg'

for i in range(len(url)):  
  im = Image.open(requests.get(url[i], stream=True).raw)
  run_worflow(im,model,0.7)

#===========================================
print("Test the trained model and enhanced model with small objects (before and after)")
img={}
img[0]="/content/drive/MyDrive/Learning/Machine Learning/DETR/test1.jpg"
img[1]="/content/drive/MyDrive/Learning/Machine Learning/DETR/test2.jpg"
img[2]="/content/drive/MyDrive/Learning/Machine Learning/DETR/test3.jpg"
img[3]="/content/drive/MyDrive/Learning/Machine Learning/DETR/test4.jpg"
img[4]="/content/drive/MyDrive/Learning/Machine Learning/DETR/test5.jpg"

for x in range(5):
  im = Image.open(img[x])
  run_worflow(im,model,0.7)
  run_worflow2(im,model,0.7)