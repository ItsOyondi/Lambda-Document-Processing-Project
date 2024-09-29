import os
import xml.etree.ElementTree as ET
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image

# Define category mapping for PRImA dataset
CATEGORY_MAPPING = {
    "TextRegion": 0,
    "ImageRegion": 1,
    "TableRegion": 2,
    "SeparatorRegion": 3
}

# Function to load PRImA dataset
def get_prima_dicts(dataset_dir):
    dataset_dicts = []
    img_dir = os.path.join(dataset_dir, "Images")
    xml_dir = os.path.join(dataset_dir, "XML")

    namespace = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}
    
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            try:
                record = {}
                tree = ET.parse(os.path.join(xml_dir, xml_file))
                root = tree.getroot()
                page = root.find('.//pc:Page', namespace)
                if page is None:
                    continue
                
                img_file_name = page.get('imageFilename')
                img_file = os.path.join(img_dir, img_file_name)
                if not os.path.exists(img_file):
                    continue
                
                with Image.open(img_file) as img:
                    width, height = img.size
                
                record["file_name"] = img_file
                record["image_id"] = xml_file
                record["height"] = height
                record["width"] = width
                
                objs = []
                for region in root.findall('.//pc:TextRegion', namespace) + \
                            root.findall('.//pc:ImageRegion', namespace) + \
                            root.findall('.//pc:TableRegion', namespace) + \
                            root.findall('.//pc:SeparatorRegion', namespace):
                    category = region.tag.split('}')[-1]  
                    coords = region.find('.//pc:Coords', namespace)
                    if coords is None:
                        continue
                    
                    points = [(int(pt.get('x')), int(pt.get('y'))) for pt in coords.findall('.//pc:Point', namespace)]
                    if len(points) < 3:
                        continue
                    
                    x_coords, y_coords = zip(*points)
                    obj_dict = {
                        "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": CATEGORY_MAPPING[category],
                        "segmentation": [list(sum(points, ()))],
                        "iscrowd": 0
                    }
                    objs.append(obj_dict)
                
                if objs:
                    record["annotations"] = objs
                    dataset_dicts.append(record)
            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")
    
    return dataset_dicts

# Set dataset path in Kaggle (update this to your dataset location in Kaggle)
dataset_dir = "/kaggle/input/prima-doc/PRImA Layout Analysis Dataset"  # Adjust based on your Kaggle environment

# Register the PRImA dataset
DatasetCatalog.register("prima_train", lambda: get_prima_dicts(dataset_dir))
MetadataCatalog.get("prima_train").set(thing_classes=list(CATEGORY_MAPPING.keys()))

# Verify dataset loading
prima_dataset = DatasetCatalog.get("prima_train")
print(f"Number of items in prima_train dataset: {len(prima_dataset)}")

# Load the X101 model configuration and weights for training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("prima_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

# Set the number of classes to match PRImA
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

# Training parameters
cfg.SOLVER.IMS_PER_BATCH = 2  # Reduce batch size to fit into GPU memory
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate for fine-tuning
cfg.SOLVER.MAX_ITER = 10000    # Set the number of iterations based on dataset size
cfg.SOLVER.STEPS = []  # No learning rate decay
cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoints every 1000 iterations

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Batch size per image (reduce if memory issues)
cfg.MODEL.DEVICE = "cuda"  # Ensure GPU is used

# Set output directory in Kaggle's working directory
cfg.OUTPUT_DIR = "/kaggle/working/train_output_x101"

# Make sure the output directory exists
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Create the trainer and start training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

# Print logs to confirm training is starting
print("Starting training...")

# Start training
trainer.train()

# Confirm that training has completed
print("Training completed. Check the output directory for results.")
