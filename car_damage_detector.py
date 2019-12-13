# Authors: Car Damage Mask R-CNN Group
# Date: 12-02-19
# Class: CSCI 4202-001 -- Intro to Artificial Intelligence

# Imports bundled/trivially installable with Anaconda
import argparse
import re
import os
import numpy as np
import json
import math
import random

# Imports from external sources (conda or git)
import torch
import torch.utils.data
import torchvision
from PIL import Image, ImageFont, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ToTensor
# Imports from PyTorch references as utility functions
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# Extended from the Torchvision Finetuning Instance Segmentation Tutorial
# ----------------
class CarDamageDataset(torch.utils.data.Dataset):
    def __init__(self, root, imgs, masks, transforms=None):
        self.root = root
        # Each element in imgs must match 1:1 with the mask at the corresponding index
        self.imgs = imgs
        self.masks = masks
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        #print(img_path)
        mask_path = self.masks[idx]
        #print(mask_path)
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            #print([xmin, ymin, xmax, ymax])
            # If xmin >= xmax or ymin >= ymax, 
            #   the box is invalid as there is technically no area & the mask attempts to be skipped
            # This should not happen with a properly pre-parsed dataset.
            if (xmin < xmax and ymin < ymax):
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        #print("This is the length:" + str(len(self.imgs)))
        return len(self.imgs)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
# ----------------
    
parent_parser = argparse.ArgumentParser(prog='car_damage_detector.py')
subparsers = parent_parser.add_subparsers(help='Subcommands', dest='subparser')

parser_train = subparsers.add_parser('train', help="fine-tunes the Resnet50 (COCO) dataset using desired training data & saves the weights to the desired path. \
                                     Requires json_path, root_folder, image_folder_name, and save_path")
parser_train.add_argument("json_path", type=str, help="Path to VGG JSON-formatted polygon masks")
parser_train.add_argument("root_folder", type=str, help="Path to root folder containing the images folder (and masks folder if it exists) -- must have trailing slash")
parser_train.add_argument("image_folder_name", type=str, help="Name of folder containing the images referenced in JSON (currently, all images must be in the same folder)")
parser_train.add_argument("save_path", type=str, help="Path where weights of model are saved")

parser_load = subparsers.add_parser('load', help="loads desired weights for a pretrained model & applies a mask to an input image \
                                    Requires checkpoint_path and image_path")
parser_load.add_argument("checkpoint_path", type=str, help="Path to checkpoint for pre-trained model + optimizer")
parser_load.add_argument("image_path", type=str, help="Path to image which will be analyzed")

args = parent_parser.parse_args()
mode = args.subparser

if (mode == "train"):
    json_path = args.json_path
    print("Path to JSON file: " + json_path)
    
    root_folder = args.root_folder
    print("Path to picture folder for training: " + root_folder)
    
    image_folder_name = args.image_folder_name
    print("Name of image folder: " + image_folder_name)
    
    save_path = args.save_path
    print("Path to save the trained model: " + save_path)
    
    if not os.path.exists(root_folder+"masks/"):
        try:
            os.mkdir(root_folder+"masks/")
        except OSError:
            print("ERROR: Could not create masks folder in provided root folder.")
            raise
    
    if not os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except OSError:
            print("ERROR: Could not make save path.")
            raise
            
    # Each element is [filename, path+filename]
    parsed_img_list = [] # Contains all images from the JSON without validating polygons
    validated_img_list = [] # Contains all images from the JSON after validating polygons
    
    # Each element is a path+filename to a mask image (we must generate them if they do not exist)
    validated_mask_img_list = []
    
    # Stores <key, value> pairs of <image name, [[type of region, x vertices, y vertices]]>
    parsed_mask_polygon_dict = {}
    
    if ((type(json_path) is str) and json_path != ""):
        try:
            with open(json_path) as f:
                data = None
                try:
                    data = json.load(f)
                except:
                    raise Exception("Unable to load JSON")
                #print(data)
                metadata_dict = data["_via_img_metadata"]
                # Each value is a subdictionary containing metadata for a specific image            
                for key, value in metadata_dict.items():
                    current_filename = value['filename']
                    mask_regions_list = value['regions']
                    parsed_region_list = []
                    for region in mask_regions_list:
                        # Each element of the list is a dictionary containing parameters of a given region
                        # Skipping images with only regions defined as 'car' for now...
                        valid_region_identifier = False
                        try:
                            # Identifier from our handcreated dataset (platform.ai + VGG image annotator)
                            if (region['region_attributes']['Type'] and region['region_attributes']['Type'].casefold() != "car".casefold()):
                                valid_region_identifier = True
                        except:
                            pass
                        try:
                            # Identifier from the Nicolas Metallo car damage dataset + VGG image annotator
                            if (region['region_attributes']['name']):
                                valid_region_identifier = True
                        except:
                            pass
                        if (valid_region_identifier):
                            parsed_region_list.append(["damage", region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']])
                    if (len(parsed_region_list) != 0):
                        parsed_img_list.append([current_filename, root_folder + image_folder_name + "/" + current_filename])
                        parsed_mask_polygon_dict[current_filename] = parsed_region_list
        except:
            raise
    
    # Create a new image encoding the data of the polygon as a mask
    for image_data in parsed_img_list:
        filename_with_extension = image_data[0]
        # Regex to remove extension
        filename_without_extension = re.sub('((\.jpg$)|(\.JPEG$))', '', filename_with_extension)
        #print(filename_with_extension)
        #print(filename_without_extension)
        polygon_data = parsed_mask_polygon_dict[filename_with_extension]
        #print(polygon_data)
        # Each element is a list containing [region name, [xi, yi]]
        polygon_coords_xy = []
        for polygon in polygon_data:
            polygon_name = polygon[0]
            polygon_x = polygon[1]
            #print(polygon_name)
            polygon_y = polygon[2]
            min_x = min(polygon_x)
            max_x = max(polygon_x)
            min_y = min(polygon_y)
            max_y = max(polygon_y)
            if (min_x < max_x and min_y < max_y):   
                polygon_xy = polygon_x + polygon_y
                # Replaces each even-indexed element with the x-coordinate
                polygon_xy[::2] = polygon_x
                # Replaces each odd-indexed element with the y-coordinate
                polygon_xy[1::2] = polygon_y
                polygon_coords_xy.append([polygon_name, polygon_xy])
        if (len(polygon_coords_xy) == 0):
            print("WARNING: No valid polygonal masks exist for image " + filename_with_extension + " -- Ignoring image.")
        else:                
            image = Image.open(image_data[1])
            mask_image = Image.new('L', image.size, 0)
            color = 1
            for polygon in polygon_coords_xy:
                ImageDraw.Draw(mask_image).polygon(polygon[1], outline=color, fill=color)
                color += 1
            mask_image_save_location = root_folder+"masks/"+filename_without_extension+"_mask.jpg"
            mask_image.save(mask_image_save_location)
            validated_mask_img_list.append(mask_image_save_location)
            validated_img_list.append(image_data)
    
    # From research into other implementations, there is a common standard to reserve ~10% of the dataset for validation.
    # We're interested in the results with different divisions of testing vs. validation
    dataset = CarDamageDataset(root_folder, list(np.array(validated_img_list)[:,1]), validated_mask_img_list, transforms=get_transform(train=True))
    dataset_test = CarDamageDataset(root_folder, list(np.array(validated_img_list)[:,1]), validated_mask_img_list, transforms=get_transform(train=False))
    
    print("Total amount of data in dataset: " + str(len(dataset)))
    
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    number_of_indices_reserved_for_validation = math.floor(len(dataset)*0.5)
    
    dataset = torch.utils.data.Subset(dataset, indices[:-number_of_indices_reserved_for_validation])
    print("Amount of data in training dataset: " + str(len(dataset)))
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-number_of_indices_reserved_for_validation:])
    print("Amount of data in validation dataset: " + str(len(dataset_test)))
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    
    # our dataset has two classes only - background and car damage
    num_classes = 2
    
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    # let's train it for 10 epochs
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    
    torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, save_path+"checkpoint.pt")
    
elif (mode == "load"):
    checkpoint_path = args.checkpoint_path
    print("Path to checkpoint file: " + checkpoint_path)
    
    image_path = args.image_path
    print("Path to image for analysis: " + image_path)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # our dataset has two classes only - background and car damage
    num_classes = 2
    
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # move model to the right device
    model.to(device)
    
    # We must convert a PIL image into a tensor for evaluation
    img = Image.open(image_path).convert("RGB")
    img_tensor = ToTensor()(img)
    
    model.eval()
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
        #print(prediction)
        rescaled_img_array = img_tensor.mul(255).permute(1, 2, 0).byte().numpy()
        rescaled_img = Image.fromarray(rescaled_img_array).convert("RGB")
        
        # NOTE: Banding will occur by process of converting floating point values into bytes (clamped to 255)
        # Future implementation would benefit from conserving precision of segmentation mask values, which would correspond to a more precise image composition
        segmentation_mask_array = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

        random.seed()
        # Generates a random color for a given mask
        random_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        mask_color = Image.new('RGB', rescaled_img.size, random_rgb)
        segmentation_mask_img = Image.fromarray(segmentation_mask_array).convert("L")
        segmentation_mask_img.save("segmentation_mask.png")
        
        composite_img = Image.composite(mask_color, rescaled_img, segmentation_mask_img)
        
        text_shadow_color = "black"
    
        # Image indices of maximum located damage in the mask
        max_xy = np.where(segmentation_mask_array == np.amax(segmentation_mask_array))
        max_x = np.min(max_xy[0])
        max_y = np.min(max_xy[1])
        
        # Font size of 15 with an image of ~260 W is comfortable to read.
        base_font_legibility_img_width = 260.0
        optimal_font_scale = rescaled_img.size[0] / base_font_legibility_img_width
        
        font = ImageFont.truetype("arial.ttf", math.ceil(15*optimal_font_scale))
        text = "DAMAGE -- max conf:\n" + str(segmentation_mask_array[max_x][max_y]/255.0)
        
        draw_coords = (10, 10)
        
        composite_draw = ImageDraw.Draw(composite_img)
        
        # Drawing shadow behind text to improve legibility
        composite_draw.multiline_text((draw_coords[0]-1, draw_coords[1]), text, font=font, fill=text_shadow_color)
        composite_draw.multiline_text((draw_coords[0]+1, draw_coords[1]), text, font=font, fill=text_shadow_color)
        composite_draw.multiline_text((draw_coords[0], draw_coords[1]-1), text, font=font, fill=text_shadow_color)
        composite_draw.multiline_text((draw_coords[0], draw_coords[1]+1), text, font=font, fill=text_shadow_color)
        
        composite_draw.multiline_text((draw_coords[0]-1, draw_coords[1]-1), text, font=font, fill=text_shadow_color)
        composite_draw.multiline_text((draw_coords[0]+1, draw_coords[1]-1), text, font=font, fill=text_shadow_color)
        composite_draw.multiline_text((draw_coords[0]-1, draw_coords[1]+1), text, font=font, fill=text_shadow_color)
        composite_draw.multiline_text((draw_coords[0]+1, draw_coords[1]+1), text, font=font, fill=text_shadow_color)
        
        composite_draw.text(draw_coords, text, font=font, fill="white")
        
        composite_img.save("composite.jpg")
        
        print("Output files saved in the current operating directory.")
        