import torch
import cv2
import json
import os
import time # Testing how long things take to run
import sys # For quitting
import argparse # For command line arguments

FRAME_INTERVAL_CHECK_TIME = 2 # How many seconds we should skip in-between reading frames
CONFIDENCE_LEVEL = .4 # How much confidence we have that this object is really a traffic light. In %

def load_model(model_name):
    """
    Loads a custom trained model or 
    a Yolo5 model from pytorch hub if no model is specified.
    :param model_name: Path to the trained model
    :return: Trained Pytorch model.
    """
    if model_name:
        model = torch.hub.load('.', 'custom', path=model_name, source='local')
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def grab_image(file_name):
    """
    ...
    """
    if file_name != '.mp4':
        # IDK TBH
        return cv2.imread(file_name)
    else:
        capture = cv2.VideoCapture(file_name)
        for frame in range(1, 4800): # First however many frames are blank
            ret, img = capture.read()
        return img
    \
        
def show_all_images(image, images_to_show):
    """
    Shows all images. Opens up each image in a new window
    """
    for idx in range(len(image)):
        image[idx].show()
        if idx >= images_to_show-1:
            break
    
    
def show_all_colors(data):
    """
    Prints all the colors of each traffic light found
    """
    print("Found ", len(data), " different lights")
    for idx in range(len(data)):
        print('Image ', idx)
        for row in range(len(data[idx].pandas().xyxy[0]['name'])):
            print("Light", row, ": ", data[idx].pandas().xyxy[0]['name'][row])


def process_data():
    
    start = time.time()
    
    # Model
    model = load_model('best.pt')

    # Load our video
    capture = cv2.VideoCapture(input_file)

    # Grab the video info
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Make sure there's actually a video
    print(input_file, capture.get(cv2.CAP_PROP_FPS))
    if (total_frames == 0):
        print("No video provided or no frames found, quitting.");
        return False
    
    # Get some more video information
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_seconds = int(total_frames / fps)
    total_minutes = total_seconds / 60

    # Loop through the whole video, only check every ${3seconds}.
    # Hopefully this time is low enough to catch every stoplight
    frames_with_traffic_lights = []
    for second in range(0, total_seconds, FRAME_INTERVAL_CHECK_TIME):
        frame = int(second * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame) # Set the frame position to the frame we want to read
        ret, img = capture.read()
        
        # Fixup the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Looks like life
        
        # Use the model to see if there are any traffic lights in this frame
        results = model(img)
        
        # Get data about each object found in the frame
        name = results.pandas().xyxy[0]['name'] # Returns a table
        
        # If this frame has a traffic light in it
        if not name.empty:
            
            # Gets the models confidence about the lights in the frame
            confidence = results.pandas().xyxy[0]['confidence'] # Returns a table
            
            # Quick check
            if (confidence[0] < CONFIDENCE_LEVEL):
                continue
            
            # Loop over every traffic light found
            toggler = True # Only want to do some operations once
            # inside_data = [] # Data of each traffic light found
            for idx in range(name.count()):
                
                # Make sure it's probably a traffic light
                if confidence[idx] < CONFIDENCE_LEVEL:
                    continue
                
                # Only save one image
                if toggler:
                    # Save the picture
                    frames_with_traffic_lights.append(results)
                    toggler = False
                    
        # If you don't want it to run for a while
        if second >= 5400:
            break
        
    end = time.time()
    return frames_with_traffic_lights

if __name__ == "__main__":

    # Default values
    show_images = False
    images_to_show = 10
    show_colors = False
    colors_to_show = 0
    input_file = ''

    # Arguments handling
    parser = argparse.ArgumentParser(description='Reads in a video file and outputs the number of red and green lights passed')
    parser.add_argument('-v', '--video_file', type=str, required=True, help='Input video file')
    parser.add_argument('-i', '--images', type=int, metavar='N', help='Number of images to display')
    parser.add_argument('-si', '--show_images', action='store_true', help='Toggles showing images in a new window')
    args, unknown = parser.parse_known_args()
    
    if args.video_file:
        input_file = args.video_file
    
    if args.images:
        images_to_show = args.images
        
    if args.show_images:
        show_images = True
        
    if unknown:
        print('Unknown arguments:', unknown)
        
    # Run the model on the video
    frames_with_traffic_lights = process_data()
    
    if (frames_with_traffic_lights == False):
        sys.exit()
        
    # Parse data and do whatever we want with each frame that has a traffic light
    if show_images:
        show_all_images(frames_with_traffic_lights, images_to_show)
    
    if show_colors:
        show_all_colors(frames_with_traffic_lights)
        
    # Count number of green and blue
    green = 0
    red = 0
    for idx in range(len(frames_with_traffic_lights)):
        for row in range(len(frames_with_traffic_lights[idx].pandas().xyxy[0]['name'])):
            light_color = frames_with_traffic_lights[idx].pandas().xyxy[0]['name'][row]
            if light_color == 'ga':
                green += 1
            elif light_color == 'gf':
                green += 1
            elif light_color == 'r':
                red += 1
            elif light_color == 'rf':
                red += 1
                
    print('\n\nFINISHED')
    print("Green lights hit: ", green);
    print("Red lights hit: ", red);
        
