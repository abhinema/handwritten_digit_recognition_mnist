"""
To predict digits from JPG/PNg files.
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import fc_model
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
import os, random
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

"""
model_classification =   {"0":"zero",
                          "1":"one",
                          "2":"two",
                          "3":"three",
                          "4":"four",
                          "5":"five",
                          "6":"six",
                          "7":"seven",
                          "8":"eight",
                          "9":"nine"}
"""

model_classification =   {"0":"0",
                          "1":"1",
                          "2":"2",
                          "3":"3",
                          "4":"4",
                          "5":"5",
                          "6":"6",
                          "7":"7",
                          "8":"8",
                          "9":"9"}

def parse_input_args():
    #Read https://pymotw.com/3/argparse/ for more details.
    parser = argparse.ArgumentParser(description='Predict Digits with MNIST Dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',       help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',           help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',         help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',      help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,         help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',              help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',     help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint', type=str, default="./model_checkpoint/checkpoint_e40.pth", help='Path to save Check point')
    parser.add_argument('--input-image', type=str, default="/home/abhinema/Desktop/ml/ocr/testSample/img_1.jpg", help='Path to save Check point')
    parser.add_argument('--image_similar_to_mnist', action="store_true", default=False,             help='use for images similarm to MNIST Dataset format')
    parser.add_argument('--debug', action="store_true", default=False,           help='use for to print debug logs')

    return parser.parse_args()
#End of Parse_input_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             #checkpoint['hidden_layers']
                             )
    model.load_state_dict(checkpoint['state_dict'])

    return model
#End of load_checkpoint

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
#End of imshow

def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Done: Process a PIL image for use in a PyTorch model
    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # preprocess the image
    img_tensor = preprocess(pil_image)

    # add dimension for batch
    img_tensor.unsqueeze_(0)

    return img_tensor
#End of process_image

def predict(image_path, model, device,learnrate, use_cuda, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learnrate)
    image = image.to(device)

    #to change 2D to 1D
    #image = image.view(1, 784)

    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)

    topk_probs_tensor, topk_idx_tensor = ps.topk(topk)

    return topk_probs_tensor, topk_idx_tensor #probs, classes
#End of predict

"""
Main purpose of this function is to convert roi area of input image to MNIST format and apply prediction algorithm.
"""

def main():
    #Parse input arguments
    args = parse_input_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    if(args.debug == True):
        print("Processor :",device)

    model = load_checkpoint(args.checkpoint)
    #Debug to print Model information
    if(args.debug == True):
        print("\n\nloaded model############################################\n\n", model)
        print("\n############################################\n")

    imgpath1 = args.input_image
    img = cv2.imread(imgpath1,1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # To invert colors of images to mak esimilar to MNIST Dataset
    if(args.image_similar_to_mnist == True):
        img1 = img
        gray = img
    else:
        img1 = cv2.bitwise_not(img)
        # Convert Image to gray
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #To show inverted colour image
    if(args.debug == True):
        plt.imshow(img1)
        plt.show()

    #Apply threshold.
    #ret, thresh = cv2.threshold(gray, 75, 255, 0)
#    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # find contours in images
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]

    for c in zip(contours, hierarchy):

        if c[1][2] > 0 or c[1][3] < 0 :
            x, y, w, h = cv2.boundingRect(c[0])
            #Print x,y,w,h values
            if(args.debug == True):
                print("X:{}, Y:{}, W:{}, H:{}".format(x,y,w,h))

            # draw a green rectangle to visualize the bounding rect
            # Create region of interest
            roi = gray[y:y+h, x:x+w]
            #Convert image to PIL image
            pil_roi = Image.fromarray(roi)
            # creates white canvas of 28x28 pixels
            newImage = Image.new('L', (28, 28), (0))

            #This is resize with maintaining aspect ratio
            if w > h:  # check which dimension is bigger
                nheight = int(round((20.0 / w * h), 0))  # resize height according to ratio width
                if (nheight == 0):  # rare case but minimum is 1 pixel
                    nheight = 1
                # resize and sharpen
                img2 = pil_roi.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
                newImage.paste(img2, (4, wtop))  # paste resized image on white canvas
            else:
                # Height is bigger. Heigth becomes 20 pixels.
                nwidth = int(round((20.0 / h * w), 0))  # resize width according to ratio height
                if (nwidth == 0):  # rare case but minimum is 1 pixel
                    nwidth = 1
                # resize and sharpen
                img2 = pil_roi.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
                newImage.paste(img2, (wleft, 4))  # paste resized image on white canvas

            #Intermediate images per rectangular box can be saved for debugging purpose.
            #newImage.save('Sample.jpg')

            #Define number of top predictions to get, usually 1.
            top_k = 1
            probs_tensor, classes_tensor = predict(newImage, model, device,args.lr,use_cuda,topk=top_k)

            # Convert the probabilities and classes tensors into lists
            probs = probs_tensor.tolist()[0]

            model.class_to_idx = model_classification
            classes = [model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (classes_tensor).tolist()[0]]

            print("Number :{}, Probability :{}".format(classes,probs))
            #Create rectangular box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #Put text with predicted number
            cv2.putText(img,str(classes),(x+w-10, y+h+5), font, 2,(255,0,0),2,cv2.LINE_AA)
            plt.imshow(img)
    #Matplot lib show
    plt.show()
#End of Main()

if __name__ == '__main__':
    main()
#end of file
