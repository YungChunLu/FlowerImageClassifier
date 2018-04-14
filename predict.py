import torch, json, argparse
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
pretrained_models = {'resnet18': resnet18, 'resnet50': resnet50}

def load_checkpoint(filename, use_gpu):
    checkpoint = torch.load(filename)
    model = pretrained_models[checkpoint["arch"]]
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(checkpoint['num_ftrs'], checkpoint['num_hidden_units']),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(checkpoint['num_hidden_units'], checkpoint['num_classes']),
        nn.Softmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    use_gpu = use_gpu & checkpoint['use_gpu']
    if use_gpu:
        model.cuda()
    optimizer = optim.SGD(model.fc.parameters(), lr=checkpoint['lr'], momentum=checkpoint['momentum'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.CrossEntropyLoss()
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    # Resize
    img = img.resize((256,256))
    # Crops
    img = img.crop((16, 16, 240, 240))
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(img, dtype=np.float64) / 255
    np_image = (np_image - mean) / std
    return np_image.transpose((2, 0, 1))

def predict(image_path, model, use_gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    if use_gpu:
        inputs = Variable(torch.from_numpy(np.expand_dims(image, axis=0)).float()).cuda()
    else:
        inputs = Variable(torch.from_numpy(np.expand_dims(image, axis=0)).float())
    outputs = model(inputs)
    probs, classes = outputs.topk(5)
    return probs.data.cpu().numpy()[0], [model.class_to_idx[c] for c in classes.data.cpu().numpy()[0]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type = bool, default = True, 
                        help = 'whether using GPU')
    parser.add_argument('--topk', type = int, default = 5, 
                        help = 'print out the top K classes')
    parser.add_argument('--checkpoint_path', type = str, default = "resnet50_checkpoint.pth.tar", 
                        help = 'the path of a checkpoint')
    parser.add_argument('--image_path', type = str, default = "flower_data/test/1/image_06743.jpg", 
                        help = 'the path of a image')
    parser.add_argument('--class_names', type = str, default = "cat_to_name.json", 
                        help = 'the path of a json file containing class names')
    
    in_args = parser.parse_args()
    with open(in_args.class_names, 'r') as f:
        cat_to_name = json.load(f)
    filename, image_path, use_gpu, topk = in_args.checkpoint_path, in_args.image_path, in_args.use_gpu, in_args.topk
    use_gpu = in_args.use_gpu & torch.cuda.is_available()
    model = load_checkpoint(filename, use_gpu)
    probs, classes = predict(image_path, model, use_gpu, topk=topk)
    classnames = [cat_to_name[c] for c in classes]
    for p, c, n in zip(probs, classes, classnames):
        print("{} - {} - {:.2%}".format(c, n, p))

if __name__ == "__main__":
    main()