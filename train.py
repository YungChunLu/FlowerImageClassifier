import argparse, copy, time, os, torch, gc, shutil
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
pretrained_models = {'resnet18': resnet18, 'resnet50': resnet50}

def train_model(arch, num_hidden_units, lr, momentum, data_dir, use_gpu, num_epochs=25):
    sub_dirs = ['train', 'valid', 'test']
    batch_size = 5
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(5),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    # Load the datasets with ImageFolder
    image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
                    for x in sub_dirs}
    # Using the image datasets and the trainforms to define the dataloaders
    # dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
    #                             num_workers=4, shuffle=True)
    #             for x in sub_dirs}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                num_workers=4, sampler=SubsetRandomSampler(range(5)))
                for x in sub_dirs}
    dataset_sizes = {x: len(image_datasets[x]) for x in sub_dirs}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    model = pretrained_models[arch]
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(num_hidden_units, num_classes),
    )
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                gc.collect()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4%}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4%}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc, optimizer, num_ftrs, class_names, num_classes

def save_checkpoint(state, arch, filename='checkpoint.pth.tar'):
    filename = "{}_{}".format(state["arch"], filename)
    torch.save(state, filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type = bool, default = True, 
                        help = 'whether using GPU')
    parser.add_argument('--dir', type = str, default = "flower_data", 
                        help = 'a directory containing train, validate and test data sets')
    parser.add_argument('--model', type = str, default = "resnet18", 
                        help = 'pretrained model')
    parser.add_argument('--lr', type = float, default = 0.01, 
                        help = 'learning rate') 
    parser.add_argument('--num_hidden_units', type = int, default = 4096, 
                        help = 'number of hidden units')
    parser.add_argument('--num_epochs', type = int, default = 5, 
                        help = 'training epochs')
    in_args = parser.parse_args()

    use_gpu = in_args.use_gpu & torch.cuda.is_available()
    arch, num_epochs, lr, data_dir, num_hidden_units, momentum = in_args.model, in_args.num_epochs, in_args.lr, in_args.dir, in_args.num_hidden_units, 0.9
    model, best_acc, optimizer, num_ftrs, class_names, num_classes = train_model(arch, num_hidden_units, lr, momentum, data_dir, use_gpu, num_epochs=num_epochs)
    save_checkpoint({
        'num_epochs': num_epochs,
        'num_hidden_units': num_hidden_units,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc,
        'class_to_idx': class_names,
        'lr': lr,
        'momentum': momentum,
        'num_ftrs': num_ftrs,
        'num_classes': num_classes,
        'optimizer' : optimizer.state_dict(),
        'use_gpu': use_gpu,
        'data_dir': data_dir
    }, arch)

if __name__ == "__main__":
    main()