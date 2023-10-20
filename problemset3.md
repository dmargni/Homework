# Problem set 3

In this problem set we will use the Flower 102 dataset and load pretrained AlexNet. We will use one image from the data set and apply filters to it.
The notebook can be seen [here](https://colab.research.google.com/drive/1ETB9I7qm8p_siwdpIFzHTf9Ai63XBMhT#scrollTo=qkMqr835N-W2)

## Load Flowers 102 dataset

First we import Flower 102 dataset on our notebook and choose which image we wanr.

### Code

    Load Flower 102 and visualisation

    import matplotlib.pyplot as plt
    def plot(x,title=None):
        # Move tensor to CPU and convert to numpy
        x_np = x.cpu().numpy()

    # If tensor is in (C, H, W) format, transpose to (H, W, C)
    if x_np.shape[0] == 3 or x_np.shape[0] == 1:
        x_np = x_np.transpose(1, 2, 0)

    # If grayscale, squeeze the color channel
    if x_np.shape[2] == 1:
        x_np = x_np.squeeze(2)

    x_np = x_np.clip(0, 1)

    fig, ax = plt.subplots()
    if len(x_np.shape) == 2:  # Grayscale
        im = ax.imshow(x_np, cmap='gray')
    else:
        im = ax.imshow(x_np)
    plt.title(title)
    ax.axis('off')
    fig.set_size_inches(10, 10)
    plt.show()

    # Downloading and extracting the dataset
    # Uncomment the following lines if you are running this in a Jupyter Notebook
    !wget https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
    !wget https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip
    !unzip 'flower_data.zip'

    import torch
    from torchvision import datasets, transforms
    import os
    import pandas as pd

    # Directory and transforms
    data_dir = '/content/flower_data/'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
    dataset_labels = pd.read_csv('Oxford-102_Flower_dataset_labels.txt', header=None)[0].str.replace("'", "").str.strip()

    # Load the dataset into a DataLoader for batching
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Extract the batch of images and labels
    images, labels = next(iter(dataloader))

    print(f"Images tensor shape: {images.shape}")
    print(f"Labels tensor shape: {labels.shape}")
  

    i = 55
    plot(images[i],dataset_labels[i]);
![download](https://github.com/dmargni/Homework/assets/142944606/93fb547f-0ad0-495a-930b-ff49bfb2130c)


## Load AleNet

The next step is to load pretrained AlexNet in our notebook.

### Code

    import torch
    from torchvision import models, transforms
    import requests
    from PIL import Image
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #define alexnet model
    alexnet = models.alexnet(pretrained=True).to(device)
    labels = {int(key):value for (key, value) in requests.get('https://s3.amazonaws.com/mlpipes/pytorch-quick- start/labels.json').json().items()}

    #transform image for use in model
    preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    img = images[i]

    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    img = to_pil(img)

    img_t = preprocess(img).unsqueeze_(0).to(device)

    img_t.shape

    # labels

    #classify the image with alexnet
    scores, class_idx = alexnet(img_t).max(1)
    print('Predicted class:', labels[class_idx.item()])

## Finetuning

We proceed with finetuning, which consists in this case to exchange the last layer of AlexNet pretrained with a new one and match the numbers outputs of the FLower 102. In this case it is trained to achieve an accuracy of 75% or more. for the FLower 102 data set that we are using.

### Code

    w0 = alexnet.features[0].weight.data
    w1 = alexnet.features[3].weight.data
    w2 = alexnet.features[6].weight.data
    w3 = alexnet.features[8].weight.data
    w4 = alexnet.features[10].weight.data
    w5 = alexnet.classifier[1].weight.data
    w6 = alexnet.classifier[4].weight.data
    w7 = alexnet.classifier[6].weight.data

    # Save and Load
    # w = [w0,w1,w2,w3,w4,w5,w6,w7]
    # torch.save(w, 'Hahn_Alex.pt')
    # w = torch.load('Hahn_Alex.pt')
    # [w0,w1,w2,w3,w4,w5,w6,w7] = w
    # [w0,w1,w2,w3,w4,w5,w6,w7] = torch.load('Hahn_Alex.pt')

    img_t.shape,w0.shape

    img_t.shape

    img_t[0,:,:,:].shape

    def scale(img):
        # Normalize the NumPy array to the range [0, 1]
        max_value = img.max()
        min_value = img.min()
       return normalized_array

    def tensor_plot(img_t,index=0):
        numpy_array = img_t[index,:,:,:].cpu().numpy()
        numpy_array_transposed = numpy_array.transpose(1, 2, 0)
        numpy_array_transposed = scale(numpy_array_transposed)
        plt.imshow(numpy_array_transposed)
        plt.show()

    tensor_plot(img_t)

![img1](https://github.com/dmargni/Homework/assets/142944606/d5bdfb7f-f3fd-4c95-a1ac-c21c67825a1e)


## Visualization

To conclude, after the accuracy is found, we show the visualozation results. 

### Code

    # for i in range(64):
    #     tensor_plot(w0,i)
    #     plt.imshow(f0[0,i,:,:].cpu().numpy())
    #     plt.show()

    import torch
    import matplotlib.pyplot as plt

    def plot_feature_maps_with_filters(feature_maps, filters):
        # Remove batch dimension if it exists
        if feature_maps.dim() == 4:
            feature_maps = feature_maps.squeeze(0)

        # Normalize feature maps to [0, 1]
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps.max() - feature_maps.min())

        def add_filter_to_feature_map(filter_tensor, feature_map_tensor):
            # Ensure the feature map is 2D [H, W]
            if feature_map_tensor.dim() > 2:
                feature_map_tensor = feature_map_tensor.squeeze(0)

            # Convert grayscale feature map to RGB by repeating the single channel 3 times
            feature_map_rgb = feature_map_tensor.unsqueeze(0).repeat((3, 1, 1))

            # Normalize the filter to [0, 1]
            filter_tensor = (filter_tensor - filter_tensor.min()) / (filter_tensor.max() - filter_tensor.min())

            # Ensure the filter fits into the feature map
            min_dim = min(feature_map_tensor.shape)
            filter_size = min(filter_tensor.shape[-1], min_dim)
 
            # Crop the filter if needed
            filter_cropped = filter_tensor[:, :filter_size, :filter_size]

            # Overlay the RGB filter at the lower-left corner of the feature map
            feature_map_rgb[:, -filter_size:, :filter_size] = filter_cropped
  
            # Clip the values to be in the range [0, 1]
            feature_map_rgb = torch.clamp(feature_map_rgb, 0, 1)

            return feature_map_rgb

        # Plot montage of feature maps
        fig, axes = plt.subplots(8, 8, figsize=(15, 15))
  
        for ax, feature_map, filter_ in zip(axes.flat, feature_maps, filters):
            # Add RGB filter to grayscale feature map
            modified_feature_map = add_filter_to_feature_map(filter_, feature_map)

            # Plot modified feature map
            ax.imshow(modified_feature_map.permute(1, 2, 0).cpu().numpy(), interpolation='none')  # Added 'none' interpolation
            ax.axis('off')

        plt.show()



    f0.shape,w0.shape

    plot_feature_maps_with_filters(f0, w0)

![image](https://github.com/dmargni/Homework/assets/142944606/ffbe3094-8bee-46ee-b3e1-a2ab48a28f7f)






