# Problem set 3

In this problem set we will use the Flower 102 dataset and load preset AlexNet. We will use one image from the data set and apply filters to it.
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
