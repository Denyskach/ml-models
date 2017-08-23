import matplotlib.pyplot as plt                        
import matplotlib.patches as mpatches
import numpy as np

from vis.utils.utils import find_layer_idx
from vis.utils.utils import draw_text
from vis.visualization import visualize_activation

def visualize_layer(model, name, cols_per_row=2, filters=None):
    # name - name of the layer we want to visualize
    layer_idx = find_layer_idx(model, name)

    if filters is None:
        # Visualize all filters in this layer.
        filters = np.arange(get_num_filters(model.layers[layer_idx]))

    plt.axis('off')
    
    i = cols_per_row
    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        # add new area for each cols_per_row new images 
        if i % cols_per_row == 0:
            fgr = plt.figure()
        i+=1
        
        plt.subplot(1, cols_per_row, i % cols_per_row + 1)
        img = visualize_activation(model, layer_idx, filter_indices=idx)
        
        img = draw_text(img, 'Filter {}'.format(idx))
        plt.imshow(img)

    # Generate stitched image palette with columns_per_row cols.
    plt.title(name)
    plt.show()
    
def learning_plot():
    loss = []
    acc = []
    val_loss = []
    val_acc = []
    for line in open("saved_models/dogs_image_augmentation_52_epoch.txt", "r"):
        loss.append(line[56:63])
        acc.append(line[71:77])
        val_loss.append(line[90:97])
        val_acc.append(line[109:115])
    
    plt.subplot(2, 1, 1)
    plt.plot(loss, 'r', val_loss, 'b')
    plt.ylabel('loss')

    train_patch = mpatches.Patch(color='red', label='train')
    valid_patch = mpatches.Patch(color='blue', label='valid')
    plt.legend(handles=[train_patch, valid_patch])

    plt.subplot(2, 1, 2)
    plt.plot(acc, 'r', val_acc, 'b')
    plt.ylabel('accuracy')
    plt.xlabel('# epoch')
    plt.show()