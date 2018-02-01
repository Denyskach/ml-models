import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_learning_curve(path):
    training_df = pd.read_csv(path)
    
    plt.subplot(211) # #row, cols, figure num
    plt.plot(training_df["acc"].values, 'r', training_df["val_acc"].values, 'b')
    training_patch = mpatches.Patch(color='red', label='training')
    validation_patch = mpatches.Patch(color='blue', label='validation')

    plt.legend(handles=[training_patch, validation_patch])

    plt.ylabel('accuracy')
    plt.xlabel('# epoch')

    plt.subplot(212)
    plt.plot(training_df["loss"].values, 'r', training_df["val_loss"].values, 'b')
    plt.ylabel('loss')
    plt.xlabel('# epoch')


    plt.show()