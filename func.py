import matplotlib.pyplot as plt

def plot_history(history, file_name=None):
    # This fuction plots the loss and the validation loss of the trained algorithm
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    # Loss
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training acc (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation acc (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Acc')
    plt.yscale('log', nonposy='clip')
    plt.legend()
    if file_name:
        plt.savefig(file_name)
        plt.clf()
    else:
        plt.show()
    return
