import torch
import torch.nn as nn
import cnn


def test(test_set, PATH):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = ('0', '1')

    batch_size = 4
    num_workers = 2

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = cnn.Net().to(device)
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():

        epoch_loss = 0.0

        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for l in range(batch_size):
                label = labels[l]
                pred = predicted[l]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    for m in range(2):
        acc = 100 * n_class_correct[m] / n_class_samples[m]
        print('Accuracy of %s: %.3f %%' % (classes[m], acc))

    accuracy = 100 * correct / total
    loss_t = 100 * epoch_loss / i

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (accuracy))
    print('Loss of the network on the 10000 test images: %.3f %%' % (loss_t))

    return loss_t, accuracy
