import sys
import training
import divide_dataset
import groups_generator


def groups_loss():

    PATHS = ['./mnist_net1.pth', './mnist_net2.pth', './mnist_net3.pth', './mnist_net4.pth', './mnist_net5.pth']

    min_loss = sys.maxsize * 2 + 1

    min_groups = groups_generator.generate_groups()

    for i in range(5):

        groups = groups_generator.generate_groups()

        (train_d_set, test_d_set) = divide_dataset.divide_dataset(groups)

        loss = training.train(train_d_set, PATHS[i])

        print(f'Groups: {groups}')

        print(f'Loss: {loss}')

        if loss < min_loss:
            min_loss = loss
            min_groups = groups
            PATH = PATHS[i]

    print(f'\nThe minimum loss: {min_loss}')

    return PATH, min_groups
