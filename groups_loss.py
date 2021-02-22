import sys
import training
import testing
import divide_dataset
import groups_generator


def groups_loss():

    PATHS = ['./mnist_net1.pth', './mnist_net2.pth', './mnist_net3.pth', './mnist_net4.pth', './mnist_net5.pth']

    min_loss = sys.maxsize * 2 + 1

    min_loss_t = sys.maxsize * 2 + 1

    min_groups = groups_generator.generate_groups()

    min_groups_t = groups_generator.generate_groups()

    max_accuracy = 0

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

        (loss_t, accuracy) = testing.test(test_d_set, PATHS[i])

        if loss_t < min_loss_t:
            min_loss_t = loss_t
            min_groups_t = groups
            PATH_T = PATHS[i]
            min_accuracy = accuracy

        if accuracy > max_accuracy:
            max_accuracy = accuracy

    print(f'\nThe minimum loss: {min_loss}')
    print(f'\nThe minimum loss for training: {min_loss_t}')
    print(f'\nThe accuracy is: {min_accuracy}')
    print(f'\nThe best accuracy is: {max_accuracy}')

    return PATH, min_groups, PATH_T, min_groups_t
