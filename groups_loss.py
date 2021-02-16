import training
import divide_dataset
import groups_generator


def groups_loss():

    PATHS = ['./mnist_net1.pth', './mnist_net2.pth', './mnist_net3.pth', './mnist_net4.pth', './mnist_net5.pth']

    min_groups = groups_generator.generate_groups()

    (train_d_set, test_d_set) = divide_dataset.divide_dataset(min_groups)

    min_loss = training.train(train_d_set, PATHS[0])

    print(f'Groups: {min_groups}')

    print(f'Loss: {min_loss}')

    for i in range(4):

        groups = groups_generator.generate_groups()

        (train_d_set, test_d_set) = divide_dataset.divide_dataset(groups)

        loss = training.train(train_d_set, PATHS[i+1])

        print(f'Groups: {groups}')

        print(f'Loss: {loss}')

        if loss < min_loss:
            min_loss = loss
            min_groups = groups
            PATH = PATHS[i+1]

    print(f'The best loss: {min_loss}')

    return PATH, min_groups
