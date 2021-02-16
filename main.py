import time
import groups_loss


def mnist_cnn():

    start = time.time()

    PATH = ''

    (PATH, groups) = groups_loss.groups_loss()

    finish = time.time()

    print(f'The best groups: {groups}')
    print(PATH)

    print('Seconds passed: %.3f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn()
