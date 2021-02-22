import time
import groups_loss


def mnist_cnn():

    start = time.time()

    PATH = ''

    PATH_T = ''

    (PATH, groups, PATH_T, groups_t) = groups_loss.groups_loss()

    finish = time.time()

    print(f'The best groups: {groups}')
    print(f'The best groups for testing: {groups_t}')

    print('Seconds passed: %.3f' % (finish - start))


if __name__ == '__main__':
    mnist_cnn()
