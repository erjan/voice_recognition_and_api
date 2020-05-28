import sys

if __name__ == '__main__':

    print(sys.argv)
    print('length of sys argv %d' % len(sys.argv))
    if len(sys.argv) < 2 :
        print('taking default file 15.wav')

    else:
        print('taking the given specified file:')
        print(sys.argv[1])
