import keras
import multiprocessing
import argparse
from keras import backend as K

_y = 0


def f(x):
    global _y
    return K.tf.multiply(x,_y).__str__()

def main():
    ap = argparse.ArgumentParser(description="Tests multiprocessing memory consumption")
    ap.add_argument("--processes", required=True, type=int, help="Number of processes to spawn")
    args = vars(ap.parse_args())

    n_processes = args['processes']

    print 'Using Keras version: {}'.format(keras.__version__)
    print 'Using Tensoflow version: {}'.format(K.tf.__version__)
    global _y
    _y = 5
    print 'Creating {} processes'.format(n_processes)
    pool = multiprocessing.Pool(processes=4)  # start 4 worker processes

    results = [pool.apply_async(f, (i,)) for i in range(10000)]
    print [res.get(timeout=5) for res in results]

if __name__ == '__main__':
    main()
