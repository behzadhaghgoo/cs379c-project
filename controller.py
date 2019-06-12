import subprocess
from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    pool = Pool(cpu_count())
    print('cpu count', cpu_count())

    # for decision_eps in [.5, 1]:
    #     for mean in [0, -.5, -10]:
    #         for variance in [0, .5, 10]:
    #             for method in ['PER', 'average_over_batch', 'average_over_buffer']:
    for decision_eps in [.5]:
        for mean in [0]:
            for variance in [0]:
                for method in ['PER']:
                    args_list = ['python', 'train.py', 
                                 "--decision_eps", str(decision_eps),
                                 '--mean', str(mean),
                                 '--method', method,
                                 '--variance', str(variance)]

                    print(subprocess.check_call(args_list))

    pool.close()
    pool.join()