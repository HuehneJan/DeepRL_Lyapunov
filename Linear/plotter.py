from matplotlib import pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="DDPG_TEST", help="Save directory name in ./data/save_dir")
    parser.add_argument("--reference", type=int, default=1, help="Reference Environment. Either 1 or 2")
    args = parser.parse_args()
    project_dir = "./data/" + args.save_dir
    if args.reference == 1:
        optimal_k1 = -4.9611
        optimal_k2 = -4.4107
    else:
        optimal_k1 = -1.2875
        optimal_k2 = 0.1637
    try:
        k1 = np.load(project_dir + "/k1.npy")
        k2 = np.load(project_dir + "/k2.npy")
        abs_e1 = np.load(project_dir + "/abs_e1.npy")
        abs_e2 = np.load(project_dir + "/abs_e2.npy")
    except Exception:
        raise Exception("Error reading project result files. Make sure they exist in the save directory path!")

    k_1_optimal_diff = np.abs(k1 - optimal_k1)
    k_2_optimal_diff = np.abs(k2 - optimal_k2)

    epochs = np.arange(0, len(k1), step=1).astype(int)
    plt.locator_params(integer=True)
    plt.plot(epochs, k1, "--", label=r"$k_1$")
    plt.plot(epochs, k2, "--", label=r"$k_2$")
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel(r"$k_1, k_2$")
    plt.legend()
    plt.savefig(project_dir + "/controller_values.eps", format="eps")
    plt.show()
    plt.clf()

    plt.locator_params(integer=True)
    plt.plot(epochs, k_1_optimal_diff, "--", label=r"$k_1$")
    plt.plot(epochs, k_2_optimal_diff, "--", label=r"$k_2$")
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel(r"difference to optimal LQR value")
    plt.legend()
    plt.savefig(project_dir + "/optimal_difference.eps", format="eps")
    #plt.show()
    plt.clf()

    plt.locator_params(integer=True)
    plt.plot(epochs, abs_e1, "--", label=r"$|\lambda_1|$")
    plt.ylabel(r"$|\lambda_1|$")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.savefig(project_dir + "/absolute_e1.eps", format="eps")
    #plt.show()
    plt.clf()

    plt.locator_params(integer=True)
    plt.plot(epochs, abs_e1, "--", label=r"$|\lambda_2|$")
    plt.ylabel(r"$|\lambda_2|$")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.savefig(project_dir + "/absolute_e2.eps", format="eps")
    #plt.show()
    plt.clf()



