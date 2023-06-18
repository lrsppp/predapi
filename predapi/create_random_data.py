import numpy as np
import argparse


def create_sample(n=500, n_classes=2, n_per_image=1000):
    sample = []
    xedges = np.arange(-10, 11)
    yedges = np.arange(-10, 11)
    for _ in range(n):
        H = np.zeros((20, 20), dtype=int)
        rnd = np.random.choice(np.arange(1, n_classes + 1))
        for _ in range(rnd):
            mean = np.random.uniform(-6, 6)
            sigma = np.random.uniform(0.5, 2)

            x, y = np.random.multivariate_normal(
                [mean, mean], [[sigma, 0], [0, sigma]], size=n_per_image
            ).T
            H_, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            H += np.round(H_).astype(int)

        sample.append([H.T, rnd])
    return sample


def main(args):
    data = create_sample(args.n, args.n_classes, args.n_per_image)
    with open(args.output, "wb") as f:
        np.save(f, np.array(data, dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--n_per_image", type=int, default=500)
    parser.add_argument("--output", type=str, default="data.npy")

    main(parser.parse_args())
