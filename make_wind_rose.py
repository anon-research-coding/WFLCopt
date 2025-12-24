import numpy as np
from windrose import WindroseAxes
import matplotlib.pyplot as plt

def sample_the_wind_rose(bins, centers, densities, speed, N, seed):
    # sample N wind directions, using probability density, and return them along with the corresponding wind speeds.
    rng = np.random.default_rng(seed)
    bins = np.asarray(bins)
    centers = np.asarray(centers)
    densities = np.asarray(densities)

    # Compute probability mass per bin
    bin_widths = bins[:,1] - bins[:,0]
    probs = densities * bin_widths
    probs = probs / probs.sum()

    counts = rng.multinomial(N, probs)

    dirs = np.concatenate([rng.uniform(low=start, high=end, size=c) for (start, end), c in zip(bins, counts) if c > 0])

    speeds = np.full(dirs.shape, float(speed))
    return dirs, speeds

# n_bins must be a divisor of 360
def uniform_wind_rose(n_bins, random_seed):
    # Create a random "wind rose" with "bin_width_deg"-sized bins.
    # The probabilities within each bins is uniform, but the bins are not.

    rng = np.random.default_rng(random_seed)

    #n_bins = 360 // bin_width_deg
    bin_width_deg = 360 // n_bins
    edges = np.arange(0, 360 + bin_width_deg, bin_width_deg)
    print("check the edges of the bins:", edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    print("check the centers of the bins:", centers)



    probs = rng.dirichlet(alpha=np.ones(n_bins))
    print("check the probabilities of the bins sum to 1:", sum(probs[i] for i in range(n_bins)))

    # Tells you how much probability per degree within the bin
    density = probs / bin_width_deg

    bins = [(i, i + bin_width_deg) for i in range(0, 360, bin_width_deg)]

    print("len(centers)")
    print( len(centers) )
    print("len(bins)")
    print( len(bins) )
    print("len(probs)")
    print( len(probs) )
    print("len(density)")
    print( len(density) )

    return bins, centers, probs, density

if __name__ == "__main__":
    #width = 10
    n_bins = 36
    bins, centers, probs, density = uniform_wind_rose(n_bins, 12345)
    wind_speeds = [12 for i in range(len(probs))] # m/s assume its the same, but in theory it could be a vector

    for b, p, d in zip(bins, probs, density):
            print(f"Bin {b}: prob={p:.3f}")

    # apparently this windrose package takes raw data, so we need to sample the bins
    # also the color corresponds to wind speed, so for constant wind speed, we get one color
    wd, ws = sample_the_wind_rose(bins, centers, density, 12, 1000000, 12345)
    print("wd")
    print( wd )
    print("ws")
    print( ws )
    print("len(wd)")
    print( len(wd) )
    print("len(ws)")
    print( len(ws) )

    ax = WindroseAxes.from_ax()
    ax.bar(centers, probs, normed=True, opening=0.8, edgecolor="black")
    #ax.contourf(wd, ws, nsector=len(bins), normed=True, cmap=plt.cm.viridis)
    #ax.contour(wd, ws, nsector=len(bins), normed=True, colors="black")
    plt.show() # I don't know how to get this to use the same bins ... but at least we can visualize them!
