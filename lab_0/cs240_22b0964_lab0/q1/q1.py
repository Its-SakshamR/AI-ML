import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from scipy.stats import expon

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []
    random_num = []

    # TODO: first generate random numbers from the uniform distribution

    if distribution == "exponential":
        random_num = np.random.uniform(0,1,size=num_samples)
        z = kwargs["lambda"]
        samples = expon.ppf(random_num, scale=z)

    if distribution == "cauchy":
        random_num = np.random.uniform(0,1,size=num_samples)
        x = kwargs["peak_x"]
        z = kwargs["gamma"]
        samples = cauchy.ppf(random_num, loc=x, scale=z)

    np.round(samples,4)

    # END TODO
            
    return samples.tolist()


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"

        plt.hist(samples, bins=100)
        plt.savefig('q1_'+distribution+'.png')
        plt.clf()

        # END TODO
