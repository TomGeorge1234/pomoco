class Generator(): 
    """
    The generator class is a base class for all data generators.
    It defines the common method `sample` that all generators should implement.
    sample() returns a single trial's worth of data comprising times, latents, spikes in a dict.
    """
    def __init__(self):
        pass

    def sample(self, *args, **kwargs):
        """
        Samples data from the generator.

        Returns:
            dict: A dictionary containing the sampled data.
        """
        raise NotImplementedError("The sample method must be implemented by subclasses.")