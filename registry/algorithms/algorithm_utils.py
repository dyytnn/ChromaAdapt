
import torch

from abs_interface import AbsAlgorithmProcessHandler, AbstractProcess

__all__ = ["SegmentationAlgorithm"]

class SegmentationAlgorithm(AbstractProcess):
    def __init__(self):
        super().__init__()
        self.pre_processor: AbsAlgorithmProcessHandler = None
        self.main_processor: AbsAlgorithmProcessHandler = None
        self.post_processor: AbsAlgorithmProcessHandler = None

    def _pre_process(self, data):
        if self.pre_processor is not None:
            _data = self.pre_processor.handle(data)
        else:
            _data = data
        return _data

    def _main_process(self, data):
        if self.main_processor is not None:
            _data = self.main_processor.handle(data)
        else:
            _data = data
        return _data

    def _post_process(self, data):
        if self.post_processor is not None:
            _data = self.post_processor.handle(data)
        else:
            _data = data
        return _data
    
def build_dataset_class(source_dataset, dataset_class_name, **kwargs):
    """
    Dynamically builds and returns an instance of a dataset class.

    Args:
        source_dataset: The module or package containing the dataset class.
        dataset_class_name: The name of the dataset class to instantiate.
        **kwargs: Additional keyword arguments to pass to the dataset class constructor.

    Returns:
        An instance of the specified dataset class, initialized with the provided arguments.
    """
    dataset_class = getattr(source_dataset, dataset_class_name)
    return dataset_class(**kwargs)

def simple_accuracy(preds: torch.Tensor, labels: torch.Tensor):
    return (preds == labels).mean()

