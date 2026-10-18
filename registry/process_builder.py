from abs_interface import AbsAlgorithmBuilder, AbstractHandler

from .algorithms import SegmentationAlgorithm
from .registers import (
    evaluate_segmentation_builder_register,
    evaluate_segmentation_handler_register,
    training_segmentation_builder_register,
    training_segmentation_handler_register,
)


@training_segmentation_builder_register.register("training_segmentation")
class TrainingSegmentation(AbsAlgorithmBuilder):
    """
    The Builder for Object Measurement with Road Lane Detection
    """

    def __init__(self, cfg: dict) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        super().__init__(cfg)
        self.reset()

    def reset(self) -> SegmentationAlgorithm:
        """
        The function `reset` initializes the `_product` attribute with a new instance of
        `ObjMeasurementAlgorithm`.
        """
        self._product = SegmentationAlgorithm()

    @property
    def product(self) -> SegmentationAlgorithm:
        """
        The function returns the product of the measurements and then resets the object.

        Returns:
          The `product` variable is being returned.
        """
        product = self._product
        self.reset()
        return product

    def produce_pre_processor(self) -> None:
        """
        Create pre-processing pipeline
        """
        # empty pipeline
        ...

    def produce_main_processor(self) -> None:
        """
        Create the main processing pipeline for Lane Based Object Measurement.

        This function creates the main processing pipeline by iterating over the
        'mainprocessing' section of the algorithm_pipeline configuration. For each
        handler specified in the configuration, it checks if the handler is registered
        with the handler_register. If the handler is registered, it creates an instance
        of the handler and adds it to the list of handlers. If the handler is not
        registered, it raises a ValueError.

        After creating all the handlers, it creates a pipeline by setting the next
        handler for each handler in the list. Finally, it sets the main processor of
        the product to the first handler in the list.

        """
        # Create a list to store the handlers
        handlers = []

        # Iterate over the mainprocessing section of the configuration
        for handler_name, param in self._cfg["pipeline"].items():

            # Check if the handler is registered
            if training_segmentation_handler_register.check(handler_name):

                # If parameters are provided for the handler, create an instance
                # with the parameters. Otherwise, create an instance without
                # parameters.
                if param:
                    handlers.append(
                        training_segmentation_handler_register.get(handler_name, **param)
                    )
                else:
                    handlers.append(training_segmentation_handler_register.get(handler_name)()) if callable(training_segmentation_handler_register.get(handler_name)) else handlers.append(training_segmentation_handler_register.get(handler_name))

            # If the handler is not registered, raise an error
            else:
                raise ValueError(f"No such handler: {handler_name}")

        # If there are handlers, create a pipeline
        if handlers:
            for i in range(1, len(handlers)):
                # Set the next handler for each handler in the list
                prev_handler: AbstractHandler = handlers[i - 1]
                cur_handler: AbstractHandler = handlers[i]
                prev_handler.set_next(cur_handler)

            # Set the main processor of the product to the first handler
            self._product.main_processor = handlers[0]

    def produce_post_processor(self) -> None:
        """
        Create pre-processing pipeline
        """
        # empty pipeline
        ...


@evaluate_segmentation_builder_register.register("evaluate_segmentation")
class EvaluateSegmentation(AbsAlgorithmBuilder):
    """
    The Builder for Object Measurement with Road Lane Detection
    """

    def __init__(self, cfg: dict) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        super().__init__(cfg)
        self.reset()

    def reset(self) -> SegmentationAlgorithm:
        """
        The function `reset` initializes the `_product` attribute with a new instance of
        `ObjMeasurementAlgorithm`.
        """
        self._product = SegmentationAlgorithm()

    @property
    def product(self) -> SegmentationAlgorithm:
        """
        The function returns the product of the measurements and then resets the object.

        Returns:
          The `product` variable is being returned.
        """
        product = self._product
        self.reset()
        return product

    def produce_pre_processor(self) -> None:
        """
        Create pre-processing pipeline
        """
        # empty pipeline
        ...

    def produce_main_processor(self) -> None:
        """
        Create the main processing pipeline for Lane Based Object Measurement.

        This function creates the main processing pipeline by iterating over the
        'mainprocessing' section of the algorithm_pipeline configuration. For each
        handler specified in the configuration, it checks if the handler is registered
        with the handler_register. If the handler is registered, it creates an instance
        of the handler and adds it to the list of handlers. If the handler is not
        registered, it raises a ValueError.

        After creating all the handlers, it creates a pipeline by setting the next
        handler for each handler in the list. Finally, it sets the main processor of
        the product to the first handler in the list.

        """
        # Create a list to store the handlers
        handlers = []

        # Iterate over the mainprocessing section of the configuration
        for handler_name, param in self._cfg["pipeline"].items():

            # Check if the handler is registered
            if evaluate_segmentation_handler_register.check(handler_name):

                # If parameters are provided for the handler, create an instance
                # with the parameters. Otherwise, create an instance without
                # parameters.
                if param:
                    handlers.append(
                        evaluate_segmentation_handler_register.get(handler_name, **param)
                    )
                else:
                    handlers.append(evaluate_segmentation_handler_register.get(handler_name)()) if callable(evaluate_segmentation_handler_register.get(handler_name)) else handlers.append(evaluate_segmentation_handler_register.get(handler_name))

            # If the handler is not registered, raise an error
            else:
                raise ValueError(f"No such handler: {handler_name}")

        # If there are handlers, create a pipeline
        if handlers:
            for i in range(1, len(handlers)):
                # Set the next handler for each handler in the list
                prev_handler: AbstractHandler = handlers[i - 1]
                cur_handler: AbstractHandler = handlers[i]
                prev_handler.set_next(cur_handler)

            # Set the main processor of the product to the first handler
            self._product.main_processor = handlers[0]

    def produce_post_processor(self) -> None:
        """
        Create pre-processing pipeline
        """
        # empty pipeline
        ...

