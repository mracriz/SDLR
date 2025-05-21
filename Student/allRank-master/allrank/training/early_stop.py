from allrank.utils.ltr_logging import get_logger
###from csv import DictWriter
logger = get_logger()


class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = 0.0
        self.best_epoch = 0

    def step(self, current_value, current_epoch, Metrics_Info = None): # Metrics_Info Is Self Defined
        logger.info("Current:{} Best:{}".format(current_value, self.best_value))
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = current_epoch
            """
            # Save
            if Metrics_Info != None and len(Metrics_Info) == 3:
                (file_name, dict_of_elem, field_names) = Metrics_Info
                #append_dict_as_row(config.loss.name + alphanumeric + '.csv', val_metrics, field_names)
                with open(file_name, 'a+', newline='') as write_obj:
                    # Create a writer object from csv module
                    dict_writer = DictWriter(write_obj, fieldnames=field_names, )
                    # Add dictionary as row in the csv
                    if iter == 1:
                        dict_writer.writeheader()
                    dict_writer.writerow(dict_of_elem)
            """

    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience
