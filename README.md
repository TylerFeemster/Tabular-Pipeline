# Tabular Pipeline 

## Classes

### Explorer

The Explorer class contains various methods for *passively* examining the data.

### DataProcessor

On top of the functionality inherited from the Explorer class, the DataProcessor class can also build features and perform common preprocessing procedures.

### Model

The Model class gives various models a standardized interface. This abstracts away a lot of the tediousness of working with many different libraries.

### Modeler

The Modeler class allows us to begin with a standard model algorithm and gradually improve it with feature selection and hyperparameter tuning. There's no need to juggle multiple model instances.

### Ensembler

The Ensembler class takes a list of Modelers and creates a robust ensemble.

# TODO

* add flexibility for multi-target and classification
* ^^^  use 1d array for multiclass with lgb: or default to this and use 2d otherwise

* save versus pure memory workflow (auto checkpoints)
* examine predictiveness of target