# Tabular Pipeline 

## Classes

### Explorer

The Explorer class contains various methods for *passively* examining the data.

### DataProcessor

On top of the functionality inherited from the Explorer class, the DataProcessor class can also build features and perform common preprocessing procedures. This does not include basic cleaning tasks.

### Model

The Model class gives various models a standardized interface. This abstracts away a lot of the tediousness of working with many different libraries.

### Modeler

The Modeler class allows us to begin with a standard model algorithm and gradually improve it with feature selection and hyperparameter tuning. There's no need to juggle multiple model instances.

### Ensembler

The Ensembler class takes a list of Modelers and creates a robust ensemble.

# TODO

* seed data processor

* add flexibility for multi-target and classification

* save versus pure memory workflow (auto checkpoints)

# Thoughts

Multi-Model for Multi-Target??

## Regression: multi-target is just multi-model

## Classification: 

## single-column (binary or multi-object)

## multiple columns... 

* if classes do not sum to 1 or there is only one target, use binary:logistic
* if > 2 classes and sum to one, use multi:softmax
* otherwise, for 2 classes summing to 1, use binary:logistic 

* reg:squarederror
* multi:softmax
* binary:logistic
