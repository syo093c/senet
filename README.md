# Test of senet and pytorch lighting

1. test1 smaller train dataset
split train data to validation and train, this will cause `train_data` be smaller,
when the data set is not big enough, it will effect performance on test dataset.

So, train the model again with the best settings on full train dataset!


2. test2 adam vs SDG
SDG with moumment and step is not bad, adam overfitting?

3. read timm and mmpretrain
