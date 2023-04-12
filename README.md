# Neural network toys (personal experiments)

This was heavily inspired by [Understanding Neural Network Batch Training: A
Tutorial][tutorial], and a surprising amount of the actual code was written by
Copilot. The [iris.csv](./data/iris.csv) file is from the classic [Iris Data
Set][iris], by way of the article.

The gradient descent and backpropagation code has been preety carefully tested,
and any remaining errors are likely errors in my own understanding.

I should consider upgrading this to do some or all of the following:

- [x] Use drop-outs to prevent overfitting.
- [x] Recongnize the MNIST digits.
- [ ] Implement ReLU activation functions.
- [ ] Implement CNN layers for image processing.
- [ ] Try to match [the MNIST performance here](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist).
- [ ] Implement actual batch learning, just for fun.

Someday.

[tutorial]: https://visualstudiomagazine.com/articles/2014/08/01/batch-training.aspx
[iris]: https://archive.ics.uci.edu/ml/datasets/iris
