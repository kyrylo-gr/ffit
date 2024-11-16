# Developer Guidelines

It is not necessary to create a new class for every function, as the `ffit` package provides the `curve_fit` and `leastsq` methods. These methods return a `FitResult` object with similar functionality.

However, if you want to create a new class for a function that offers additional features, you can follow the [guidelines to create a custom class](custom_class.md). This would allow you to later contribute this function to the package.

## Contributing

To contribute to the package, you can create a pull request with your new function class. Ensure your contribution adheres to the provided [guidelines](contribution.md) and includes the necessary tests and documentation. Contributions are reviewed and merged as appropriate.
