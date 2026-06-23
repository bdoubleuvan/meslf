## Author

* A.S. Markensteijn
* [PhD Thesis] (https://doi.org/10.4233/uuid:044c083a-4eb1-4999-b789-db7150c4c7df)

## Install

To install, run `python setup.py install --user` from the root directory

## Tests

Test are performed by running `python -m pytest.py`. 

Tests include all the functions starting with `test_` or `example_` in the files in the `test` and `examples` directories.

## Examples

Examples are included in the `examples` directory. The name of the file starts with the type of network: 'G', 'E', or 'H' for single-carrier gas, electricity, or heat networks, 'MES' for multi-carrier systems and, for instance, 'EH' for a multi-carrier system consisting of only electricity and heat. The 'xN' indicates the network consists of x nodes per single-carrier network. 'OF' stands for optimal load flow. 


## License

This software is available under an [MIT License]
