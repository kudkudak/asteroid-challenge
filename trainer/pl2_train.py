
import pylearn2
from pl2_dataset import AsteroidDataset
from pylearn2.config import yaml_parse

layer1_yaml = open('pl2_train_layer1.yaml', 'r').read()
hyper_params_l1 = {'train_stop' : 50000,
                'batch_size' : 100,
                'nhid': 40,
                'monitoring_batches' : 5,
                'max_epochs' : 10,
                'save_path' : '.'}
layer1_yaml = layer1_yaml % (hyper_params_l1)



mlp_yaml = open('pl2_train_mlp.yaml', 'r').read()
hyper_params_mlp = {
                    'batch_size' : 100,
                    'max_epochs' : 50,
                    'save_path' : '.'}
mlp_yaml = mlp_yaml % (hyper_params_mlp)
print mlp_yaml



def learn1():
    from pylearn2.config import yaml_parse
    train = yaml_parse.load(layer1_yaml)
    train.main_loop()


def learn2():
    layer2_yaml = open('pl2_train_layer2.yaml', 'r').read()
    hyper_params_l2 = {'train_stop' : 50000,
                    'batch_size' : 100,
                    'monitoring_batches' : 5,
                    'nvis' : hyper_params_l1['nhid'],
                    'nhid' : 500,
                    'max_epochs' : 10,
                    'save_path' : '.'}
    layer2_yaml = layer2_yaml % (hyper_params_l2)
    print layer2_yaml
    train = yaml_parse.load(layer2_yaml)
    train.main_loop()

def learn3():
    train = yaml_parse.load(mlp_yaml)
    train.main_loop()

   

learn3()


