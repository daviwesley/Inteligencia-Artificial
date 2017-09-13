#!usr/bin/python
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#conjunto de dados
ds = SupervisedDataSet(2,1)

#adicionando as amostras
ds.addSample((0.8,0.4),(0.7))
ds.addSample((0.5,0.7),(0.5))
ds.addSample((1.0,0.8),(0.95))

# 2 entradas
# 4 neuronios na camada oculta
# 1 neuronio de saida metodo BIAS
RNA = buildNetwork(2,4,1, bias=True)

treinador = BackpropTrainer(RNA,ds)

for i in xrange(2000):
    print(treinador.train())

while True:
    dormiu = float(raw_input("Dormiu\n"))
    estudou = float(raw_input("Estudou\n"))

    z = RNA.activate((dormiu/10,estudou/10))[0] * 10.0

    print("Previsao da nota",str(z))
