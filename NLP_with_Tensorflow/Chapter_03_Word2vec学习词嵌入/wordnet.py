# coding:utf-8
import nltk
from nltk.corpus import wordnet as wn


# ========================================Various Synset Relationships=====
# car的所有synsets
# synset : 表示一群或一组同义词
word = 'car'
car_syns = wn.synsets(word)
print('All the available Synsets for ',word)
print('\t',car_syns,'\n')

# The definition of the first two synsets
# 每个synset 都有一个definition，用于解释synset表示的内容
syns_defs = [car_syns[i].definition() for i in range(len(car_syns))]
print('Example definitions of available Synsets ...')
for i in range(3):
    print('\t',car_syns[i].name(),': ',syns_defs[i])
print('\n')

# Get the lemmas(同义词) for the first Synset
print('Example lemmas(同义词) for the Synset ',car_syns[i].name())
car_lemmas = car_syns[0].lemmas()[:3]
print('\t',[lemma.name() for lemma in car_lemmas],'\n')

# Let us get hypernyms for a Synset (general superclass)
syn = car_syns[0]
print('Hypernyms(上位词) of the Synset ',syn.name())
print('\t',syn.hypernyms()[0].name(),'\n')

# Let us get hyponyms for a Synset (specific subclass)
syn = car_syns[0]
print('Hyponyms(下位词) of the Synset ',syn.name())
print('\t',[hypo.name() for hypo in syn.hyponyms()[:3]],'\n')

# Let us get part-holonyms for a Synset (specific subclass)
# also there is another holonym category called "substance-holonyms"
syn = car_syns[2]
print('Holonyms (Part)(整体词) of the Synset ',syn.name())
print('\t',[holo.name() for holo in syn.part_holonyms()],'\n')

# Let us get meronyms for a Synset (specific subclass)
# also there is another meronym category called "substance-meronyms"
syn = car_syns[0]
print('Meronyms (Part)(部分词) of the Synset ',syn.name())
print('\t',[mero.name() for mero in syn.part_meronyms()[:3]],'\n')
print("="*20)


# ======================Similarity between Synsets==============
word1, word2, word3 = 'car','lorry','tree'
w1_syns, w2_syns, w3_syns = wn.synsets(word1), wn.synsets(word2), wn.synsets(word3)

print('Word Similarity(相似性) (%s)<->(%s): '%(word1,word2),wn.wup_similarity(w1_syns[0], w2_syns[0]))
print('Word Similarity(相似性) (%s)<->(%s): '%(word1,word3),wn.wup_similarity(w1_syns[0], w3_syns[0]))

'''
All the available Synsets for  car
     [Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), 
     Synset('car.n.04'), Synset('cable_car.n.01')] 

Example definitions of available Synsets ...
     car.n.01 :  a motor vehicle with four wheels; 
                 usually propelled by an internal combustion engine
     car.n.02 :  a wheeled vehicle adapted to the rails of railroad
     car.n.03 :  the compartment that is suspended from an airship 
                 and that carries personnel and the cargo and the power plant


Example lemmas(同义词) for the Synset  car.n.03
     ['car', 'auto', 'automobile'] 

Hypernyms(上位词) of the Synset  car.n.01
     motor_vehicle.n.01 

Hyponyms(下位词) of the Synset  car.n.01
     ['ambulance.n.01', 'beach_wagon.n.01', 'bus.n.04'] 

Holonyms (Part)(整体词) of the Synset  car.n.03
     ['airship.n.01'] 

Meronyms (Part)(部分词) of the Synset  car.n.01
     ['accelerator.n.01', 'air_bag.n.01', 'auto_accessory.n.01'] 

====================
Word Similarity(相似性) (car)<->(lorry):  0.6956521739130435
Word Similarity(相似性) (car)<->(tree):  0.38095238095238093


'''
