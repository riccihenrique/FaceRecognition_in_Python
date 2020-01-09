import dlib
import cv2
import os # Acessa diretorios do SO
import glob # Le arquivos do diretorio
import _pickle as cPickle # Utilizado para salvar o arq de treinamento
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("source/shape_predictor_5_face_landmarks.dat") # pontos faciais
reconhecimentoFacial = dlib.face_recognition_model_v1("source/dlib_face_recognition_resnet_model_v1.dat") # arquivo treinado para o reconhecimento facial

index = {} # armazena o nome do arquivo
i = 0 # indice do dic acima
descritoresFaciais = None # Armazena os descritores faciais

for arquivo in glob.glob(os.path.join("train", "*.jpg")): # Varre as pastas do sistema operacional
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 1) #detecta as faces
    numeroFacesDetectadas = len(facesDetectadas)

    if(numeroFacesDetectadas != 1): #impede a execução com imagens que não tenham 1 face apenas
        print("File incorrect {}".format(arquivo))
        exit(-1)

    for face in facesDetectadas: # Para cada face encontrada
        pontosFaciais = detectorPontos(imagem, face) #Localiza os pontos faciis
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais) # Extrai as caracteristicas principais da imagem

        #print(format(arquivo))
        #print(len(descritorFacial))
        #print(descritorFacial)

        listaDescritorFacial = [df for df in descritorFacial] # Cria uma lista de 128 posições
        #print(listaDescritorFacial)

        npArrayDescritorFacial = np.array(listaDescritorFacial, dtype=np.float64) # Cria um array do Numpy
        #print(npArrayDescritorFacial)

        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :] # Cria uma coluna extra
        #print(npArrayDescritorFacial)

        if descritoresFaciais is None: # Concatena os descritores
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

        index[i] = arquivo
        i += 1

#print("Tamanho {} Formato {}".format(len(descritoresFaciais), descritoresFaciais.shape))
#print(index)

np.save("source/descriptors.npy", descritoresFaciais)
with open("source/index.pickle", "wb") as f:
    cPickle.dump(index, f)

