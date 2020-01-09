import dlib
import cv2
import os
import glob
import numpy as np
import math

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("source/shape_predictor_5_face_landmarks.dat") # pontos faciais
reconhecimentoFacial = dlib.face_recognition_model_v1("source/dlib_face_recognition_resnet_model_v1.dat") # arquivo treinado para o reconhecimento facial
limiar  = 0.5

index = np.load("source/index.pickle", allow_pickle=True)
descritoresFaciais = np.load("source/descriptors.npy")

for arquivo in glob.glob(os.path.join("test", "*.jpg")): # Varre as pastas do sistema operacional
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem, 2)

    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        listaDescritorFacial = [df for df in descritorFacial]  # Cria uma lista de 128 posições
        npArrayDescritorFacial = np.array(listaDescritorFacial, dtype=np.float64)  # Cria um array do Numpy
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :] # Cria uma coluna extra

        #print("ds {}".format(descritoresFaciais))
        #print(list(descritoresFaciais))
        #print("")
        #print(npArrayDescritorFacial)
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1) # Calcula a distancia euclidiana
        #print("Distancias {}".format(distancias)) # Quanto menor a diferença, mais proximo de ser a face que procuramos

        minimo = np.argmin(distancias)
        #print(minimo)

        distanciaMinima = distancias[minimo]

        if(distanciaMinima <= limiar):
            nome = os.path.split(index[minimo])[1].split(".")[0] # Seta o nome
        else:
            nome = ""

        cv2.putText(imagem, nome, (face.left(), face.top() - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2) # Coloca o nome
        cv2.rectangle(imagem, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)

    cv2.imshow("Face Recognition", imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()