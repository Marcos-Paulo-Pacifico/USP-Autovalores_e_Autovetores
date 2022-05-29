import numpy as np
import copy 
import matplotlib.pyplot as plt
import pandas as pd



def QR(A, t, modo,  dtype=np.double):

    c1 = np.array([],dtype=np.double)  #guardar c1, c2, ci nessa ordem
    s1 = np.array([],dtype=np.double)  #guardar s1, s2, si nessa ordem
    b=A.shape        #determinar a dimensão da matriz
    alfan=A[b[0]-1][b[0]-1]                    #calculo dos indices da heuristica                                
    alfanmenos1=A[b[0]-2][b[0]-2]
    betanmenos1=A[b[0]-1][b[0]-2]
    u1 = mi(alfanmenos1, alfan, betanmenos1, t)
    if modo == 7:                                                   # modo 7 equivale a fazer com deslocamneto
        matriz=(copy.deepcopy(A))-(np.identity(b[0])*u1)            #copia a matriz A com deslocamento
    else:
        matriz=(copy.deepcopy(A))                                   #copia a matriz A sem deslocamento
    
    for x in range(0, b[0]-1):
        A=copy.deepcopy(matriz)                
        cx=A[x][x]/np.sqrt(np.square(A[x][x]) + np.square(A[x+1][x]))         #i da formula = x
        sx=-A[x+1][x]/np.sqrt(np.square(A[x][x]) + np.square(A[x+1][x]))       #j da formula = x+1                                                 
        c1=np.append(c1,[cx])                                                  # com k =0,1, .... n-1
        s1=np.append(s1,[sx])
        for k in range(0, b[0]):
            matriz[x][k] = ((c1[x]*A[x][k]) - (s1[x]*A[x+1][k]))                     #bi,k = cai,k − saj,k e 
            matriz[x+1][k] = ((s1[x]*A[x][k]) + (c1[x]*A[x+1][k]))                   # bj,k = sai,k + caj,k 
    return matriz, c1, s1, u1                                                                            

def atualizar(matriz, c, s, u, modo, dtype=np.double):                      ##funcao atualizar obtem as matrizes A(1), V(1), etc

    b=matriz.shape                  #determinar a dimensao da matriz
    aux=copy.deepcopy(matriz)   #copia a matriz 
    
    for x in range(0, b[0]-1):
       matriz=copy.deepcopy(aux)                
       for k in range(0, b[0]):
            aux[k][x] = ((c[x]*matriz[k][x]) - (s[x]*matriz[k][x+1]))                     #bi,k = cai,k − saj,k e 
            aux[k][x+1] = ((s[x]*matriz[k][x]) + (c[x]*matriz[k][x+1]))            # bj,k = sai,k + caj,k 
    if (modo == 7):
        return (aux + u*np.identity(b[0]))
    else:
        return aux                                                                          

def mi(a, b, c, k, dtype=np.double):                                           #cálculo de mi para cada iteracao
    if k == 0:
        return 0.0    
    dk=(a-b)/2
    if dk >= 0:
        return b + dk - (np.sqrt(np.square(dk) + np.square(c)))
    else: 
        return b + dk + (np.sqrt(np.square(dk) + np.square(c)))




###############################################funcao eig retorna tupla com lambda e autovetores de matriz ordem n##############
def eig(A, modo, dtype=np.double): 
    B=copy.deepcopy(A)
    m=A.shape
    autovetoresl= np.identity(m[0])                 #guarda as linhas da matriz V(k) quando o criterio de parada e atingido
    autovetoresc= np.identity(m[0])                 #guarda as colunas da matriz V(k) quando o criterio de parada e atingido
    autovalores= np.array([], dtype=np.double)        
    V=np.identity(m[0])
    t=0                                 #numero de iteracoes para cada matriz de ordem n(usada dentro de lacos e depois descartada)
    erro=0.000001                       #criterio de parada    
    n=0                                #numero de reducoes de ordem da matriz
    it=0                                #numero total de iteracoes
    for n in range (0, m[0]-1):       #algoritmo fornecido
        auxiliar = V.shape
        while abs(B[1][0]) > erro:
            result=QR(B,t, modo)
            B=atualizar(result[0],result[1],result[2], result[3], modo)
            V=atualizar(V,result[1],result[2], result[3], 8)
            t=t+1

        for p in range (0, auxiliar[0]):
            autovetoresl[n][p+n]=V[0][p]
        for p in range (0, auxiliar[0]):
            autovetoresc[p+n][n]=V[p][0]
        autovalores = np.append(autovalores,[B[0][0]])
        if n == m[0]-2:
            autovalores = np.append(autovalores, [B[1][1]])
            
        it=t+it 
        t=0
         
    
        B=np.delete(B, 0, 0)                    #matrizes tem sua ordem diminuida para novas atualizacoes
        B=np.delete(B, 0, 1)
        V=np.delete(V, 0, 0)
        V=np.delete(V, 0, 1)
    

    for i in range(0, m[0]):                                #autovaloresl e autovaloresc sao combinados para formar a matriz com autovalores e vetores
        for j in range(0, m[0]):
            if autovetoresl[i][j] != 0.0: 
                autovetoresc[i][j]=autovetoresl[i][j]

    autovetoresc[m[0]-1][m[0]-1]=V[0][0]

    return autovalores, autovetoresc, it
####################################separa valores fornecidos em array para outros menores ##########################
def plotar(t, array, parametro, cor, dtype=np.double):
    aux=int(parametro)
    array1= np.array([], dtype=np.double) 
    array2= np.array([], dtype=np.double)
    array3= np.array([], dtype=np.double)
    array4= np.array([], dtype=np.double)
    array5= np.array([], dtype=np.double)
    
    
    for i in range(0, aux):
        array1 =np.append(array1,[array[(5*i)]])
     

    for i in range(0, aux):
        array2 =np.append(array2,[array[(1+i*5)]])
        
    for i in range(0, aux):
        array3 =np.append(array3,[array[(2+i*5)]])
        
    for i in range(0, aux):
        array4 =np.append(array4,[array[(3+i*5)]])
           
    for i in range(0, aux):
        array5 =np.append(array5,[array[(4+i*5)]])
    
    
    printag(t, array1, "posição(m)", "Posição da Massa 1 em função do tempo", cor)
    printag(t, array2, "posição(m)", "Posição da Massa 2 em função do tempo", cor)
    printag(t, array3, "posição(m)", "Posição da Massa 3 em função do tempo", cor)
    printag(t, array4, "posição(m)", "Posição da Massa 4 em função do tempo", cor)
    printag(t, array5, "posição(m)", "Posição da Massa 5 em função do tempo", cor)
 #####################################################mesmo funcionamento da funcao anterior, so que trabalha com mais arrays################  
def plotarc(t, array, parametro, cor, dtype=np.double):
    aux=int(parametro)
    array1= np.array([], dtype=np.double) 
    array2= np.array([], dtype=np.double)
    array3= np.array([], dtype=np.double)
    array4= np.array([], dtype=np.double)
    array5= np.array([], dtype=np.double)
    array6= np.array([], dtype=np.double) 
    array7= np.array([], dtype=np.double)
    array8= np.array([], dtype=np.double)
    array9= np.array([], dtype=np.double)
    array10= np.array([], dtype=np.double)
    
    
    for i in range(0, aux):
        array1 =np.append(array1,[array[(10*i)]])
     
    for i in range(0, aux):
        array2 =np.append(array2,[array[(1+i*10)]])
        
    for i in range(0, aux):
        array3 =np.append(array3,[array[(2+i*10)]])
        
    for i in range(0, aux):
        array4 =np.append(array4,[array[(3+i*10)]])
           
    for i in range(0, aux):
        array5 =np.append(array5,[array[(4+i*10)]])

    for i in range(0, aux):
        array6 =np.append(array6,[array[(5+i*10)]])
        
    for i in range(0, aux):
        array7 =np.append(array7,[array[(6+i*10)]])
        
    for i in range(0, aux):
        array8 =np.append(array8,[array[(7+i*10)]])
           
    for i in range(0, aux):
        array9 =np.append(array9,[array[(8+i*10)]])

    for i in range(0, aux):
        array10 =np.append(array10,[array[(9+i*10)]])
    
 
    printag(t, array1, "posição(m)", "Posição da Massa 1 em função do tempo", cor)
    printag(t, array2, "posição(m)", "Posição da Massa 2 em função do tempo", cor)
    printag(t, array3, "posição(m)", "Posição da Massa 3 em função do tempo", cor)
    printag(t, array4, "posição(m)", "Posição da Massa 4 em função do tempo", cor)
    printag(t, array5, "posição(m)", "Posição da Massa 5 em função do tempo", cor)
    printag(t, array6, "posição(m)", "Posição da Massa 6 em função do tempo", cor)
    printag(t, array7, "posição(m)", "Posição da Massa 7 em função do tempo", cor)
    printag(t, array8, "posição(m)", "Posição da Massa 8 em função do tempo", cor)
    printag(t, array9, "posição(m)", "Posição da Massa 9 em função do tempo", cor)
    printag(t, array10, "posição(m)", "Posição da Massa 10 em função do tempo", cor)   
  
########Funcao para plotar um grafico e mostra na tela
def printag(t,x,nomey,titulo, cor): 
    
 
    plt.plot(t,x,cor)
    plt.grid()
    plt.xlabel("tempo(s)")
    plt.ylabel(nomey)
    plt.title(titulo)
    plt.show()
    plt.close()

print("inicio do programa")
deslocamento = input('Deseja utilizar deslocamento nos itens? (sim ou nao)' ) 
teste = input('Qual teste deseja realizar? (a, b ou c)' ) 
if deslocamento == 'sim': 
    deslocamento = 7.0
if deslocamento == 'nao':
    deslocamento = 8.0

if teste == 'a':                 ##caso o usuario escolha o item a, procedemos com os testes da letra a
    dimensao =  input('Que dimensao de matriz deseja? (4, 8, 16 ou 32)' )
    dimensao = int(dimensao)
    matriz_a = np.identity(dimensao)*2    #matriz criada de acordo com o parametro do usuario
    for i in range(0, (matriz_a.shape[0])-1):                          
                for j in range(0, (matriz_a.shape[0])-1):
                    if i == j:
                        matriz_a[i][j+1] = -1
                        matriz_a[i+1][j]= -1
    resultado = eig(matriz_a, deslocamento)
    desejo = input('Deseja imprimir autovalores, autovetores ou ambos? (autovalores, autovetores ou ambos)' )
    if desejo == 'autovalores':  #varias possibilidades de impressao 
        print(resultado[0])
        print("foram contabilizadas", resultado[2], "iterações")
    if desejo == 'autovetores':
        print(resultado[1])
        print("foram contabilizadas", resultado[2], "iterações")
    if desejo == 'ambos':
        print(resultado[0])
        print(resultado[1])
        print("foram contabilizadas", resultado[2], "iterações")
        
if teste == 'b': #caso o usuario proceda com o item b
    tipoB = input('Qual teste dentro de B deseja realizar? (primeiro, segundo, terceiro, customizavel)' )
    matriz_b = np.identity(5)*2                                     #criacao da matriz molas
    matriz_b[0][0]=86.0                                         
    matriz_b[1][1]=90.0
    matriz_b[2][2]=94.0
    matriz_b[3][3]=98.0
    matriz_b[4][4]=102.0
    matriz_b[0][1]=-44.0
    matriz_b[1][2]=-46.0
    matriz_b[2][3]=-48.0
    matriz_b[3][4]=-50.0
    for i in range(0, 5):                   
        for j in range(0, 5):
            matriz_b[j][i] = matriz_b[i][j]                        ##fim da matriz molas######
    matriz_b=matriz_b*0.5
    resultadob = eig(matriz_b, deslocamento)
    QT=np.transpose(resultadob[1])                                ##transposta de autovetores##
    t=np.arange(0,10.025,0.025)                                   #tempo de simulacao
    parametro = (10.025/0.025)                                    #numero de pontos por massa
    
    if tipoB == 'primeiro':   #primeiro teste da letra B
        X0=np.array([[-2.0,-3.0,-1.0,-3.0,-1.0]])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(5)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 5):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadob[0][i])*j)
            x = resultadob[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        plotar(t, armazenar, parametro, 'g')
        
    if tipoB == 'segundo': #segundo teste da letra B
        X0=np.array([[1.0,10.0,-4.0,3.0,-2.0]])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(5)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 5):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadob[0][i])*j)
            x = resultadob[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        
        plotar(t, armazenar, parametro, 'm')
        
    if tipoB == 'terceiro': #terceiro teste da letra B
        X0=copy.deepcopy(QT[:][0])
        X0=np.array([X0])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(5)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 5):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadob[0][i])*j)
            x = resultadob[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        
        plotar(t, armazenar, parametro, 'c')
        
    if tipoB == 'customizavel': # teste customizado da letra B
        string1 = input('digite um valor para posição da massa 1' )
        string2 = input('digite um valor para posição da massa 2' )
        string3 = input('digite um valor para posição da massa 3' )
        string4 = input('digite um valor para posição da massa 4' )
        string5 = input('digite um valor para posição da massa 5' )
        string1=float(string1)
        string2=float(string2)
        string3=float(string3)
        string4=float(string4)
        string5=float(string5)
        X0 = np.array([[]], dtype=np.double)
        X0 = np.append(X0,[string1]) 
        X0 = np.append(X0,[string2]) 
        X0 = np.append(X0,[string3]) 
        X0 = np.append(X0,[string4]) 
        X0 = np.append(X0,[string5])  
        X0=np.array([X0])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(5)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 5):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadob[0][i])*j)
            x = resultadob[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        
        plotar(t, armazenar, parametro, 'r')

if teste == 'c': #caso o usuario proceda com o item c
    tipoC = input('Qual teste dentro de C deseja realizar? (primeiro, segundo, terceiro, customizavel)' )
    matriz_c = np.identity(10)
    matriz_c = matriz_c*80.0
    for i in range(0, 9):
        if i%2 == 0:
            matriz_c[i][i+1]= -42.0
        else:
            matriz_c[i][i+1] = -38.0
            
    for i in range(0, 10):                   
        for j in range(0, 10):
            matriz_c[j][i] = matriz_c[i][j]                        ##fim da matriz molas######
    
    matriz_c=matriz_c*0.5
    resultadoc = eig(matriz_c, deslocamento)
    QT=np.transpose(resultadoc[1])                                ##transposta de autovet##
    t=np.arange(0,10.025,0.025)
    parametro = (10.025/0.025)
    
    if tipoC == 'primeiro':  #primeiro teste da letra C
        X0=np.array([[-2.0,-3.0,-1.0,-3.0,-1.0,-2.0,-3.0,-1.0,-3.0,-1.0]])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(10)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 10):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadoc[0][i])*j)
            x = resultadoc[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        plotarc(t, armazenar, parametro, 'gold')
        
    if tipoC == 'segundo':  #segundo teste da letra C
        X0=np.array([[1.0,10.0,-4.0,3.0,-2.0,1.0,10.0,-4.0,3.0,-2.0]])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(10)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 10):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadoc[0][i])*j)
            x = resultadoc[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        plotarc(t, armazenar, parametro, 'salmon')
        
    if tipoC == 'terceiro':  #terceiro teste da letra C
        df = pd.DataFrame(resultadoc[1])
        df.to_excel('letraC3.xlsx')
        X0=copy.deepcopy(QT[:][0])
        X0=np.array([X0])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(10)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 10):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadoc[0][i])*j)
            x = resultadoc[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        
        plotarc(t, armazenar, parametro, 'aquamarine') 
        
    if tipoC == 'customizavel': #teste customizado da letra C
        string1 = input('digite um valor para posição da massa 1' )
        string2 = input('digite um valor para posição da massa 2' )
        string3 = input('digite um valor para posição da massa 3' )
        string4 = input('digite um valor para posição da massa 4' )
        string5 = input('digite um valor para posição da massa 5' )
        string6 = input('digite um valor para posição da massa 6' )
        string7 = input('digite um valor para posição da massa 7' )
        string8 = input('digite um valor para posição da massa 8' )
        string9 = input('digite um valor para posição da massa 9' )
        string10 = input('digite um valor para posição da massa 10' )
        string1=float(string1)
        string2=float(string2)
        string3=float(string3)
        string4=float(string4)
        string5=float(string5)
        string6=float(string6)
        string7=float(string7)
        string8=float(string8)
        string9=float(string9)
        string10=float(string10)
        X0 = np.array([[]], dtype=np.double)
        X0 = np.append(X0,[string1]) 
        X0 = np.append(X0,[string2]) 
        X0 = np.append(X0,[string3]) 
        X0 = np.append(X0,[string4]) 
        X0 = np.append(X0,[string5])
        X0 = np.append(X0,[string6]) 
        X0 = np.append(X0,[string7]) 
        X0 = np.append(X0,[string8]) 
        X0 = np.append(X0,[string9]) 
        X0 = np.append(X0,[string10])
        X0=np.array([X0])
        X0=np.transpose(X0)
        Y0=QT.dot(X0)
        yT=np.zeros(10)
        armazenar=np.array([[]], dtype = np.double)
        for j in t:
            for i in range (0, 10):
                yT[i] = Y0[i][0]*np.cos(np.sqrt(resultadoc[0][i])*j)
            x = resultadoc[1].dot(yT)
            armazenar = np.append(armazenar,[x])
        
        plotarc(t, armazenar, parametro, 'r')
        
print("fim do programa")
    
  
            


























