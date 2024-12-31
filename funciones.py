import numpy as np
# FUNCIONES AUXILIARES PARA CALCULO CON P
# Función para determinar el indice referido al valor maximo absoluto de un arreglo
def mayorIndice(v):
    mayor_indice = 0
    for i in range(len(v)):
        if abs(v[i]) > abs(v[mayor_indice]):
            mayor_indice = i

    return mayor_indice

# Función que intercambia los valores de las filas n y m de una matriz
def intercambiarFilas(A, n, m):
    A[[n, m], :] = A[[m, n], :]
    return A

# Similar a la función anterior, solo con la salvedad de que solo intercambia los valores ubicados debajo de la diagonal
def intercambiarFilasTriangInf(A, n, m):
    A[[n, m], :n] = A[[m, n], :n]
    return A


# CALCULO LU CON P
import numpy as np
def calcularLU(A):
    filas, columnas = A.shape
    A = A.astype(float)
    U = A.copy()
    L = np.eye(filas)
    P = np.eye(filas)

    if filas != columnas:
        print("No se puede descomponer la siguiente matriz, ya que la matriz no es cuadrada")
        return None

    for i in range(filas-1):

        mayor_indice = i + mayorIndice(U[i:, i])
        #Antes de aplicar el algoritmo de eliminación Gaussiana se busca permutar la matriz para colocar su máximo valor absoluto de la columna corrrespondiente como pivote para minimizar el error numerico
        if U[i, mayor_indice] == 0:
            return("No se puede descomponer la siguiente matriz, ya que no se encuentra ningun pivote no nulo")
            return None

        elif mayor_indice != i:
            U = intercambiarFilas(U, i, mayor_indice)
            P = intercambiarFilas(P, i, mayor_indice)
            if i > 0:
                L = intercambiarFilasTriangInf(L, i, mayor_indice)

        #Algoritmo de eliminación Gaussiana
        for j in range(i+1, filas):
            factor = U[j, i] / U[i, i]
            U[j, :] = U[j, :] - factor * U[i, :]
            L[j, i] = factor

    return L, U, P

# CALCULO DE INVERSA CON LU
from scipy.linalg import solve_triangular
def inversaLU(L, U, P):
    filas, columnas = L.shape
    L = L.astype(float)
    U = U.astype(float)
    Inv = np.zeros((filas, columnas))
    identidad = np.eye(filas)

    for i in range(filas):
        N = solve_triangular(L, identidad[:, i], lower=True)
        Y = solve_triangular(U, N, lower=False)
        Inv[:, i] = Y

    Inv = np.dot(Inv, P.T)
    return Inv
