import torch
import numpy as np


def Compute_BandWidth(X, Y, Iteration = 23, Stop_Threshold = 0.00023):
    import time
    import gc
    with torch.no_grad():
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        Spent_Time = time.time()
        Unique_Labels = torch.sort(torch.unique(Y)).values
        Unique_Labels = Unique_Labels[1:]
        #Initial_BandWidth = np.random.randint(np.min(X), np.max(X), size = (Unique_Labels.shape[0], X.shape[-1])).astype(np.float32)
        #Initial_BandWidth = torch.randint(low = int(torch.max(X)), high = int(torch.add(torch.max(X), 1.23)), size = (Unique_Labels.shape[0], X.shape[-1])).type(torch.float32)
        Initial_BandWidth = torch.ones(size=(Unique_Labels.shape[0], X.shape[-1])).type(torch.float64) * 23
        X = torch.Tensor(X)
        #print(X.shape)
        #input("SS")
        X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
        #print("X:", X.shape)
        #print("Y:", Y.shape)
        #print("Unique_Labels:", Unique_Labels.shape)
        #print("Initial_BandWidth:", Initial_BandWidth.shape, "\n", Initial_BandWidth)
        Previous_BandWidth = torch.zeros(size = Initial_BandWidth.shape, dtype = torch.float32)
        for k in range(Iteration):
            if torch.mean(torch.absolute(torch.subtract(Initial_BandWidth, Previous_BandWidth))) < Stop_Threshold:
                break
            print("\nIteration", k + 1, end = "... ")
            #Previous_BandWidth = np.copy(Initial_BandWidth)
            Previous_BandWidth = torch.clone(Initial_BandWidth)
            for i in range(Unique_Labels.shape[0]):
                #print("Unique Label:", i + 1, end = "\t")
                Selected_Indices = torch.where(Y == Unique_Labels[i])
                #print(Selected_Indices)
                X_Temp = X[Selected_Indices[0], :]
                #print("X_Temp:", X_Temp.shape)
                #print(X_Temp, "\n\n")
                Temp_3 = torch.zeros(size = X_Temp.shape, dtype = torch.float32)
                #print(Temp_3, "\n\n")
                for j in range(X_Temp.shape[0]):
                    #if (j + 1) % 100 == 0: print("Data", j + 1, " Of", X_Temp.shape[0])
                    #Temp = np.delete(X_Temp, j, axis=0)
                    Temp = list(range(X_Temp.shape[0]))
                    Temp.remove(j)
                    Temp = X_Temp[Temp]
                    #print(X_Temp[: ,:5], "\n", Temp[: ,:5])
                    if Temp.shape[0] == 0:
                        break
                    #print(Temp)
                    Temp = torch.Tensor(Temp)
                    Temp = torch.subtract(Temp, X_Temp[j])
                    Temp = torch.pow(Temp, 2)
                    #print("Sub:", Temp)
                    Temp_2 = torch.divide(Temp, torch.multiply(2, torch.add(torch.pow(Initial_BandWidth[i], 2), 1e-23)))
                    #print("\nTemp_2:", Temp_2.shape, "\n", Temp_2)
                    Temp_2 = torch.sum(Temp_2, dim = 1)
                    Temp_2 = Temp_2.reshape((Temp_2.shape[0], 1))
                    Temp_2 = torch.exp(torch.multiply(-1, Temp_2))
                    #print("\nTemp_2:", Temp_2.shape, "\n", Temp_2)
                    #print("\nTemp_3:", np.multiply(Temp_2, Temp))
                    Temp_2 = torch.divide(torch.sum(torch.multiply(Temp_2, Temp), axis = 0), torch.add(torch.sum(Temp_2), 1e-23)) # Temp_3
                    #print("\nDivide:", Temp_2)
                    Temp_3[j] = Temp_2
                #print("\nTemp_3:", Temp_3)
                Temp_3 = torch.sum(Temp_3, axis = 0)
                #print("\nSum Temp_3:", Temp_3)
                Temp_3 = torch.divide(Temp_3, X_Temp.shape[0])
                #print("\nDivide Sum Temp_3:", Temp_3)
                Temp_3 = torch.sqrt(Temp_3)
                #print("\nSqrt Divide Sum Temp_3:", Temp_3)
                Initial_BandWidth[i] = Temp_3.type(torch.float32)
                #print("-----------------------")
                torch.cuda.empty_cache()
                gc.collect()
            print("Is Done")
            #print(Initial_BandWidth)
            #print("-----------------------")
            #input("C?")
    print(Initial_BandWidth)
    print("It Is Took", int(time.time() - Spent_Time), "s For ", k + 1, " Of", Iteration, " Iteration")
    Temp = torch.clone(Initial_BandWidth)
    Initial_BandWidth = {}
    for i in range(Unique_Labels.shape[0]):
        Initial_BandWidth[Unique_Labels.numpy()[i]] = Temp[i].to(torch.float64).numpy().tolist()
    return Initial_BandWidth


