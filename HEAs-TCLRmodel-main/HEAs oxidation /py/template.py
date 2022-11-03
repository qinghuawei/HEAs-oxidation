if __name__ == "__main__":

    import sys
    # Configure a temporary address
    # locdir is the address of your computer where the folder 'TreeClassifierforLinearRegression' is located
    locdir = '/Users/jackwey/Desktop/TCLRmodel-main-old-version/Source Code'
    sys.path.append(locdir)

    #import TCLRmodel.TCLRalgorithm as TCLR
    import TCLRmodel.TCLRalgorithmOldVersion as TCLR



    dataSet = "Z-Data_高熵合金-氧化.csv"



    #dataSet = "High entropy alloy Corrosion_Deng-LuCheng.csv"

    # :param dataSet：the input dataset
    correlation = 'PearsonR(+)'



    minsize = 3
    threshold = 1
    mininc = 0.01

    TCLR.start(dataSet,correlation, minsize, threshold, mininc)




