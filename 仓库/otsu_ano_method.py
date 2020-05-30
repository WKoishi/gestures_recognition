#利用OTSU算法计算直方图中设定区间[sta:fin]的最佳阈值
def OTSU_Threshold_o(img,sta,fin):

    hist = cv2.calcHist([img],[0],None,[fin-sta],[sta,fin])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(fin-sta)
    fn_min = np.inf
    thresh = -1

    for i in range(1,fin-sta):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[fin-sta-1]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
        if q2==0:
            q2=0.000001
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    thresh+=(sta-1)

    return thresh