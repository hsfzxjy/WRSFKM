def anderson_iteration(X, U, V, labels, p, logger):

    def multi_update_V(V, U, X):
        delta_V = 100

        while delta_V > 1e-1:
            new_V = update_V(V, U, X, epsilon)
            delta_V = l21_norm(new_V - V)
            V = new_V

        return V

    V_len = V.flatten().shape[0]

    mAA = 0
    V_old = V
    U_old = U

    iterations = t = 0
    mmax = p.mmax or 4
    AAstart = p.AAstart or 0
    droptol = p.droptol or 1e10

    gamma = p.gamma
    epsilon = p.epsilon
    max_iteration = p.max_iterations or 300

    fold = gold = None
    g = np.ndarray(shape=(V_len, 0))
    Q = np.ndarray(shape=(V_len, 1))
    R = np.ndarray(shape=(1, 1))

    old_E = np.Infinity
    VAUt = None

    while True:
        U_new = solve_U(X, V_old, gamma, epsilon)

        delta_U, is_converged = U_converged(U_new, U_old)

        new_E = E(U_new, V_old, X, gamma, epsilon)

        if is_converged:
            return U_new, V_old, t, metric(U_new, labels)

        if t > max_iteration:
            return U_new, V_old, t, metric(U_new, labels)

        if new_E >= old_E:
            mAA = 0
            iterations = 0
            V_old = VAUt
            g = np.ndarray(shape=(V_len, 0))
            Q = np.ndarray(shape=(V_len, 1))
            R = np.ndarray(shape=(1, 1))
            U_new = solve_U(X, V_old, gamma, epsilon)
            old_E = E(U_new, V_old, X, gamma, epsilon)

        # AA Start
        # VAUt = gcur = update_V(V_old, U_new, X, epsilon)
        VAUt = gcur = multi_update_V(V_old, U_new, X)
        fcur = gcur - V_old

        if iterations > AAstart:
            delta_f = (fcur - fold).reshape((V_len, 1))
            delta_g = (gcur - gold).reshape((V_len, 1))

            if mAA < mmax:
                g = np.hstack((g, delta_g))
            else:
                g = np.hstack((g[:, 1:mAA], delta_g))

            mAA += 1

        fold, gold = fcur, gcur

        if mAA == 0:
            V_new = gcur
        else:
            if mAA == 1:
                delta_f_norm = l21_norm(delta_f)
                Q[:, 0] = delta_f.flatten() / delta_f_norm
                R[0, 0] = delta_f_norm
            else:
                if mAA > mmax:
                    Q, R = qr_delete(Q, R, 1)
                    mAA -= 1

                R = np.resize(R, (mAA, mAA))
                Q = np.resize(Q, (V_len, mAA))
                for i in range(0, mAA - 1):
                    R[i, mAA - 1] = Q[:, i].T @ delta_f
                    delta_f = delta_f - (R[i, mAA - 1] * Q[:, i]).reshape((V_len, 1))

                delta_f_norm = l21_norm(delta_f)
                Q[:, mAA - 1] = delta_f.flatten() / delta_f_norm
                R[mAA - 1, mAA - 1] = delta_f_norm

            while np.linalg.cond(R) > droptol and mAA > 1:
                Q, R = qr_delete(Q, R, 1)
                mAA -= 1

            Gamma = scipy.linalg.solve(R, Q.T @ fcur.reshape(V_len, 1))
            V_new = gcur - (g @ Gamma).reshape(V.shape)

        delta_V, _ = U_converged(V_new, V_old)
        V_old = V_new
        U_old = U_new
        old_E = new_E
        logger.log_middle(E(U_new, V_new, X, gamma, epsilon), metric(U_new, labels))
        t += 1
        iterations += 1
