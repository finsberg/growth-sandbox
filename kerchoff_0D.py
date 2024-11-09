import matplotlib.pyplot as plt
import numpy as np

# global constants
f_ff_max = 0.3
f_f = 150
s_l50 = 0.06
F_ff50 = 1.35
f_l_slope = 40
f_cc_max = 0.1
c_f = 75
s_t50 = 0.07
F_cc50 = 1.28
c_th_slope = 60


def k_growth(F_g_cum, slope, F_50):
    return 1 / (1 + np.exp(slope * (F_g_cum - F_50)))


def incr_fiber_growth(s_l, dt, F_l_cum):
    if s_l >= 0:
        k_ff = k_growth(F_l_cum, f_l_slope, F_ff50)
        frac = f_ff_max * dt / (1 + np.exp(-f_f * (s_l - s_l50)))
        return k_ff * frac + 1

    else:
        frac = -f_ff_max * dt / (1 + np.exp(f_f * (s_l + s_l50)))
        return frac + 1


def incr_trans_growth(s_t, dt, F_c_cum):
    if s_t >= 0:
        k_cc = k_growth(F_c_cum, c_th_slope, F_cc50)
        frac = f_cc_max * dt / (1 + np.exp(-c_f * (s_t - s_t50)))
        return np.sqrt(k_cc * frac + 1)
    else:
        frac = -f_cc_max * dt / (1 + np.exp(c_f * (s_t + s_t50)))
        return np.sqrt(frac + 1)


def incr_growth_tensor(s_l, s_t, dt, F_g_cum):
    F_g_i_ff = incr_fiber_growth(s_l, dt, F_g_cum[0, 0])
    F_g_i_cc = incr_trans_growth(s_t, dt, F_g_cum[1, 1])
    F_g_i = np.eye(3)

    F_g_i[0, 0] = F_g_i_ff
    F_g_i[1, 1] = F_g_i_cc
    F_g_i[2, 2] = F_g_i_cc
    return F_g_i


def grow_unit_cube(lmbda, T, N, E_f_set=0, E_c_set=0):
    """ "
    Plot the growth of a unit cube over time, resulting from a
    constant stretch lmbda
    """
    # time measured in days, N steps
    time = np.linspace(0, T, N + 1)
    dt_growth = T / N

    # cumulative growth tensor components:
    Fg_tot = np.eye(3)
    F = np.eye(3)
    F[0, 0] = lmbda
    F[1, 1] = 1 / np.sqrt(lmbda)
    F[2, 2] = 1 / np.sqrt(lmbda)

    E = 0.5 * (F.T @ F - np.eye(3))

    F_g_f_tot = np.ones_like(time)
    F_g_c_tot = np.ones_like(time)

    F_0 = F.copy()

    for i in range(N):
        print("Step ", i)
        # growth stimuli:
        sl = E[0, 0] - E_f_set
        st = E[1, 1] - E_c_set

        Fg_i = incr_growth_tensor(sl, st, dt_growth, Fg_tot)
        Fg_tot = Fg_tot @ Fg_i
        F_g_f_tot[i + 1] = Fg_tot[0, 0]
        F_g_c_tot[i + 1] = Fg_tot[1, 1]

        F = F_0 @ np.linalg.inv(Fg_tot)

        E = 0.5 * (F.T @ F - np.eye(3))

    fig, ax = plt.subplots()
    ax.plot(time, F_g_f_tot, label="Fiber")
    ax.plot(time, F_g_c_tot, label="Cross-fiber")
    ax.plot(time, np.ones_like(time) * lmbda, ":")
    ax.plot(time, np.ones_like(time) * 1 / np.sqrt(lmbda), ":")

    ax.set_title(rf"Uniaxial stretch, $\lambda$ = {lmbda}")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Cumulative growth tensor components")
    ax.legend()
    fig.savefig("cumulative_growth.png")
    # plt.show()


if __name__ == "__main__":
    grow_unit_cube(lmbda=1.1, T=300, N=5000, E_f_set=0, E_c_set=0)
