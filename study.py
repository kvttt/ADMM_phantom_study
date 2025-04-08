import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse.linalg import lsqr, LinearOperator
from scipy.ndimage import zoom
from itertools import product


def rmse(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def A_forward_vert(x, factor=2):
    return x[::factor, :]

def A_forward_horiz(x, factor=2):
    return x[:, ::factor]

def A_adj_vert(y, factor=2, output_shape=None):
    if output_shape is None:
        raise ValueError("output_shape must be provided.")
    x = np.zeros(output_shape)
    x[::factor, :] = y
    return x

def A_adj_horiz(y, factor=2, output_shape=None):
    if output_shape is None:
        raise ValueError("output_shape must be provided.")
    x = np.zeros(output_shape)
    x[:, ::factor] = y
    return x

def get_rho_schedule(k, rho0=1.0, gamma=1.0):
    return rho0 * (gamma ** k)

def get_alpha_schedule(k, max_iter_admm, alpha_start=1.0, alpha_end=1.0):
    return alpha_start + (alpha_end - alpha_start) * (k / max_iter_admm)

def get_alpha_schedule_cosine(k, max_iter_admm, alpha_start=1.0, alpha_end=1.0):
    alpha = 0.5 * (alpha_start + alpha_end) + 0.5 * (alpha_start - alpha_end) * np.cos(np.pi * k / max_iter_admm)
    return alpha

def get_shepp_logan_phantom(shape=(256, 256)):
    return np.clip(phantom(shape, sl_amps, sl_scales, sl_offsets, sl_angles, float)[::-1, :], 0, 1)

sl_amps = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

sl_scales = [
    [0.6900, 0.920, 0.810],  # white big
    [0.6624, 0.874, 0.780],  # gray big
    [0.1100, 0.310, 0.220],  # right black
    [0.1600, 0.410, 0.280],  # left black
    [0.2100, 0.250, 0.410],  # gray center blob
    [0.0460, 0.046, 0.050],
    [0.0460, 0.046, 0.050],
    [0.0460, 0.046, 0.050],  # left small dot
    [0.0230, 0.023, 0.020],  # mid small dot
    [0.0230, 0.023, 0.020],
]

sl_offsets = [
    [0.0, 0.0, 0],
    [0.0, -0.0184, 0],
    [0.22, 0.0, 0],
    [-0.22, 0.0, 0],
    [0.0, 0.35, -0.15],
    [0.0, 0.1, 0.25],
    [0.0, -0.1, 0.25],
    [-0.08, -0.605, 0],
    [0.0, -0.606, 0],
    [0.06, -0.605, 0],
]

sl_angles = [
    [0, 0, 0],
    [0, 0, 0],
    [-18, 0, 10],
    [18, 0, 10],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

def phantom(shape, amps, scales, offsets, angles, dtype):
    if len(shape) == 2:
        ndim = 2
        shape = (1, shape[-2], shape[-1])

    elif len(shape) == 3:
        ndim = 3

    else:
        raise ValueError("Incorrect dimension")

    out = np.zeros(shape, dtype=dtype)

    z, y, x = np.mgrid[
        -(shape[-3] // 2): ((shape[-3] + 1) // 2),
        -(shape[-2] // 2): ((shape[-2] + 1) // 2),
        -(shape[-1] // 2): ((shape[-1] + 1) // 2),
    ]

    coords = np.stack(
        (
            x.ravel() / shape[-1] * 2,
            y.ravel() / shape[-2] * 2,
            z.ravel() / shape[-3] * 2,
        )
    )

    for amp, scale, offset, angle in zip(amps, scales, offsets, angles):
        ellipsoid(amp, scale, offset, angle, coords, out)

    if ndim == 2:
        return out[0, :, :]

    else:
        return out

def ellipsoid(amp, scale, offset, angle, coords, out):
    R = rotation_matrix(angle)
    coords = (np.matmul(R, coords) - np.reshape(offset, (3, 1))) / np.reshape(
        scale, (3, 1)
    )
    r2 = np.sum(coords**2, axis=0).reshape(out.shape)
    out[r2 <= 1] += amp

def rotation_matrix(angle):
    cphi = np.cos(np.radians(angle[0]))
    sphi = np.sin(np.radians(angle[0]))
    ctheta = np.cos(np.radians(angle[1]))
    stheta = np.sin(np.radians(angle[1]))
    cpsi = np.cos(np.radians(angle[2]))
    spsi = np.sin(np.radians(angle[2]))
    alpha = [
        [
            cpsi * cphi - ctheta * sphi * spsi,
            cpsi * sphi + ctheta * cphi * spsi,
            spsi * stheta,
        ],
        [
            -spsi * cphi - ctheta * sphi * cpsi,
            -spsi * sphi + ctheta * cphi * cpsi,
            cpsi * stheta,
        ],
        [
            stheta * sphi, -stheta * cphi, ctheta
        ],
    ]
    return np.array(alpha)


if __name__ == "__main__":
    def run(tv_weight, rho0, gamma, alpha_start, alpha_end, show_recon_plot=False, show_rmse_plot=False, show_rho_alpha_plot=True, cosine=False):
        # configs
        high_res_shape = (512, 512)  # simulated phantom fov
        R = 4  # undersampling factor 
        max_iter_admm = 20  # max number of ADMM iterations
        max_iter_lsqr = 20  # max number of LSQR iterations
        tol_lsqr = 1e-5  # tolerance for LSQR
        show_recon_plot = show_recon_plot
        show_rmse_plot = show_rmse_plot
        show_rho_alpha_plot = show_rho_alpha_plot

        # rho and alpha schedule
        rho0 = rho0  # initial rho
        gamma = gamma  # rho scaling factor
        alpha_start = alpha_start  # initial alpha
        alpha_end = alpha_end  # final alpha
        rho_schedule = lambda k: get_rho_schedule(k, rho0=rho0, gamma=gamma)
        if cosine:
            alpha_schedule = lambda k: get_alpha_schedule_cosine(k, max_iter_admm, alpha_start=alpha_start, alpha_end=alpha_end)
        else:
            alpha_schedule = lambda k: get_alpha_schedule(k, max_iter_admm, alpha_start=alpha_start, alpha_end=alpha_end)
        
        # phantom (hr)
        phantom_hr = get_shepp_logan_phantom(high_res_shape)

        # phantom (lr)
        y1 = A_forward_vert(phantom_hr, factor=R)
        y2 = A_forward_horiz(phantom_hr, factor=R)

        # initial guesses
        x = np.zeros(high_res_shape)
        z = np.zeros(high_res_shape)
        u = np.zeros(high_res_shape)

        # TV denoising weight
        tv_weight = tv_weight

        # print
        suffix = f'tv_{tv_weight}_r_{rho0}_g_{gamma}_a_{alpha_start}_{alpha_end}'
        # print(f'Running with tv_weight={tv_weight}, rho0={rho0}, gamma={gamma}, alpha_start={alpha_start}, alpha_end={alpha_end}')

        # forward and adjoint operators 
        def build_A_operator(rho, factor, im_shape):
            N = im_shape[0] * im_shape[1]
            N1 = (im_shape[0] // factor) * im_shape[1]
            N2 = im_shape[0] * (im_shape[1] // factor)
            n_rows = N1 + N2 + N
            n_cols = N
            def matvec(x_vec):
                x_img = x_vec.reshape(im_shape)
                r1 = A_forward_vert(x_img, factor=factor)
                r2 = A_forward_horiz(x_img, factor=factor)
                r3 = np.sqrt(rho) * x_img
                return np.concatenate([r1.ravel(), r2.ravel(), r3.ravel()])
            def rmatvec(y_vec):
                y1 = y_vec[0:N1].reshape((im_shape[0] // factor, im_shape[1]))
                y2 = y_vec[N1:N1+N2].reshape((im_shape[0], im_shape[1] // factor))
                y3 = y_vec[N1+N2:].reshape(im_shape)
                a1 = A_adj_vert(y1, factor=factor, output_shape=im_shape)
                a2 = A_adj_horiz(y2, factor=factor, output_shape=im_shape)
                a3 = np.sqrt(rho) * y3
                return (a1 + a2 + a3).ravel()
            return LinearOperator((n_rows, n_cols), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

        # ADMM
        history = {'rmse_x': [], 'rmse_z': [], 'rmse_all': []}

        for k in range(max_iter_admm):
            # rho and alpha
            current_rho = rho_schedule(k)
            current_alpha = alpha_schedule(k)
            
            # lhs
            A_op = build_A_operator(current_rho, factor=R, im_shape=high_res_shape)
            
            # rhs
            b_rhs = np.concatenate([
                y1.ravel(),
                y2.ravel(),
                (np.sqrt(current_rho) * (z - u)).ravel()
            ])
            
            # step 1
            sol = lsqr(A_op, b_rhs, iter_lim=max_iter_lsqr, atol=tol_lsqr, btol=tol_lsqr)
            x_vec = sol[0]
            x = x_vec.reshape(high_res_shape)
            # callback
            current_rmse_x = rmse(x, phantom_hr)
            history['rmse_x'].append(current_rmse_x)
            history['rmse_all'].append(current_rmse_x)
            
            # step 2
            x_plus_u = x + u
            denoised = denoise_tv_chambolle(x_plus_u, weight=tv_weight)
            z = current_alpha * denoised + (1 - current_alpha) * x_plus_u
            # callback
            current_rmse_z = rmse(z, phantom_hr)
            history['rmse_z'].append(current_rmse_z)
            history['rmse_all'].append(current_rmse_z)
            
            # step 3
            u = u + (x - z)

        print(f'Final nRMSE (x): {current_rmse_x * 100:.2f}%')
        # print(f'Final nRMSE (z): {current_rmse_z * 100:.2f}%')

        # plot recon
        fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout='constrained')
        for ax in axes.flat:
            ax.axis('off')
        axes[0, 0].imshow(phantom_hr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, 0].set_title('High resolution')
        axes[0, 1].imshow(zoom(y1, (R, 1), order=0), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, 1].set_title('Low resolution (vertical)')
        axes[0, 2].imshow(zoom(y2, (1, R), order=0), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, 2].set_title('Low resolution (horizontal)')
        axes[0, 3].imshow(x, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, 3].set_title('Recon (x)')
        axes[0, 4].imshow(z, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, 4].set_title('Recon (z)')
        axes[1, 0].imshow(np.zeros_like(phantom_hr), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 1].imshow(np.abs(phantom_hr - zoom(y1, (R, 1), order=0)), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 1].set_title(f'nRMSE: {rmse(zoom(y1, (R, 1), order=0), phantom_hr) * 100:.2f}%')
        axes[1, 2].imshow(np.abs(phantom_hr - zoom(y2, (1, R), order=0)), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 2].set_title(f'nRMSE: {rmse(zoom(y2, (1, R), order=0), phantom_hr) * 100:.2f}%')
        axes[1, 3].imshow(np.abs(phantom_hr - x), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 3].set_title(f'nRMSE: {rmse(x, phantom_hr) * 100:.2f}%')
        axes[1, 4].imshow(np.abs(phantom_hr - z), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1, 4].set_title(f'nRMSE: {rmse(z, phantom_hr) * 100:.2f}%')
        fig.savefig(f'phantom_recon_{suffix}.png', dpi=300)
        if show_recon_plot:
            plt.show()
        plt.close('all')

        # plot nrmse
        fig, axes = plt.subplots(1, 1, figsize=(9, 6), layout='constrained')
        axes.plot(history['rmse_x'], label='nRMSE after ADMM (x)', marker='o', color='C0')
        axes.plot(history['rmse_z'], label='nRMSE after TV denoising (z)', marker='s', color='C1')
        axes.set_xlabel('ADMM Iteration')
        axes.set_ylabel('nRMSE')
        axes.set_xticks(range(0, len(history['rmse_x'])))
        axes.set_title('nRMSE vs ADMM Iteration')
        axes.legend()
        axes.grid(True)
        fig.savefig(f'phantom_rmse_{suffix}.png', dpi=300)
        if show_rmse_plot:
            plt.show()
        plt.close('all')

        # plot rho and alpha
        fig, axes = plt.subplots(1, 1, figsize=(9, 6), layout='constrained')
        axes.plot([rho_schedule(k) for k in range(max_iter_admm)], label='rho', marker='o', color='C0')
        axes.set_ylabel('rho', color='C0')
        axes.set_ylim(-0.05, 1.05)
        axes.tick_params(axis='y', labelcolor='C0')
        axes_other = axes.twinx()
        axes_other.plot([alpha_schedule(k) for k in range(max_iter_admm)], label='alpha', marker='s', color='C1')
        axes_other.set_ylabel('alpha', color='C1')
        axes_other.set_ylim(-0.05, 1.05)
        axes_other.tick_params(axis='y', labelcolor='C1')
        axes.set_xlabel('ADMM Iteration')
        axes.set_xticks(range(0, max_iter_admm))
        axes.set_title('rho and alpha vs ADMM Iteration')
        axes.grid(True)
        fig.savefig(f'phantom_rho_alpha_{suffix}.png', dpi=300)
        if show_rho_alpha_plot:
            plt.show()
        plt.close('all')

    # for 4/9/2025, present the following
    # 1. show what this demo is about
    tv_weight = 1.0
    rho0 = 1.0
    gamma = 1.0
    alpha_start = 1.0
    alpha_end = 1.0
    run(tv_weight, rho0, gamma, alpha_start, alpha_end, show_recon_plot=True, show_rmse_plot=True, show_rho_alpha_plot=True)

    # 2. to demonstrate the effect of alpha schedule, 
    #    we fix tv_weight to be a relatively large value to reflect the fact 
    #    we have a relatively strong, biased, and smoothing denoiser.
    #    we also fix a fixed rho schedule.
    #    metion that here we use a cosine schedule but a linear schedule also works fine.
    tv_weight_lst = [1.0]
    rho0_lst = [1.0]
    gamma_lst = [1.0]
    alpha_start_lst = [1.0]
    alpha_end_lst = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    cosine = True
    lst = list(product(tv_weight_lst, rho0_lst, gamma_lst, alpha_start_lst, alpha_end_lst))
    for tv_weight, rho0, gamma, alpha_start, alpha_end in lst:
        print(f'alpha_end = {alpha_end}')
        run(tv_weight, rho0, gamma, alpha_start, alpha_end, show_rho_alpha_plot=True, cosine=cosine)

    # 3. to demonstrate the effect of rho schedule,
    #    we fix tv_weight to be a relatively large value to reflect the fact
    #    we have a relatively strong, biased, and smoothing denoiser.
    #    we also fix a cosine alpha schedule.
    #    here we use an exponential schedule but I don't know if a linear schedule works.
    tv_weight_lst = [1.0]
    rho0_lst = [1.0]
    gamma_lst = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    alpha_start_lst = [1.0]
    alpha_end_lst = [0.0]
    cosine = True
    lst = list(product(tv_weight_lst, rho0_lst, gamma_lst, alpha_start_lst, alpha_end_lst))
    for tv_weight, rho0, gamma, alpha_start, alpha_end in lst:
        print(f'gamma = {gamma}')
        run(tv_weight, rho0, gamma, alpha_start, alpha_end, show_rho_alpha_plot=True, cosine=cosine)
