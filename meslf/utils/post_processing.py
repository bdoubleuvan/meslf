""" Some function that can be used for processing of results """
from decimal import Decimal
import numpy as np

def error(x_res,x_sol):
    """Relative error between solution and result.

    Parameters
    ----------
    x_res : np array or float
        Variables result.
    x_sol : np array or float
        Variables solution.
    """
    if np.abs(x_sol).all():
        err = np.max(np.abs(x_sol-x_res)/np.abs(x_sol))
    else:
        err = np.max(np.abs(x_sol-x_res))
    return err

def max_error_ind(x_res,x_sol):
    """Index of maximum relative error.

    Parameters
    ----------
    x_res : np array or float
        Variables result.
    x_sol : np array or float
        Variables solution.
    """
    return np.argmax(np.abs(x_sol-x_res)/np.abs(x_sol))

def fexp(number):
    """Returns the exponent of number, when number is written in scientific notation. Returns a string."""
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    """Returns the significand or mantissa of number, when number is written in scientific notation."""
    return Decimal(number).scaleb(-fexp(number)).normalize()

def exp_tex(number):
    """Combine the exponent and the significand to write number in a scientific notation that can be used in Latex. Returns a string."""
    if fexp(number)>-3 and fexp(number)<0:
        str_tex = r'${:.5f}$'.format(number)
    elif fexp(number)<3 and fexp(number)>0:
        str_tex = r'${:.3f}$'.format(number)
    elif fexp(number) == 0:
        if number == 0:
            str_tex = str(int(number))
        else:
            str_tex = r'${:.4f}$'.format(number)
    else:
        str_tex = r'${:.3f}'.format(fman(number))
        str_tex += r'\cdot 10^{'+'{:d}'.format(fexp(number))+r'}$'
    return str_tex

def conv_plot_layout_thesis(ax,scale_var=None,tol=None):
    if tol:
        left,right = ax.get_xlim()
        ax.semilogy([left,right],[tol,tol],':k',alpha=.5,label=r'$\tau$')
        ax.set_xlim(left=left,right=right)
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlabel(r'Iteration $k$')
    if scale_var == None:
        ax.set_ylabel(r'$\|F^k\|_2$')
    else:
        ax.set_ylabel(r'$\|\hat{F}^k\|_2$')

def conv_plot_DD_layout_thesis(ax,scale_var=None,tol_lf=None,tol_dd=None,art_zero=None):
    left,right = ax.get_xlim()
    if tol_lf:
        ax.semilogy([left,right],[tol_lf,tol_lf],'k:',alpha=.5,label=r'$\tau_F$')
    if tol_dd:
        ax.semilogy([left,right],[tol_dd,tol_dd],'k-.',alpha=.5,label=r'$\tau$')
    if art_zero:
        ax.semilogy([left,right],[art_zero,art_zero],'k--',alpha=.5,label=r'artifical 0')
    ax.set_xlim(left=left,right=right)
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
    ax.set_xlabel(r'Iteration $k$')
    if scale_var == None:
        ax.set_ylabel(r'$e^k$')
    else:
        ax.set_ylabel(r'$\hat{e}^k$')

def conv_order_plot_DD_layout_thesis(ax,scale_var=None):
    if scale_var == None:
        ax.set_xlabel(r'$e^k / e^0$')
        ax.set_ylabel(r'$e^{k+1} / e^0$')
    else:
        ax.set_xlabel(r'$\hat{e}^k / \hat{e}^0$')
        ax.set_ylabel(r'$\hat{e}^{k+1} / \hat{e}^0$')
    x_min,x_max = ax.get_xlim()
    x_slope = np.linspace(x_min,x_max)
    y_slope2 = x_slope**2
    ax.loglog(x_slope,x_slope,linestyle='--',color='k',label='order 1')
    ax.loglog(x_slope,y_slope2,linestyle='-.',color='k',label='order 2')
    ax.grid(which='major',color='k', linestyle='--', alpha=.2)
    ax.grid(which='minor',color='k', linestyle=':', alpha=.05)
