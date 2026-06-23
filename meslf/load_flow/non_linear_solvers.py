"""Class to solve non-linear system of equations arising from steady-state load flow problems"""
import abc
import warnings
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.optimize as spo
import scipy.sparse.linalg
from meslf.load_flow.system_of_equations import NonLinearSystem
import time

# ===========================================================================
class Solver(metaclass=abc.ABCMeta):
    """Abstract base class for non-linear solver.
    """
    @abc.abstractmethod
    def solve(self,nlsys,x_init,*args,**kwargs):
        """ Abstract (instance) method to solve nlsys.F(x)=0, starting at x_init
        
        Parameters
        ----------
        nlsys : NonLinearSystem
            Non-linear system to be solved.
        x_init : np array
            Initial guess. 
        """
        
# ===========================================================================
class NR(Solver):
    """Class for using basic Newton-Raphson, with analytical Jacobian.
    
    Attributes
    ----------
    iters : int
        iteration number
    err_vec : list
        list with error per iteration
    tol : float
        Tolerance :math:`\varepsilon` of NR. Default is :math:`\varepsilon = 10^{-6}` 
        
    Returns
    -------
    x_new : np array
        the latest vector x
    """
    def __init__(self):
        self.iters = 0
        self.err_vec = None
        self.tol = 1.0e-6
        self.nl_time = 0
        self.l_time = 0
        
    def solve(self,nlsys,x_init,max_iter,D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),det_tol=1e-8,return_all_x=False,lin_solver='solve',max_iter_lin=None):
        """Basic Newton-Raphson method. Iterations are stopped if the error is smaller than the specified tolerance, if the iteration number is larger than the specified maximum number of iterations, if the determinant of the Jacobian becomes too small, that is if :math:`|J(x)|<\\varepsilon_J`, or if the update :math:`\delta x` becomes 0. 
        
        If both a scaling matrix :math:`D` and a permutation matrix :math:`P` are given, then the transformation matrix becomes :math:`T = PD`. Otherwise, :math:`T = D` or :math:`T = P`. If a scaling matrix or a permutation matrix are provided, the transformed system of equations is used for the stopping criterion. That is, :math:`||T_F F(x)||_2` is used as the error, such that NR has converged if :math:`||T_F F(x)|| \\leq \\varepsilon`. Otherwise, the original system of equations is used in the stopping criterion, i.e. :math:`||F(x)|| \\leq \\varepsilon`. If a scaling matrix :math:`D_x` or transformation matrix :math:`T_x` for :math:`x` is also specified, the scaled function :math:`D_F F(x)` is used during solving (as wel as the scaled stopping criterion)
        
        If at some iteration the determinant of the Jacobian becomes too small, that is if :math:`|J(x)|<\\varepsilon_J`, the solver is stopped. The current :math:`x` is returned.
        Note: If :math:`D_F` and :math:`D_x` of :math:`P_F` and :math:`P_x` are np arrays, then the scaled Jacobian matrix will become a np array instead of a scipy sparse matrix. 
        
        Parameters
        ----------
        nlsys : NonLinearSystem
            Non-linear system to be solved.
        x_init : np array
            Initial guess. 
        max_iter : int
            Maximum number of iterations.
        D_F : array, optional
            Diagonal scaling matrix :math:`D_F` with which to scale the system of equations :math:`F`.
        D_x : array, optional
            Diagonal matrix :math:`D_x` with which to scale the variable vector :math:`x`.
        P_F : array, optional
            Permutation matrix :math:`P_F` for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix. 
        P_x : array, optional
            Permutation matrix :math:`P_x` for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix. 
        det_tol : float, optional
            Value of the determinant below which the Jacobian matrix is considered numerically singular. The solver is then stopped. Default is :math:`\\varepsilon_J = 10^{-8}`.
        return_all_x : bool, optional
            When true, the vector x is returned for every iteration. That is, a matrix with x as rows at every iteration is returned.
            
        Warns
        ------
        UserWarning
            If at some NR iteration the determinant of the Jacobian becomes too small, that is, if :math: `|J(x)|<\\varepsilon_J`. The solver is stopped.
        UserWarning
            If the linear solver did not reach convergence at some NR iteration.
        UserWarning
            If the NR update :math:`\delta x` becomes 0 at some NR iteration.
            
        Raises
        ------
        TypeError
            If nlsys is not an instance of NonLinearSystem.
        ValueError 
            If an incompatible or invalid linear solver is provided.
        """
        if not isinstance(nlsys,NonLinearSystem):
            raise TypeError("nlsys has to be an instance of NonLinearSystem")
        # create the transformation matrices T, and the inverse matrix of T_x
        T_F,T_x,T_x_inv,T_F_len,T_x_len = nlsys.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        x_new = x_init
        if return_all_x:
            x_mat = x_new.copy()
        F_new = nlsys.F(x_new)
        J_new = nlsys.J(x_new)
        if T_F_len and T_x_len: # scale x, F and J
            x_new = T_x.dot(x_new)
            F_new = T_F.dot(F_new)
            J_new = T_F.dot(J_new.dot(T_x_inv))
        iter_nr = 0
        err_vec = []
        # check if initial guess happens to be correct
        if T_F_len and (not T_x_len): # Scale stopping criterion only when only T_F is specified. If T_x is specified, F_new is already scaled. 
            error = np.linalg.norm(T_F.dot(F_new))
        else:
            error = np.linalg.norm(F_new)
        err_vec.append(error)
        nl_start_time = time.time()
        while error > self.tol and iter_nr < max_iter:
            x_old = x_new
            F_old = F_new
            J_old = J_new
            try: 
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("error","Matrix is",sps.linalg.MatrixRankWarning) # A warning of (sub)class sps.linalg.MatrixRankWarning, of which the message starts with 'Matrix is' is raised as an exception
                    warnings.filterwarnings("error","{} did not reach convergence".format(lin_solver),UserWarning)
                    warnings.filterwarnings("error","dx is 0",UserWarning)
                    if sps.issparse(J_old):
                        if lin_solver == 'solve':
                            l_start_time = time.time()
                            dx =  sps.linalg.spsolve(J_old,F_old)
                            self.l_time += time.time()-l_start_time
                        elif lin_solver == 'gmres':
                            l_start_time = time.time()
                            dx,info = sps.linalg.gmres(J_old,F_old,tol=self.tol/error,maxiter=max_iter_lin,restart=max(int(max_iter_lin/10),20))
                            self.l_time += time.time()-l_start_time
                            if info:
                                warnings.warn('{} did not reach convergence after {} iterations, in NR iteration {}.'.format(lin_solver,max_iter_lin,iter_nr))
                        elif lin_solver == 'bicgstab':
                            l_start_time = time.time()
                            dx,info = sps.linalg.bicgstab(J_old,F_old,tol=self.tol/error,maxiter=max_iter_lin)
                            self.l_time += time.time()-l_start_time
                            if info:
                                warnings.warn('{} did not reach convergence after {} iterations, in NR iteration {}.'.format(lin_solver,max_iter_lin,iter_nr))
                        else:
                            raise ValueError('Enter a valid value for lin_solver',UserWarning)
                        if np.all(dx==0):
                            warnings.warn('dx is 0 for {}, in NR iteration {}.'.format(lin_solver,iter_nr))
                        x_new = x_old - dx
                    else:
                        if lin_solver == 'solve':
                            l_start_time = time.time()
                            x_new = x_old - np.linalg.solve(J_old,F_old)
                            self.l_time += time.time()-l_start_time
                        elif lin_solver == 'gmres':
                            raise ValueError('gmres can only be used for sparse systems')
                        elif lin_solver == 'bicgstab':
                            raise ValueError('bicgstab can only be used for sparse systems')
                        else:
                            raise ValueError('Enter a valid value for lin_solver')
            except Exception as e:
                if isinstance(e,sps.linalg.MatrixRankWarning):
                    if sps.issparse(J_old):
                        det_J = np.linalg.det(J_old.todense())
                    else:
                        det_J = np.linalg.det(J_old)
                    warnings.warn(str(e) + ' Determinant |J| = {:4e}, NR is stopped after {} iterations.'.format(det_J,iter_nr))
                else:
                    warnings.warn(str(e) + ' NR is stopped after {} iterations.'.format(iter_nr))
                break
            if T_F_len and T_x_len:
                # F and J need unscaled x
                F_new = nlsys.F(T_x_inv.dot(x_new))
                J_new = nlsys.J(T_x_inv.dot(x_new))
                # scale F and J
                F_new = T_F.dot(F_new)
                J_new = T_F.dot(J_new.dot(T_x_inv))
            else:
                F_new = nlsys.F(x_new)
                J_new = nlsys.J(x_new)
            if T_F_len and (not T_x_len): # Scale stopping criterion only when only T_F is specified. If T_x is specified, F_new is already scaled. 
                error = np.linalg.norm(T_F.dot(F_new))
            else:
                error = np.linalg.norm(F_new)
            iter_nr += 1
            err_vec.append(error)
            if return_all_x:
                if T_F_len and T_x_len: # scale x back
                    x_mat = np.vstack((x_mat,T_x_inv.dot(x_new)))
                else:
                    x_mat = np.vstack((x_mat,x_new))
        self.iters = iter_nr
        self.err_vec = err_vec
        if T_F_len and T_x_len: # scale x back
            x_new = T_x_inv.dot(x_new)
        self.nl_time = time.time() - nl_start_time # total time spent in the non-linear solver
        if self.iters:
            self.l_time = self.l_time / self.iters # average time spent in linear solver over all non-linear iterations
        if return_all_x:
            return x_new,x_mat
        else:
            return x_new
    
class NR_FD(Solver):
    """Class for using basic Newton-Raphson, with FD Jacobian.
    
    Attributes
    ----------
    iters : int
        iteration number
    err_vec : list
        list with error per iteration
    tol : float
        tolerance of NR. Default = 1.0e-6
    h : float
        step size used by FD
        
    Returns
    -------
    x_new : np array
        the latest vector x
    """
    def __init__(self):
        self.iters = 0
        self.err_vec = None
        self.tol = 1.0e-6
        self.h = 1.0e-6
        self.nl_time = 0
        self.l_time = 0
        
    def solve(self,nlsys,x_init,max_iter,D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([]),lin_solver='solve',max_iter_lin=None):
        """Basic Newton Raphson solver. Iterations are stopped if the error is smaller than the specified tolerance, if the iteration number is larger than the specified maximum number of iterations, or if the update :math:`\delta x` becomes too small. 
        
        If both a scaling matrix :math:`D` and a permutation matrix :math:`P` are given, then the transformation matrix becomes :math:`T = PD`. Otherwise, :math:`T = D` or :math:`T = P`. If a scaling matrix or a permutation matrix are provided, the transformed system of equations is used for the stopping criterion. That is, :math:`||T_F F(x)||_2` is used as the error, such that NR has converged if :math:`||T_F F(x)|| \\leq \\varepsilon`. Otherwise, the original system of equations is used in the stopping criterion, i.e. :math:`||F(x)|| \\leq \\varepsilon`. If a scaling matrix :math:`D_x` or transformation matrix :math:`T_x` for :math:`x` is also specified, the scaled function :math:`D_F F(x)` is used during solving (as wel as the scaled stopping criterion)
        
        At every iteration, this solver tries to solve a linear system of equations. If this is not possible, this solvers switches to a least-squares algorithm.
        
        Note: the FD Jacobian matrix is np array, not a scipy sparse matrix. 
        
        Parameters
        ----------
        nlsys : NonLinearSystem
            Non-linear system to be solved.
        x_init : np array
            Initial guess. 
        max_iter : int
            Maximum number of iterations.
        D_F : array, optional
            Diagonal scaling matrix :math:`D_F` with which to scale the system of equations :math:`F`.
        D_x : array, optional
            Diagonal matrix :math:`D_x` with which to scale the variable vector :math:`x`.
        P_F : array, optional
            Permutation matrix :math:`P_F` for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix. 
        P_x : array, optional
            Permutation matrix :math:`P_x` for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix. 
        det_tol : float, optional
            Value of the determinant below which the Jacobian matrix is considered numerically singular. The solver is then stopped. Default is :math:`\\varepsilon_J = 10^{-8}`.
        return_all_x : bool, optional
            When true, the vector x is returned for every iteration. That is, a matrix with x as rows at every iteration is returned.
        
        Warns
        ------
        UserWarning
            If at some iteration the solver switches to a least-squares algorithm.
        UserWarning
            If the linear solver did not reach convergence at some NR iteration.
        UserWarning
            If the NR update :math:`\delta x` becomes 0 at some NR iteration.
            
        Raises
        ------
        TypeError
            If nlsys is not an instance of NonLinearSystem.
        ValueError 
            If an incompatible or invalid linear solver is provided.    
        """
        if not isinstance(nlsys,NonLinearSystem):
            raise TypeError("nlsys has to be an instance of NonLinearSystem")
        # create the transformation matrices T, and the inverse matrix of T_x
        T_F,T_x,T_x_inv,T_F_len,T_x_len = nlsys.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        # make sure inverse of transformation matrix is not sparse
        if sps.issparse(T_x_inv):
            T_x_inv = np.asarray(T_x_inv.todense()) # todense returns a np matrix, but np ndarray is needed.
        x_new = x_init
        F_new = nlsys.F(x_new)
        J_new = nlsys.J_FD(x_new,self.h)
        if T_F_len and T_x_len: # scale x, F and J
            x_new = T_x.dot(x_new)
            F_new = T_F.dot(F_new)
            J_new = T_F.dot(J_new.dot(T_x_inv))
        iter_nr = 0
        err_vec = []
        # check if initial guess happens to be correct
        if T_F_len and (not T_x_len): # Scale stopping criterion only when only T_F is specified. If T_x is specified, F_new is already scaled. 
            error = np.linalg.norm(T_F.dot(F_new))
        else:
            error = np.linalg.norm(F_new)
        err_vec.append(error)
        while error > self.tol and iter_nr < max_iter:
            x_old = x_new
            F_old = F_new
            J_old = J_new
            try: 
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("error","Matrix is",sps.linalg.MatrixRankWarning) # A warning of (sub)class sps.linalg.MatrixRankWarning, of which the message starts with 'Matrix is' is raised as an exception
                    warnings.filterwarnings("error","{} did not reach convergence".format(lin_solver),UserWarning)
                    warnings.filterwarnings("error","dx is 0",UserWarning)
                    if sps.issparse(J_old):
                        if lin_solver == 'solve':
                            l_start_time = time.time()
                            dx =  sps.linalg.spsolve(J_old,F_old)
                            self.l_time += time.time()-l_start_time
                        elif lin_solver == 'gmres':
                            l_start_time = time.time()
                            dx,info = sps.linalg.gmres(J_old,F_old,tol=self.tol/error,maxiter=max_iter_lin,restart=max(int(max_iter_lin/10),20))
                            self.l_time += time.time()-l_start_time
                            if info:
                                warnings.warn('{} did not reach convergence after {} iterations, in NR iteration {}.'.format(lin_solver,max_iter_lin,iter_nr))
                        elif lin_solver == 'bicgstab':
                            l_start_time = time.time()
                            dx,info = sps.linalg.bicgstab(J_old,F_old,tol=self.tol/error,maxiter=max_iter_lin)
                            self.l_time += time.time()-l_start_time
                            if info:
                                warnings.warn('{} did not reach convergence after {} iterations, in NR iteration {}.'.format(lin_solver,max_iter_lin,iter_nr))
                        else:
                            raise ValueError('Enter a valid value for lin_solver',UserWarning)
                        if np.all(dx==0):
                            warnings.warn('dx is 0 for {}, in NR iteration {}.'.format(lin_solver,iter_nr))
                        x_new = x_old - dx
                    else:
                        if lin_solver == 'solve':
                            l_start_time = time.time()
                            x_new = x_old - np.linalg.solve(J_old,F_old)
                            self.l_time += time.time()-l_start_time
                        elif lin_solver == 'gmres':
                            raise ValueError('gmres can only be used for sparse systems')
                        elif lin_solver == 'bicgstab':
                            raise ValueError('bicgstab can only be used for sparse systems')
                        else:
                            raise ValueError('Enter a valid value for lin_solver')
            except Exception as e:
                if isinstance(e,sps.linalg.MatrixRankWarning):
                    if sps.issparse(J_old):
                        det_J = np.linalg.det(J_old.todense())
                    else:
                        det_J = np.linalg.det(J_old)
                    warnings.warn(str(e) + ' Determinant |J| = {:4e}, NR is stopped after {} iterations.'.format(det_J,iter_nr))
                else:
                    warnings.warn(str(e) + ' NR is stopped after {} iterations.'.format(iter_nr))
                break
            if T_F_len and T_x_len:
                # F and J need unscaled x
                F_new = nlsys.F(T_x_inv.dot(x_new))
                J_new = nlsys.J_FD(T_x_inv.dot(x_new),self.h)
                # scale F and J
                F_new = T_F.dot(F_new)
                J_new = T_F.dot(J_new.dot(T_x_inv))
            else:
                F_new = nlsys.F(x_new)
                J_new = nlsys.J_FD(x_new,self.h)
            if T_F_len and (not T_x_len): # Scale stopping criterion only when only T_F is specified. If T_x is specified, F_new is already scaled. 
                error = np.linalg.norm(T_F.dot(F_new))
            else:
                error = np.linalg.norm(F_new)
            iter_nr += 1
            err_vec.append(error)
        self.iters = iter_nr
        self.err_vec = err_vec
        if T_F_len and T_x_len: # scale x back
            x_new = T_x_inv.dot(x_new)
        return x_new
    
# ===========================================================================
class Fsolver(Solver):
    """Class for using scipy.optimize.fsolve(), with analytical Jacobian.
    Note that fsolve() needs a dense Jacobian matrix. 
    
    Attributes
    ----------
    inf_dict : dict
        A dictionary of optional outputs. See scipy documentation.
    flag : int
        Equal to 1 if a solution was found. Otherwise, mssg contains more information. See scipy documentation.
    mssg : str
        Has details when a solution is not found. See scipy documentation.
    iters : int
        Number of Jacobian evaluations needed.
    err_vec : list
        List with error for initial guess, and with final error.
    tol : float
        Tolerance :math:`\varepsilon` of solver. Default is :math:`\varepsilon = 10^{-6}` 
        
    Returns
    -------
    x_new : np array
        the latest vector x
    """
    def __init__(self):
        self.inf_dict = None
        self.flag = 0
        self.mssg = None
        self.iters = 0
        self.err_vec = None
        self.tol = 1.0e-6
        
    def solve(self,nlsys,x_init,max_iter,D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([])):
        """fsolve from scipy.optimize.
        
        Note: If D_F and D_x are np arrays, then the scaled Jacobian matrix will become a np array instead of a scipy sparse matrix. 
        
        Parameters
        ----------
        nlsys : NonLinearSystem
            Non-linear system to be solved.
        x_init : np array
            Initial guess. 
        max_iter : int
            Maximum number of function evaluations.
        D_F : array, optional
            Diagonal scaling matrix :math:`D_F` with which to scale the system of equations :math:`F`.
        D_x : array, optional
            Diagonal matrix :math:`D_x` with which to scale the variable vector :math:`x`.
        P_F : array, optional
            Permutation matrix :math:`P_F` for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix. 
        P_x : array, optional
            Permutation matrix :math:`P_x` for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix. 
        det_tol : float, optional
            Value of the determinant below which the Jacobian matrix is considered numerically singular. The solver is then stopped. Default is :math:`\\varepsilon_J = 10^{-8}`.
        return_all_x : bool, optional
            When true, the vector x is returned for every iteration. That is, a matrix with x as rows at every iteration is returned.
            
        Warns
        ------
        UserWarning
            If a solution has not been found (i.e. if flag != 1)
            
        Raises
        ------
        TypeError
            If nlsys is not an instance of NonLinearSystem     
        """
        if not isinstance(nlsys,NonLinearSystem):
            raise TypeError("nlsys has to be an instance of NonLinearSystem")
        # create the transformation matrices T, and the inverse matrix of T_x
        T_F,T_x,T_x_inv,T_F_len,T_x_len = nlsys.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        err_vec = []
        # check if initial guess happens to be correct
        F_init = nlsys.F(x_init)
        if T_F_len:
            error = np.linalg.norm(T_F.dot(F_init))
        else:
            error = np.linalg.norm(F_init)
        err_vec.append(error)
        if error < self.tol:
            self.err_vec = err_vec
            self.iters = 0
            return x_init
        else:
            # solve the system, using scaling
            if T_F_len and T_x_len: # scale x, F and J
                x_init = T_x.dot(x_init)
                def Jac(x):
                    x_unscaled = T_x_inv.dot(x)
                    J_unscaled = nlsys.J(x_unscaled)
                    J_scaled = T_F.dot(J_unscaled.dot(T_x_inv))
                    if sps.issparse(J_scaled):
                        J_scaled = J_scaled.todense()
                    return J_scaled
                def Func(x):
                    x_unscaled = T_x_inv.dot(x)
                    F_unscaled = nlsys.F(x_unscaled)
                    F_scaled = T_F.dot(F_unscaled)
                    return F_scaled
                x_zero,inf_dict,flag,mssg = spo.fsolve(Func,x_init,fprime=Jac,full_output=True,xtol=self.tol,maxfev=max_iter)
                x_zero = T_x_inv.dot(x_zero)
            # solve the system, usscaled
            else:
                x_zero,inf_dict,flag,mssg = spo.fsolve(nlsys.F,x_init,fprime=nlsys.J_dense,full_output=True,xtol=self.tol,maxfev=max_iter)
            self.inf_dict = inf_dict
            self.flag = flag
            self.mssg = mssg
            self.iters = self.inf_dict["njev"]
            err_vec.append(np.linalg.norm(self.inf_dict["fvec"]))
            self.err_vec = err_vec
            if not self.flag == 1:
                warnings.warn('Solution has not been found. Flag: {}. Message: {}'.format(self.flag,self.mssg))
            return x_zero
    
class Root(Solver):
    """Class for using scipy.optimize.root(), with analytical Jacobian.
    Note that fsolve() needs a dense Jacobian matrix. 
    
    Attributes
    ----------
    inf_dict : dict
        A dictionary of optional outputs. See scipy documentation.
    flag : int
        Equal to 1 if a solution was found. Otherwise, mssg contains more information. See scipy documentation.
    mssg : str
        Has details when a solution is not found. See scipy documentation.
    iters : int
        Number of Jacobian evaluations needed.
    err_vec : list
        List with error for initial guess, and with final error.
    tol : float
        Tolerance :math:`\varepsilon` of solver. Default is :math:`\varepsilon = 10^{-6}` 
        
    Returns
    -------
    x_new : np array
        the latest vector x
    """
    def __init__(self):
        self.inf_dict = None
        self.flag = 0
        self.mssg = None
        self.iters = 0
        self.err_vec = None
        self.tol = 1.0e-6
        
    def solve(self,nlsys,x_init,max_iter,D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([])):
        """fsolve from scipy.optimize.
        
        Note: If D_F and D_x are np arrays, then the scaled Jacobian matrix will become a np array instead of a scipy sparse matrix. 
        
        Parameters
        ----------
        nlsys : NonLinearSystem
            Non-linear system to be solved.
        x_init : np array
            Initial guess. 
        max_iter : int
            Maximum number of function evaluations.
        D_F : array, optional
            Diagonal scaling matrix :math:`D_F` with which to scale the system of equations :math:`F`.
        D_x : array, optional
            Diagonal matrix :math:`D_x` with which to scale the variable vector :math:`x`.
        P_F : array, optional
            Permutation matrix :math:`P_F` for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix. 
        P_x : array, optional
            Permutation matrix :math:`P_x` for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix. 
        det_tol : float, optional
            Value of the determinant below which the Jacobian matrix is considered numerically singular. The solver is then stopped. Default is :math:`\\varepsilon_J = 10^{-8}`.
        return_all_x : bool, optional
            When true, the vector x is returned for every iteration. That is, a matrix with x as rows at every iteration is returned.
            
        Warns
        ------
        UserWarning
            If a solution has not been found (i.e. if flag != 1)
            
        Raises
        ------
        TypeError
            If nlsys is not an instance of NonLinearSystem     
        """
        if not isinstance(nlsys,NonLinearSystem):
            raise TypeError("nlsys has to be an instance of NonLinearSystem")
        # create the transformation matrices T, and the inverse matrix of T_x
        T_F,T_x,T_x_inv,T_F_len,T_x_len = nlsys.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        err_vec = []
        # check if initial guess happens to be correct
        F_init = nlsys.F(x_init)
        if T_F_len:
            error = np.linalg.norm(T_F.dot(F_init))
        else:
            error = np.linalg.norm(F_init)
        err_vec.append(error)
        if error < self.tol:
            self.err_vec = err_vec
            self.iters = 0
            return x_init
        else:
            solver_options = {'ftol':self.tol,'maxiter':max_iter}
            method = 'lm' # Default is 'hybr', which is what fsolve uses?
            # solve the system, using scaling
            if T_F_len and T_x_len: # scale x, F and J
                x_init = T_x.dot(x_init)
                def Jac(x):
                    x_unscaled = T_x_inv.dot(x)
                    J_unscaled = nlsys.J(x_unscaled)
                    J_scaled = T_F.dot(J_unscaled.dot(T_x_inv))
                    if sps.issparse(J_scaled):
                        J_scaled = J_scaled.todense()
                    return J_scaled
                def Func(x):
                    x_unscaled = T_x_inv.dot(x)
                    F_unscaled = nlsys.F(x_unscaled)
                    F_scaled = T_F.dot(F_unscaled)
                    return F_scaled
                sol = spo.root(Func,x_init,method=method,jac=Jac,options=solver_options)
                x_sol = T_x_inv.dot(sol.x)
            # solve the system, usscaled
            else:
                sol = spo.root(nlsys.F,x_init,method=method,jac=nlsys.J_dense,options=solver_options)
                x_sol = sol.x
            self.flag = sol.status
            self.mssg = sol.message
            self.iters = sol.njev
            err_vec.append(np.linalg.norm(sol.fun))
            self.err_vec = err_vec
            if not self.flag == 1:
                warnings.warn('Solution has not been found. Flag: {}. Message: {}'.format(self.flag,self.mssg))
            return x_sol
