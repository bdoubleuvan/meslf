"""Base class to create the (non-linear) system of equations, and the Jacobian matrix, for steady-state load flow analysis."""
import abc
import numpy as np
import scipy.sparse as sps
import meslf.utils.basic_math as bm
import matplotlib.pyplot as plt
import warnings

# ===========================================================================
class NonLinearSystem(metaclass=abc.ABCMeta):
    """Abstract class that creates the (non-linear) system of equations F(x)=0
    and the corresponding Jacobian matrix J(x).
    """
    @abc.abstractmethod
    def F(self,x,**kwargs):
        """Abstract (instance) method to determine F(x).

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        F : np array
            (non-linear) system of equations evaluated at x
        """

    @abc.abstractmethod
    def J(self,x,**kwargs):
        """Abstract (instance) method to determine J(x), based on analytical expressions.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        J : sps matrix
            Jacobian matrix evaluated at x
        """

    def Dx(self):
        """Determines the diagonal scaling matrix for the variable vector x.

        Returns
        -------
        Dx : sps matrix
            Diagonal matrix to scale x
        """
        pass

    def DF(self):
        """Determines the diagonal scaling matrix for the vector of equations F.

        Returns
        -------
        DF : sps matrix
            Diagonal matrix to scale F
        """
        pass

    def scal_perm_matr(self,D_F=np.array([]),D_x=np.array([]),P_F=np.array([]),P_x=np.array([])):
        """Create the transformation matrices :math:`T`, and the inverse of the matrix :math:`T_x`. These matrices are used to scale and permute the system of equations and the vector of variables. If both a scaling matrix :math:`D` and a permutation matrix :math:`P` are given, then :math:`T = PD`. Otherwise, :math:`T = D` or :math:`T = P`. The scaling matrix :math:`D` is assumed to be diagonal, and the permutation matrix :math:`P` is assumed to be an orthogonal binary matrix. NB: This is not checked in this function.

        Parameters:
        -----------
        D_F : array, optional
            Diagonal scaling matrix :math:`D_F` with which to scale the system of equations :math:`F`.
        D_x : array, optional
            Diagonal matrix :math:`D_x` with which to scale the variable vector :math:`x`.
        P_F : array, optional
            Permutation matrix :math:`P_F` with which to permute the system of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix.
        P_x : array, optional
            Permutation matrix :math:`P_x` with which to permute the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix.

        Returns:
        --------
        T_F : array
            Transformation matrix :math:`T_F` for the system of equations :math:`F`.
        T_x : array
            Transformation matrix :math:`T_x` for the variable vector :math:`x`.
        T_x_inv : array
            Inverse of the transformation matrix :math:`T_x`.
        T_F_len : float
            Length of the transformation matrix :math:`T_F`.
        T_x_len : float
            Length of the transformation matrix :math:`T_x`.
        """
        # create the transformation matrices T, and the inverse matrix of T_x
        if sps.issparse(D_x):
            D_x_inv = sps.diags(1/D_x.data[0])
            D_x_len = D_x.shape[0]
        else:
            D_x_inv = np.diag(1/np.diag(D_x))
            D_x_len = len(D_x)
        if sps.issparse(P_x):
            P_x_inv = P_x.transpose()
            P_x_len = P_x.shape[0]
        else:
            P_x_inv = P_x.transpose()
            P_x_len = len(P_x)
        if D_x_len and P_x_len:
            T_x = P_x.dot(D_x)
            T_x_inv = D_x_inv.dot(P_x_inv)
            T_x_len = P_x_len
        elif D_x_len:
            T_x = D_x
            T_x_inv = D_x_inv
            T_x_len = D_x_len
        elif P_x_len:
            T_x = P_x
            T_x_inv = P_x_inv
            T_x_len = P_x_len
        else:
            T_x = np.array([])
            T_x_inv = np.array([])
            T_x_len = 0
        if sps.issparse(D_F):
            D_F_len = D_F.shape[0]
        else:
            D_F_len = len(D_F)
        if sps.issparse(P_F):
            P_F_len = P_F.shape[0]
        else:
            P_F_len = len(P_F)
        if D_F_len and P_F_len:
            T_F = P_F.dot(D_F)
            T_F_len = P_F_len
        elif D_F_len:
            T_F = D_F
            T_F_len = D_F_len
        elif P_F_len:
            T_F = P_F
            T_F_len = P_F_len
        else:
            T_F = np.array([])
            T_F_len = 0
        return T_F,T_x,T_x_inv,T_F_len,T_x_len

    def J_FD(self,x,h):
        """Determines the Jacobian matrix, based on a simple finite differene scheme.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.
        h : float
            Step size for FD scheme.

        Returns
        -------
        J_FD : np array
            Jacobian matrix determined by finite difference scheme, evaluated at x.
        """
        J_FD = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            e = np.zeros(len(x))
            e[i] = 1.
            J_FD[:,i] = (self.F(x+e*h)-self.F(x))/h
        return J_FD

    def J_dense(self,x,return_full=False):
        """Return the analytical Jacobian J(x) in dense form instead of as a sparse matrix

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        J : np matrix
            Jacobian matrix evaluated at x
        """
        return self.J(x,return_full=return_full).todense()

    def compare_J(self,x,h):
        """Compares the analytical Jacobian J(x) with the finite difference Jacobian J_FD(x)

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.
        h : float
            step size for FD scheme

        Returns
        -------
        diff_max : maximum difference, in absolute value, between the analytical and finite difference Jacobian matrices.
        """
        return np.amax(abs(self.J(x) - self.J_FD(x,h)))

    def spy_plot_J(self,x,ax=None,P_F=np.array([]),P_x=np.array([]),title='Jacobian spy plot',markerfacecolor='tab:blue',markeredgecolor='tab:blue',marker='s',alpha=1,overlay=True):
        """Spy plot of Jacobian matrix J(x), with lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.
        title : str, optional
            Figure (window) title. Default is 'Jacobian spy plot'.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The spy plot of J(x)
        """
        # determine Jacobian
        D_x = self.Dx()
        D_F = self.DF()
        T_F,T_x,T_x_inv,T_F_len,T_x_len = self.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        J = self.J(x)
        if T_F_len and T_x_len:
            J = T_F.dot(J.dot(T_x_inv))
        elif T_F_len:
            J = T_F.dot(J)
        elif T_x_len:
            J = J.dot(T_x_inv)
        if not ax:
            fig = plt.figure(title)
            ax = fig.gca()
            return_fig = True
        else:
            return_fig = False
        ax.spy(J,marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,alpha=alpha)
        if overlay:
            self.plot_J_overlay(ax,P_F=P_F,P_x=P_x)
        if return_fig:
            return fig

    def imshow_J(self,x,P_F=np.array([]),P_x=np.array([]),title='Jacobian colormap',overlay=True):
        """Imshow / colormap plot of Jacobian matrix J(x), with lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.
        Imshow needs a dense matrix, so matrix is converted to a dense one.
        Zeros (i.e., values not in the sparse matrix) are not plotted. They are shown in white.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.
        title : str, optional
            Figure (window) title. Default is 'Jacobian spy plot'.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The spy plot of J(x)
        """
        # determine Jacobian
        D_x = self.Dx()
        D_F = self.DF()
        T_F,T_x,T_x_inv,T_F_len,T_x_len = self.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        J = self.J(x)
        if T_F_len and T_x_len:
            J = T_F.dot(J.dot(T_x_inv))
        elif T_F_len:
            J = T_F.dot(J)
        elif T_x_len:
            J = J.dot(T_x_inv)
        J_dense = np.matrix(np.nan*np.ones(J.shape))
        indices = J.indices
        indptr = J.indptr
        for row_ind in range(J.shape[0]):
            for col_ind in indices[indptr[row_ind]:indptr[row_ind+1]]:
                J_dense[row_ind,col_ind] = J[row_ind,col_ind]
        fig = plt.figure(title)
        ax = fig.gca()
        plt.imshow(J_dense)
        if overlay:
            self.plot_J_overlay(ax,P_F=P_F,P_x=P_x)
        plt.colorbar()
        return fig

    def plot_J_overlay(self,ax,P_F=np.array([]),P_x=np.array([])):
        """Lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.

        Parameters
        ----------
        ax : matplotlib axes
            Axes the lines are to be plotted on.
        P_F : array, optional
            Permutation matrix :math:`P_F`for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix.
        P_x : array, optional
            Permutation matrix :math:`P_x`for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix.
        """
        pass

    def spectrum_J(self,x,ax=None,P_F=np.array([]),P_x=np.array([]),title='Spectrum of Jacobian',color='tab:blue'):
        """Plot of the eigenvalues of the Jacobian matrix J(x) in the complex plane.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.
        title : str, optional
            Figure (window) title. Default is 'Spectrum of Jacobian'.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The spectrum of J(x)
        """
        # determine Jacobian
        D_x = self.Dx()
        D_F = self.DF()
        T_F,T_x,T_x_inv,T_F_len,T_x_len = self.scal_perm_matr(D_F=D_F,D_x=D_x,P_F=P_F,P_x=P_x)
        J = self.J(x)
        if T_F_len and T_x_len:
            J = T_F.dot(J.dot(T_x_inv))
        elif T_F_len:
            J = T_F.dot(J)
        elif T_x_len:
            J = J.dot(T_x_inv)
        eigs_J,_ = sps.linalg.eigs(J,k=J.shape[0]-2)
        if not ax:
            fig = plt.figure(title)
            ax = fig.gca()
            return_fig = True
        else:
            return_fig = False
        for eig in eigs_J:
            ax.plot(eig.real,eig.imag,'.',color=color)
        radius = np.max(np.abs(eigs_J))
        ax.add_artist(plt.Circle((0, 0), radius, color=color, fill=False))
        ax.plot([-radius,radius],[0,0],'k:',alpha=.5) #horizontal line through origin
        ax.plot([0,0],[-radius,radius],'k:',alpha=.5) #vertical line through origin
        ax.set_xlabel("Im")
        ax.set_ylabel("Re")
        ax.axis('equal')
        left_x,right_x = ax.get_xlim()
        ax.set_xlim(left=min(-radius,left_x),right=max(radius,right_x))
        left_y,right_y = ax.get_xlim()
        ax.set_ylim(bottom=min(-radius,left_y),top=max(radius,right_y))
        if return_fig:
            return fig

# ===========================================================================
class NonLinearSystemGas(NonLinearSystem):
    """Class that creates the (non-linear) system of equations F(x)=0
    and the corresponding Jacobian matrix J(x) for a gas network.
    """
    def __init__(self,gasnetwork,formulation='full',scale_var=None,scale_var_params=None,**kwargs):
        """Creates a NonLinearSystemGas object

        Parameters
        ----------
        gasnetwork : GasNetwork
            Gas network for which the sytem of equations is made.
        formulation : string, optional
            Formulation used to form the system of equations. Default is 'full'. Options are 'full' or 'nodal'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with parameters needed for scaling.
        """
        self.gasnetwork = gasnetwork
        self.gas_formulation = formulation
        if self.gas_formulation == 'nodal':
            for link in self.gasnetwork.get_links():
                if link.link_type in ['pipe_low_pres_hagen_poiseuille','pipe_high_pres_colebrook','pipe_high_pres_blasius','compressor']:
                    raise ValueError("Formulation is 'nodal', but the network contains at least one link of type {}. Use 'full' formulation instead!".format(link.link_type))
        self.scale_var = scale_var
        self.scale_var_params = scale_var_params
        self.F_entries = self.gasnetwork.get_F_entries(formulation=self.gas_formulation)
        self.x_entries = self.gasnetwork.get_x_entries(formulation=self.gas_formulation)
        ind_p = []
        ind_q = []
        ind_Fn = []
        ind_Fl= []
        for el in self.x_entries:
            if 'Node' in type(el).__name__:
                ind_p.append(el.number)
            elif 'Link' in type(el).__name__:
                ind_q.append(el.number)
        for el in self.F_entries:
            if 'Node' in type(el).__name__:
                ind_Fn.append(el.number)
            elif 'Link' in type(el).__name__:
                ind_Fl.append(el.number)
        self.ind_p = ind_p
        self.ind_q = ind_q
        self.ind_Fn = ind_Fn
        self.ind_Fl= ind_Fl

    def F(self,x):
        """Determines F(x) for a gas network.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        F : np array
            (non-linear) system of equations evaluated at x
        """
        self.gasnetwork.update(x,formulation=self.gas_formulation,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        F = np.zeros(len(self.F_entries))
        for ind_el,el in enumerate(self.F_entries):
            if 'Node' in type(el).__name__:
                F[ind_el] = el.node_law(network=self.gasnetwork,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            elif 'Link' in type(el).__name__:
                F[ind_el] = el.link_equation(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        return F

    def J(self,x,return_full=False):
        """Determines J(x) for a gas network, based on analytical expressions.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        J : np array
            Jacobian matrix evaluated at x
        """
        self.gasnetwork.update(x,formulation=self.gas_formulation,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        J = None
        if self.gas_formulation == 'nodal':
            Ddiag_data = []
            dp_der_data = []
            dp_der_row = []
            dp_der_col = []
            x_ind = []
            F_ind = []
            for ind_e,e in enumerate(self.gasnetwork.get_links()):
                Ddiag_data.append(-e.f_der_dp_func(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                der_start,der_end = e.pres_drop_func_der_p(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                if e.start_node in self.gasnetwork.get_nodes(): # check if start_node is in the network, i.e. if it is a GasNode
                    dp_der_data.append(der_start)
                    dp_der_row.append(ind_e)
                    ind_n_start = list(self.gasnetwork.get_nodes()).index(e.start_node)
                    dp_der_col.append(ind_n_start)
                if e.end_node in self.gasnetwork.get_nodes(): # check if end_node is in the network, i.e. if it is a GasNode
                    dp_der_data.append(der_end)
                    dp_der_row.append(ind_e)
                    ind_n_end = list(self.gasnetwork.get_nodes()).index(e.end_node)
                    dp_der_col.append(ind_n_end)
            for n in self.gasnetwork.get_x_entries(formulation=self.gas_formulation):
                x_ind.append(n.number)
            for n in self.gasnetwork.get_F_entries(formulation=self.gas_formulation):
                F_ind.append(n.number)
            D = sps.diags(Ddiag_data)
            dp_der = sps.csr_matrix((dp_der_data,(dp_der_row,dp_der_col)),shape=(len(list(self.gasnetwork.get_links())),len(list(self.gasnetwork.get_nodes()))))
            if return_full:
                J = (self.gasnetwork.A.dot(D)).dot(dp_der)
            else:
                dp_der_tilde = dp_der[:,x_ind]
                A_prime = self.gasnetwork.A[F_ind,:]
                J = (A_prime.dot(D)).dot(dp_der_tilde)
        elif self.gas_formulation == 'full':
            dp_der_data = []
            dp_der_row = []
            dp_der_col = []
            dFq_dq_vec = np.zeros(len(list(self.gasnetwork.get_links())))
            dFq_ddeltap_vec = np.zeros(len(list(self.gasnetwork.get_links())))
            dFl_derp_data = []
            dFl_derp_row = []
            dFl_derp_col = []
            for ind_e,e in enumerate(self.gasnetwork.get_links()):
                der_start,der_end = e.pres_drop_func_der_p(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                dFl_derp_start, dFl_derp_end = e.f_der_p(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                dFq_dq_vec[e.number] = (e.f_der_q(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                dFq_ddeltap_vec[e.number] = (e.f_der_dp_func(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                if e.start_node in self.gasnetwork.get_nodes(): # check if start_node is in the network, i.e. if it is a GasNode
                    dp_der_data.append(der_start)
                    dp_der_row.append(ind_e)
                    ind_n_start = list(self.gasnetwork.get_nodes()).index(e.start_node)
                    dp_der_col.append(ind_n_start)

                    dFl_derp_data.append(dFl_derp_start)
                    dFl_derp_row.append(ind_e)
                    dFl_derp_col.append(ind_n_start)
                if e.end_node in self.gasnetwork.get_nodes(): # check if end_node is in the network, i.e. if it is a GasNode
                    dp_der_data.append(der_end)
                    dp_der_row.append(ind_e)
                    ind_n_end = list(self.gasnetwork.get_nodes()).index(e.end_node)
                    dp_der_col.append(ind_n_end)

                    dFl_derp_data.append(dFl_derp_end)
                    dFl_derp_row.append(ind_e)
                    dFl_derp_col.append(ind_n_end)
            dFq_ddeltap = sps.diags(dFq_ddeltap_vec).tocsr()
            dp_der = sps.csr_matrix((dp_der_data,(dp_der_row,dp_der_col)),shape=(len(list(self.gasnetwork.get_links())),len(list(self.gasnetwork.get_nodes()))))
            dFl_dp = sps.csr_matrix((dFl_derp_data,(dFl_derp_row,dFl_derp_col)),shape=(len(list(self.gasnetwork.get_links())),len(list(self.gasnetwork.get_nodes()))))# dFq_ddeltap.dot(dp_der)
            dFl_dq = sps.diags(dFq_dq_vec).tocsr()
            if return_full:
                J = sps.bmat([
                [self.gasnetwork.A,None],
                 [dFl_dq,dFl_dp]]).tocsr()
            else:
                J = sps.bmat([
                [self.gasnetwork.A[self.ind_Fn,:][:,self.ind_q],None],
                 [dFl_dq[self.ind_Fl,:][:,self.ind_q],
                  dFl_dp[self.ind_Fl,:][:,self.ind_p]]]).tocsr()
        return J

    def Dx(self):
        """Determines the diagonal scaling matrix for the variable vector x.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        Dx : sps matrix
            Diagonal matrix to scale x
        """
        if self.scale_var_params:
            xb_g = np.zeros(len(self.x_entries))
            for ind_el,el in enumerate(self.x_entries):
                if 'Node' in type(el).__name__:
                    xb_g[ind_el] = self.scale_var_params.get('pgbase')
                elif 'Link' in type(el).__name__:
                    xb_g[ind_el] = self.scale_var_params.get('qbase')
            Dx = sps.diags(1/xb_g)
        else:
            Dx = sps.eye(len(self.x_entries))
        return Dx

    def DF(self):
        """Determines the diagonal scaling matrix for the vector of equations F.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        DF : sps matrix
            Diagonal matrix to scale F
        """
        if self.scale_var_params:
            Fb_g = np.zeros(len(self.F_entries))
            for ind_el,el in enumerate(self.F_entries):
                if 'Node' in type(el).__name__:
                    Fb_g[ind_el] = self.scale_var_params.get('qbase')
                elif 'Link' in type(el).__name__:
                    if el.link_type == 'compressor':
                        Fb_g[ind_el] = self.scale_var_params.get('pgbase')
                    else:
                        if el.link_eq_form=='dp_of_q':
                            if 'high_pres' in el.link_type:
                                Fb_g[ind_el] = self.scale_var_params.get('pgbase')**2
                            else:
                                Fb_g[ind_el] = self.scale_var_params.get('pgbase')
                        else:
                            Fb_g[ind_el] = self.scale_var_params.get('qbase')
            DF = sps.diags(1/Fb_g)
        else:
            DF = sps.eye(len(self.F_entries))
        return DF

    def plot_J_overlay(self,ax,**kwargs):
        """Lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.

        Parameters
        ----------
        ax : matplotlib axes
            Axes the lines are to be plotted on.
        x : np array
            Variable vector, possible scaled.
        """
        F_node = len(self.ind_Fn)
        F_link = len(self.ind_Fl)
        x_p = len(self.ind_p)
        x_q = len(self.ind_q)
        ax.plot([0,len(self.x_entries)-0.5],[F_node-0.5,F_node-0.5],'g--')
        ax.text(-2.5,F_node/2-0.5,r'$F^q$')
        ax.text(-2.5,F_node+F_link/2-0.5,r'$F^l$')
        ax.plot([x_q-0.5,x_q-0.5],[0,len(self.F_entries)-0.5],'g--')
        ax.text(x_q/2-0.5,-1.5,'$q$')
        ax.text(x_q+x_p/2-0.5,-1.5,'$p$')


# ===========================================================================
class NonLinearSystemElectrical(NonLinearSystem):
    """Class that creates the (non-linear) system of equations F(x)=0
    and the corresponding Jacobian matrix J(x) for an electrical network.
    """
    def __init__(self,elecnetwork,scale_var=None,scale_var_params=None,**kwargs):
        """Creates a NonLinearSystemElectrical object

        Parameters
        ----------
        elecnetwork : ElectricalNetwork
            Electrial network for which the sytem of equations is made.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with parameters needed for scaling.
        """
        self.elecnetwork = elecnetwork
        self.scale_var = scale_var
        self.scale_var_params = scale_var_params
        self.F_entries, self.known_P_nodes, self.known_Q_nodes = self.elecnetwork.get_F_entries()
        self.x_entries, self.unknown_delta_nodes, self.unknown_V_nodes = self.elecnetwork.get_x_entries()
        FP = list()
        FQ = list()
        xdelta = list()
        xV = list()
        V_vec_mag = np.zeros(len(self.elecnetwork.nodes))
        V_vec_ang = np.zeros(len(self.elecnetwork.nodes))
        for ind_n,n in enumerate(self.elecnetwork.get_nodes()):
            V_vec_mag[ind_n] = n.get_V(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            V_vec_ang[ind_n] = n.get_delta(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            if n in self.known_P_nodes:
                FP.append(ind_n)
            if n in self.known_Q_nodes:
                FQ.append(ind_n)
            if n in self.unknown_delta_nodes:
                xdelta.append(ind_n)
            if n in self.unknown_V_nodes:
                xV.append(ind_n)
        self.FP = FP
        self.FQ = FQ
        self.xdelta = xdelta
        self.xV = xV
        self.V_vec_mag = V_vec_mag
        self.V_vec_ang = V_vec_ang
        if self.scale_var == 'per_unit':
            yb = self.scale_var_params['Sbase']/self.scale_var_params['Vbase']**2
            self.Y = self.elecnetwork.Y.copy()/yb
        else:
            self.Y = self.elecnetwork.Y.copy()

    def F(self,x):
        """Determines F(x) for an electrical network.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        F : np array
            (non-linear) system of equations evaluated at x
        """
        self.elecnetwork.update(x,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        F = np.zeros(len(self.known_P_nodes)+len(self.known_Q_nodes))
        for ind_el,el in enumerate(self.known_P_nodes):
            fP,fQ = el.node_law(network=self.elecnetwork,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            F[ind_el] = fP
        for ind_el,el in enumerate(self.known_Q_nodes):
            fP,fQ = el.node_law(network=self.elecnetwork,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            F[ind_el+len(self.known_P_nodes)] = fQ
        return F

    def J(self,x,return_full=False):
        """Determines J(x) for an electrical network, based on analytical expressions.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        J : np array
            Jacobian matrix evaluated at x
        """
        self.elecnetwork.update(x,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        N = len(list(self.elecnetwork.get_nodes()))
        self.V_vec_ang[self.xdelta] = x[0:len(self.unknown_delta_nodes)]
        self.V_vec_mag[self.xV] = x[len(self.unknown_delta_nodes):]
        V_vec = bm.complex_polar(self.V_vec_mag,self.V_vec_ang)
        I_vec = (self.Y).dot(V_vec)
        V_abs_inv_mat = sps.diags(1./abs(V_vec))
        I_mat = sps.diags(I_vec)
        V_mat = sps.diags(V_vec)
        dS_ddelta = 1j*V_mat.dot(I_mat.conjugate()-(self.Y).conjugate().dot(V_mat.conjugate()))
        dP_ddelta = dS_ddelta.real
        dQ_ddelta = dS_ddelta.imag
        dS_dV = (V_mat.dot(I_mat.conjugate()+(self.Y).conjugate().dot(V_mat.conjugate()))).dot(V_abs_inv_mat)
        dP_dV = dS_dV.real
        dQ_dV = dS_dV.imag
        if return_full:
            J = sps.bmat([[
                dP_ddelta,dP_dV],
                [dQ_ddelta,dQ_dV]])
        else:
            J = sps.bmat([[
                dP_ddelta[self.FP,:][:,self.xdelta],
                dP_dV[self.FP,:][:,self.xV]],
                [dQ_ddelta[self.FQ,:][:,self.xdelta],
                 dQ_dV[self.FQ,:][:,self.xV]]])
        return J.tocsr()

    def Dx(self):
        """Determines the diagonal scaling matrix for the variable vector x.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        Dx : sps matrix
            Diagonal matrix to scale x
        """
        if self.scale_var_params:
            xb_delta = self.scale_var_params.get('deltabase')*np.ones(len(self.unknown_delta_nodes))
            xb_V = self.scale_var_params.get('Vbase')*np.ones(len(self.unknown_V_nodes))
            xb_e = np.concatenate((xb_delta,xb_V))
            Dx = sps.diags(1/xb_e)
        else:
            Dx = sps.eye(len(self.x_entries))
        return Dx

    def DF(self):
        """Determines the diagonal scaling matrix for the vector of equations F.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        DF : sps matrix
            Diagonal matrix to scale F
        """
        if self.scale_var_params:
            Fb_P = self.scale_var_params.get('Sbase')*np.ones(len(self.known_P_nodes))
            Fb_Q = self.scale_var_params.get('Sbase')*np.ones(len(self.known_Q_nodes))
            Fb_e = np.concatenate((Fb_P,Fb_Q))
            DF = sps.diags(1/Fb_e)
        else:
            DF = sps.eye(len(self.F_entries))
        return DF

    def plot_J_overlay(self,ax,**kwargs):
        """Lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.

        Parameters
        ----------
        ax : matplotlib axes
            Axes the lines are to be plotted on.
        x : np array
            Variable vector, possible scaled.
        """
        F_len = len(self.F_entries)
        x_len = len(self.x_entries)
        x_delta_len = len(self.unknown_delta_nodes)
        x_V_len = len(self.unknown_delta_nodes)
        F_P_len = len(self.known_P_nodes)
        F_Q_len = len(self.known_Q_nodes)
        # vertical
        ax.plot((x_delta_len-0.5,x_delta_len-0.5),(-0.5,F_len-0.5),'r--')
        ax.text(x_delta_len/2-0.5,-1.5,r'$\delta$')
        ax.text(x_delta_len+x_V_len/2-0.5,-1.5,r'$|V|$')
        # horizontal
        ax.plot((-0.5,x_len-0.5),(F_P_len-0.5,F_P_len-0.5),'r--')
        ax.text(-2.5,F_P_len/2-0.5,r'$F^P$')
        ax.text(-2.5,F_P_len+F_Q_len/2-0.5,r'$F^Q$')

# ===========================================================================
class NonLinearSystemHeat(NonLinearSystem):
    """Class that creates the (non-linear) system of equations F(x)=0
    and the corresponding Jacobian matrix J(x) for a heat network.
    """
    def __init__(self,heatnetwork,formulation='standard',scale_var=None,scale_var_params=None,**kwargs):
        """Creates a NonLinearSystemHeat object

        Parameters
        ----------
        heatnetwork : HeatNetwork
            Heat network for which the sytem of equations is made.
        formulation : string, optional
            Formulation used to form system of equations. Default is 'standard'. Options are 'standard' and 'half_link_flow'.
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with parameters needed for scaling.
        """
        self.heatnetwork = heatnetwork
        if not (formulation == 'standard' or formulation == 'half_link_flow'):
            raise ValueError("Enter a valid formulation. It must be either 'standard' or 'half_link_flow', not {}".format(formulation))
        self.heat_formulation = formulation
        self.scale_var = scale_var
        self.scale_var_params = scale_var_params
        self.F_entries, self.F_m_nodes, self.F_deltap_links, self.F_Ts_nodes, self.F_Tr_nodes, self.F_phi_halflinks, self.F_dT_halflinks = self.heatnetwork.get_F_entries(formulation=self.heat_formulation)
        self.x_entries, self.unknown_m_links, self.unknown_m_halflinks, self.unknown_p_nodes, self.unknown_Ts_nodes, self.unknown_Tr_nodes, self.unknown_hTs_halflinks, self.unknown_hTr_halflinks = self.heatnetwork.get_x_entries(formulation=self.heat_formulation)
        self.Fm = list()
        self.Fdeltap = list()
        self.FTs = list()
        self.FTr = list()
        self.Fphi = list()
        self.FdT = list()
        self.xm = list()
        self.xmhl = list()
        self.xp = list()
        self.xTs = list()
        self.xTr  = list()
        self.xTshl = list()
        self.xTrhl = list()
        self.all_source_ind = list()
        self.all_sink_ind = list()
        self.all_junction_ind = list()
        self.all_slack_ind = list()
        for ind_n,n in enumerate(self.heatnetwork.get_nodes()):
            if n.node_type in [0,4]:
                self.all_source_ind.append(ind_n)
            if n.node_type in [1,3,12,13,14,15] and (n.half_links[0].bc_type in [2,3,4,5,8,9] and n.half_links[0].source): # assume dphi is known for these nodes
                self.all_source_ind.append(ind_n)
            if n.node_type in [8,16]:
                self.all_sink_ind.append(ind_n)
            if n.node_type in [1,3,12,13,14,15] and (n.half_links[0].bc_type in [2,3,4,5,8,9] and n.half_links[0].sink): # assume dphi is known for these nodes:
                self.all_sink_ind.append(ind_n)
            if n.node_type in [2,5,6,7]:
                self.all_junction_ind.append(ind_n)
            if n.node_type in [0,8,9,10,11]:
                self.all_slack_ind.append(ind_n)
            if n in self.F_m_nodes:
                self.Fm.append(ind_n)
            if n in self.F_Ts_nodes:
                self.FTs.append(ind_n)
            if n in self.F_Tr_nodes:
                self.FTr.append(ind_n)
            if n in self.unknown_p_nodes:
                self.xp.append(ind_n)
            if n in self.unknown_Ts_nodes:
                self.xTs.append(ind_n)
            if n in self.unknown_Tr_nodes:
                self.xTr.append(ind_n)
        for ind_e,e in enumerate(self.heatnetwork.get_links()):
            if e in self.F_deltap_links:
                self.Fdeltap.append(ind_e)
            if e in self.unknown_m_links:
                self.xm.append(ind_e)
        if self.heat_formulation == 'half_link_flow':
            for ind_hl,hl in enumerate(self.heatnetwork.get_half_links()):
                if hl in self.F_phi_halflinks:
                    self.Fphi.append(ind_hl)
                if hl in self.F_dT_halflinks:
                    self.FdT.append(ind_hl)
                if hl in self.unknown_m_halflinks:
                    self.xmhl.append(ind_hl)
                if hl in self.unknown_hTs_halflinks:
                    self.xTshl.append(ind_hl)
                if hl in self.unknown_hTr_halflinks:
                    self.xTrhl.append(ind_hl)

    def F(self,x):
        """Determines F(x) for a heat network.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        F : np array
            (non-linear) system of equations evaluated at x
        """
        self.heatnetwork.update(x,formulation=self.heat_formulation,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        F = np.zeros(len(self.F_m_nodes)+len(self.F_deltap_links)+len(self.F_Ts_nodes)+len(self.F_Tr_nodes)+len(self.F_phi_halflinks)+len(self.F_dT_halflinks))
        for ind_el,el in enumerate(self.F_m_nodes):
            F[ind_el] = el.node_law(network=self.heatnetwork,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        for ind_el,el in enumerate(self.F_deltap_links):
            F[ind_el+len(self.F_m_nodes)] = el.link_equation(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        for ind_el,el in enumerate(self.F_Ts_nodes):
            fTs, fTr = el.mixing_rule(network=self.heatnetwork,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            F[ind_el+len(self.F_m_nodes)+len(self.F_deltap_links)] = fTs
        for ind_el,el in enumerate(self.F_Tr_nodes):
            fTs, fTr = el.mixing_rule(network=self.heatnetwork,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            F[ind_el+len(self.F_m_nodes)+len(self.F_deltap_links)+len(self.F_Ts_nodes)] = fTr
        for ind_el,el in enumerate(self.F_phi_halflinks):
            fphi = el.heat_power_equation(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            F[ind_el+len(self.F_m_nodes)+len(self.F_deltap_links)+len(self.F_Ts_nodes)+len(self.F_Tr_nodes)] = fphi
        for ind_el,el in enumerate(self.F_dT_halflinks):
            fdT = el.temp_diff_equation(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            F[ind_el+len(self.F_m_nodes)+len(self.F_deltap_links)+len(self.F_Ts_nodes)+len(self.F_Tr_nodes)+len(self.F_phi_halflinks)] = fdT
        return F

    def J(self,x,return_full=False):
        """Determines J(x) for a heat network, based on analytical expressions.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        J : np array
            Jacobian matrix evaluated at x
        """
        self.heatnetwork.update(x,formulation=self.heat_formulation,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        # Make all the full derivative matrices.
        N = len(self.heatnetwork.nodes)
        NE = len(self.heatnetwork.links)
        dm_dm = self.heatnetwork.A
        dm_dTs_data = list()
        dm_dTs_row = list()
        dm_dTs_col = list()
        dm_dTr_data = list()
        dm_dTr_row = list()
        dm_dTr_col = list()
        dFL_dm_vec = np.zeros(NE)
        dFL_ddeltap_vec = np.zeros(NE)
        dp_der_data = list()
        dp_der_row = list()
        dp_der_col = list()
        dTs_dm_data = list()
        dTs_dm_row = list()
        dTs_dm_col = list()
        dTs_dTs_data = list()
        dTs_dTs_row = list()
        dTs_dTs_col = list()
        dTs_dTr_data = list()
        dTs_dTr_row = list()
        dTs_dTr_col = list()
        dTr_dm_data = list()
        dTr_dm_row = list()
        dTr_dm_col = list()
        dTr_dTs_data = list()
        dTr_dTs_row = list()
        dTr_dTs_col = list()
        dTr_dTr_data = list()
        dTr_dTr_row = list()
        dTr_dTr_col = list()

        # derivatives of link equations
        for ind_e,e in enumerate(self.heatnetwork.get_links()):
            dFL_dm_vec[ind_e] = e.f_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            dFL_ddeltap_vec[ind_e] = e.f_der_dp_func(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            der_start,der_end = e.pres_drop_func_der_p(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            if e.start_node in self.heatnetwork.get_nodes(): # check if start_node is in the network, i.e. if it is a HeatNode
                dp_der_data.append(der_start)
                dp_der_row.append(ind_e)
                ind_n_start = list(self.heatnetwork.get_nodes()).index(e.start_node)
                dp_der_col.append(ind_n_start)
            if e.end_node in self.heatnetwork.get_nodes(): # check if end_node is in the network, i.e. if it is a HeatNode
                dp_der_data.append(der_end)
                dp_der_row.append(ind_e)
                ind_n_end = list(self.heatnetwork.get_nodes()).index(e.end_node)
                dp_der_col.append(ind_n_end)
        dFL_dm = sps.diags(dFL_dm_vec).tocsr()
        dFL_ddeltap = sps.diags(dFL_ddeltap_vec).tocsr()
        dp_der = sps.csr_matrix((dp_der_data,(dp_der_row,dp_der_col)),shape=(NE,N))
        dFL_dp = dFL_ddeltap.dot(dp_der)

        if self.heat_formulation == 'standard':
            # derivatives of nodal equations
            for ind_n,n in enumerate(self.heatnetwork.get_nodes()):
                dm_dTsi = 0.
                dm_dTri = 0.
                dTs_dTsi = 0.
                dTs_dTri = 0.
                dTr_dTri = 0.
                dTr_dTsi = 0.
                total_inflow = n.get_inflow(scale_var=self.scale_var,scale_var_params=self.scale_var_params) # loops over all links, but not halflinks
                total_outflow = n.get_outflow(scale_var=self.scale_var,scale_var_params=self.scale_var_params) # loops over all links, but not halflinks
                for e in n.get_out_links():
                    dTr_dm = 0.
                    dTs_dm = 0.
                    dTs_dTsj = 0.
                    dTr_dTrj = 0.
                    if e in self.heatnetwork.links: # HeatLinks
                        ind_e = list(self.heatnetwork.get_links()).index(e)
                        # link goes from node i to node j
                        dTsij_dTsi,dTsij_dTsj = e.supply_temp_start_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        dTrij_dTri,dTrij_dTrj = e.return_temp_start_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        if e.end_node in list(self.heatnetwork.get_nodes()):
                            ind_nj = list(self.heatnetwork.get_nodes()).index(e.end_node)
                        else:
                            ind_nj = None
                        #dTs_dm
                        dTs_dm += e.supply_temp_start(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.supply_temp_start_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        #dTs_dTs
                        dTs_dTsi += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsij_dTsi
                        dTs_dTsj += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsij_dTsj
                        #dTr_dm
                        dTr_dm -= e.return_temp_start(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.return_temp_start_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        #dTr_dTr
                        dTr_dTri -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrij_dTri
                        dTr_dTrj -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrij_dTrj
                        # adjustments
                        if ind_n in self.all_slack_ind:
                            dmhl_dm = -1
                            hl = n.half_links[0]
                            # adjustments to dm_dm
                            dm_dm[ind_n,ind_e] -= -dmhl_dm
                            # adjustments to dTs_dm
                            dTs_dm += dmhl_dm*hl.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTr_dm
                            dTr_dm -= dmhl_dm*hl.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        if ind_n in self.all_source_ind and total_outflow == 0: # non-slack node with only inflow in supply line
                            # adjustments to dTs_dm
                            dTs_dm -= n.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTs_dTsi
                            dTs_dTsi -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        if ind_n in self.all_sink_ind and total_inflow == 0: # non-slack node with only outflow in supply line
                            # adjustments to dTr_dm
                            dTr_dm += n.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTr_dTri
                            dTr_dTri += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # add data to dTs_dm
                        dTs_dm_data.append(dTs_dm)
                        dTs_dm_row.append(ind_n)
                        dTs_dm_col.append(ind_e)
                        # add data to dTr_dm
                        dTr_dm_data.append(dTr_dm)
                        dTr_dm_row.append(ind_n)
                        dTr_dm_col.append(ind_e)
                        if ind_nj != None: # Need to check if not None, because ind_nj = 0 needs to be taken into account
                            # add data to dTs_dTs
                            dTs_dTs_data.append(dTs_dTsj)
                            dTs_dTs_row.append(ind_n)
                            dTs_dTs_col.append(ind_nj)
                            # add data to dTr_dTr
                            dTr_dTr_data.append(dTr_dTrj)
                            dTr_dTr_row.append(ind_n)
                            dTr_dTr_col.append(ind_nj)
                    else: # HeatHalfLinks
                        dTshl_dTsi = e.supply_temp_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        dTrhl_dTri = e.return_temp_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        if e.sink and e.bc_type in [4,5,10,11,16,17,22,23]: # sinks with dT known. So m is actually not a function of any of the variables.
                            dmhl_dTsi = 0
                            dmhl_dTri = e.m_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            dTshl_dTri = 0
                            dTrhl_dTsi = 1
                        elif e.source and e.bc_type in [4,5,10,11,16,17,22,23]: # sources with dT known. So m is actually not a function of any of the variables.
                            dmhl_dTsi = e.m_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            dmhl_dTri = 0
                            dTshl_dTri = 1
                            dTrhl_dTsi = 0
                        else:
                            dmhl_dTsi = e.m_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            dmhl_dTri = e.m_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            dTshl_dTri = 0
                            dTrhl_dTsi = 0
                        # dm_dTs
                        dm_dTsi -= dmhl_dTsi
                        # dm_dTr
                        dm_dTri -= dmhl_dTri
                        # dTs_dTs
                        dTs_dTsi += dmhl_dTsi*e.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTshl_dTsi
                        # dTs_dTr
                        dTs_dTri += dmhl_dTri*e.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTshl_dTri
                        # dTr_dTr
                        dTr_dTri -= dmhl_dTri*e.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrhl_dTri
                        # dTr_dTs
                        dTr_dTsi -= dmhl_dTsi*e.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrhl_dTsi
                        # adjustments
                        if ind_n in self.all_source_ind and total_outflow == 0: # non-slack node with only inflow in supply line
                            # adjustments to dTs_dTsi
                            dTs_dTsi -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTs_dTri
                            dTs_dTri -= dm_dTri*n.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        if ind_n in self.all_sink_ind and total_inflow == 0: # non-slack node with only outflow in supply line
                            # adjustments to dTr_dTsi
                            dTr_dTsi += dm_dTsi*n.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTr_dTri
                            dTr_dTri += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                for e in n.get_in_links(): # only HeatLinks
                    dTr_dm = 0.
                    dTs_dm = 0.
                    dTs_dTsj = 0.
                    dTr_dTrj = 0.
                    ind_e = list(self.heatnetwork.get_links()).index(e)
                    # link goes from node j to node i
                    dTsji_dTsj,dTsji_dTsi = e.supply_temp_end_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    dTrji_dTrj,dTrji_dTri = e.return_temp_end_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    if e.start_node in list(self.heatnetwork.get_nodes()):
                        ind_nj = list(self.heatnetwork.get_nodes()).index(e.start_node)
                    else:
                        ind_nj = None
                    #dTs_dm
                    dTs_dm -= e.supply_temp_end(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.supply_temp_end_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    #dTs_dTs
                    dTs_dTsi -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsji_dTsi
                    dTs_dTsj -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsji_dTsj
                    #dTr_dm
                    dTr_dm += e.return_temp_end(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.return_temp_end_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    #dTr_dTr
                    dTr_dTri += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrji_dTri
                    dTr_dTrj += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrji_dTrj
                    # adjustments
                    if ind_n in self.all_slack_ind:
                        dmhl_dm = 1
                        hl = n.half_links[0]
                        # adjustments to dm_dm
                        dm_dm[ind_n,ind_e] -= -dmhl_dm
                        # adjustments to dTs_dm
                        dTs_dm += dmhl_dm*hl.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments to dTr_dm
                        dTr_dm -= dmhl_dm*hl.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    if ind_n in self.all_source_ind and total_outflow == 0: # non-slack node with only inflow in supply line
                        # adjustments to dTs_dm
                        dTs_dm += n.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments to dTs_dTsi
                        dTs_dTsi += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    if ind_n in self.all_sink_ind and total_inflow == 0: # non-slack node with only outflow in supply line
                        # adjustments to dTr_dm
                        dTr_dm -= n.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments to dTr_dTri
                        dTr_dTri -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    # add data to dTs_dm
                    dTs_dm_data.append(dTs_dm)
                    dTs_dm_row.append(ind_n)
                    dTs_dm_col.append(ind_e)
                    # add data to dTr_dm
                    dTr_dm_data.append(dTr_dm)
                    dTr_dm_row.append(ind_n)
                    dTr_dm_col.append(ind_e)
                    if ind_nj != None: # Need to check if not None, because ind_nj = 0 needs to be taken into account
                        # add data to dTs_dTs
                        dTs_dTs_data.append(dTs_dTsj)
                        dTs_dTs_row.append(ind_n)
                        dTs_dTs_col.append(ind_nj)
                        # add data to dTr_dTr
                        dTr_dTr_data.append(dTr_dTrj)
                        dTr_dTr_row.append(ind_n)
                        dTr_dTr_col.append(ind_nj)
                # add data to dm_dTs
                dm_dTs_data.append(dm_dTsi)
                dm_dTs_row.append(ind_n)
                dm_dTs_col.append(ind_n)
                # add data to dm_dTr
                dm_dTr_data.append(dm_dTri)
                dm_dTr_row.append(ind_n)
                dm_dTr_col.append(ind_n)
                # add data to dTs_dTs
                dTs_dTs_data.append(dTs_dTsi)
                dTs_dTs_row.append(ind_n)
                dTs_dTs_col.append(ind_n)
                # add data to dTs_dTr
                dTs_dTr_data.append(dTs_dTri)
                dTs_dTr_row.append(ind_n)
                dTs_dTr_col.append(ind_n)
                # add data to dTr_dTr
                dTr_dTr_data.append(dTr_dTri)
                dTr_dTr_row.append(ind_n)
                dTr_dTr_col.append(ind_n)
                # add data to dTr_dTs
                dTr_dTs_data.append(dTr_dTsi)
                dTr_dTs_row.append(ind_n)
                dTr_dTs_col.append(ind_n)
            # collect dm_dTs
            dm_dTs = sps.csr_matrix((dm_dTs_data,(dm_dTs_row,dm_dTs_col)),shape=(N,N))
            # collect dm_dTr
            dm_dTr = sps.csr_matrix((dm_dTr_data,(dm_dTr_row,dm_dTr_col)),shape=(N,N))
            # collect dTs_dm
            dTs_dm = sps.csr_matrix((dTs_dm_data,(dTs_dm_row,dTs_dm_col)),shape=(N,NE))
            # collect dTs_dTs
            dTs_dTs = sps.csr_matrix((dTs_dTs_data,(dTs_dTs_row,dTs_dTs_col)),shape=(N,N))
            # collect dTs_dTr
            dTs_dTr = sps.csr_matrix((dTs_dTr_data,(dTs_dTr_row,dTs_dTr_col)),shape=(N,N))
            # collect dTr_dm
            dTr_dm = sps.csr_matrix((dTr_dm_data,(dTr_dm_row,dTr_dm_col)),shape=(N,NE))
            # collect dTr_dTs
            dTr_dTs = sps.csr_matrix((dTr_dTs_data,(dTr_dTs_row,dTr_dTs_col)),shape=(N,N))
            # collect dTr_dTr
            dTr_dTr = sps.csr_matrix((dTr_dTr_data,(dTr_dTr_row,dTr_dTr_col)),shape=(N,N))

            # collect Jacobian
            if return_full:
                J = sps.bmat([
                    [dm_dm,None,dm_dTs,dm_dTr],
                    [dFL_dm,dFL_dp,None,None],
                    [dTs_dm,None,dTs_dTs,dTs_dTr],
                    [dTr_dm,None,dTr_dTs,dTr_dTr]],
                    format='csr')
            else:
                J = sps.bmat([
                    [dm_dm[self.Fm,:][:,self.xm],None,
                    dm_dTs[self.Fm,:][:,self.xTs],
                    dm_dTr[self.Fm,:][:,self.xTr]],
                    [dFL_dm[self.Fdeltap,:][:,self.xm],dFL_dp[self.Fdeltap,:][:,self.xp],None,None],
                    [dTs_dm[self.FTs,:][:,self.xm],None,
                    dTs_dTs[self.FTs,:][:,self.xTs],
                    dTs_dTr[self.FTs,:][:,self.xTr]],
                    [dTr_dm[self.FTr,:][:,self.xm],
                    None,dTr_dTs[self.FTr,:][:,self.xTs],
                    dTr_dTr[self.FTr,:][:,self.xTr]]],format='csr')
        elif self.heat_formulation == 'half_link_flow':
            NHL = len(self.heatnetwork.half_links)
            dm_dm = self.heatnetwork.A
            dm_dmhl_data = list()
            dm_dmhl_row = list()
            dm_dmhl_col = list()
            dp_der_data = list()
            dp_der_row = list()
            dp_der_col = list()
            dTs_dm_data = list()
            dTs_dm_row = list()
            dTs_dm_col = list()
            dTs_dmhl_data = list()
            dTs_dmhl_row = list()
            dTs_dmhl_col = list()
            dTs_dTs_data = list()
            dTs_dTs_row = list()
            dTs_dTs_col = list()
            dTs_dTshl_data = list()
            dTs_dTshl_row = list()
            dTs_dTshl_col = list()
            dTr_dm_data = list()
            dTr_dm_row = list()
            dTr_dm_col = list()
            dTr_dmhl_data = list()
            dTr_dmhl_row = list()
            dTr_dmhl_col = list()
            dTr_dTr_data = list()
            dTr_dTr_row = list()
            dTr_dTr_col = list()
            dTr_dTrhl_data = list()
            dTr_dTrhl_row = list()
            dTr_dTrhl_col = list()
            dphi_dmhl_vec = np.zeros(NHL)
            dphi_dTshl_vec = np.zeros(NHL)
            dphi_dTrhl_vec = np.zeros(NHL)
            dphi_dTs_data = list()
            dphi_dTs_row = list()
            dphi_dTs_col = list()
            dphi_dTr_data = list()
            dphi_dTr_row = list()
            dphi_dTr_col = list()
            ddT_dTshl_vec = np.zeros(NHL)
            ddT_dTrhl_vec = np.zeros(NHL)
            ddT_dTs_data = list()
            ddT_dTs_row = list()
            ddT_dTs_col = list()
            ddT_dTr_data = list()
            ddT_dTr_row = list()
            ddT_dTr_col = list()
            # Fill vectors
            T_shift = None
            # derivatives of nodal equations
            for ind_n,n in enumerate(self.heatnetwork.get_nodes()):
                dTs_dTsi = 0.
                dTr_dTri = 0.
                total_inflow = n.get_inflow(scale_var=self.scale_var,scale_var_params=self.scale_var_params) # loops over all links, but not halflinks
                total_outflow = n.get_outflow(scale_var=self.scale_var,scale_var_params=self.scale_var_params) # loops over all links, but not halflinks
                for e in n.get_out_links():
                    dTr_dm = 0.
                    dTs_dm = 0.
                    dTs_dTsj = 0.
                    dTr_dTrj = 0.
                    if e in self.heatnetwork.links: # HeatLinks
                        ind_e = list(self.heatnetwork.get_links()).index(e)
                        if e.end_node in list(self.heatnetwork.get_nodes()):
                            ind_nj = list(self.heatnetwork.get_nodes()).index(e.end_node)
                        else:
                            ind_nj = None
                        # link goes from node i to node j
                        dTsij_dTsi,dTsij_dTsj = e.supply_temp_start_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        dTrij_dTri,dTrij_dTrj = e.return_temp_start_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        #dTs_dm
                        dTs_dm += e.supply_temp_start(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.supply_temp_start_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        #dTs_dTs
                        dTs_dTsi += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsij_dTsi
                        dTs_dTsj += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsij_dTsj
                        #dTr_dm
                        dTr_dm -= e.return_temp_start(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.return_temp_start_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        #dTr_dTr
                        dTr_dTri -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrij_dTri
                        dTr_dTrj -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrij_dTrj
                        # adjustments
                        if ind_n in self.all_slack_ind:
                            dmhl_dm = -1
                            hl = n.half_links[0]
                            # adjustments to dm_dm
                            dm_dm[ind_n,ind_e] -= -dmhl_dm
                            # adjustments to dTs_dm
                            dTs_dm += dmhl_dm*hl.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTr_dm
                            dTr_dm -= dmhl_dm*hl.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        elif (ind_n in self.all_source_ind or ind_n in self.all_junction_ind) and total_outflow == 0: # non-slack node with only inflow in supply line
                            # adjustments to dTs_dm
                            dTs_dm -= n.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTs_dTsi
                            dTs_dTsi -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        elif (ind_n in self.all_sink_ind or ind_n in self.all_junction_ind) and total_inflow == 0: # non-slack node with only outflow in supply line
                            # adjustments to dTr_dm
                            dTr_dm += n.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTr_dTri
                            dTr_dTri += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # add data to dTs_dm
                        dTs_dm_data.append(dTs_dm)
                        dTs_dm_row.append(ind_n)
                        dTs_dm_col.append(ind_e)
                        # add data to dTr_dm
                        dTr_dm_data.append(dTr_dm)
                        dTr_dm_row.append(ind_n)
                        dTr_dm_col.append(ind_e)
                        if ind_nj != None: # Need to check if not None, because ind_nj = 0 needs to be taken into account
                            # add data to dTs_dTs
                            dTs_dTs_data.append(dTs_dTsj)
                            dTs_dTs_row.append(ind_n)
                            dTs_dTs_col.append(ind_nj)
                            # add data to dTr_dTr
                            dTr_dTr_data.append(dTr_dTrj)
                            dTr_dTr_row.append(ind_n)
                            dTr_dTr_col.append(ind_nj)
                    else: # HeatHalfLinks
                        ind_hl = list(self.heatnetwork.get_half_links()).index(e)
                        # add data to dm_dmhl
                        dm_dmhl_data.append(-1)
                        dm_dmhl_row.append(ind_n)
                        dm_dmhl_col.append(ind_hl)
                        # dTs_dmhl
                        dTs_dmhll = e.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # dTs_dTs
                        dTs_dTsi +=  e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.supply_temp_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # dTr_dmhl
                        dTr_dmhll = -e.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # dTr_dTr
                        dTr_dTri -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.return_temp_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments
                        if (ind_n in self.all_source_ind or ind_n in self.all_junction_ind) and (ind_n not in self.all_slack_ind) and total_outflow == 0: # non-slack node with only inflow in supply line
                            # adjustments to dTs_dTsi
                            dTs_dTsi -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTs_dmhl
                            dTs_dmhll -= n.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        elif (ind_n in self.all_sink_ind or ind_n in self.all_junction_ind) and (ind_n not in self.all_slack_ind) and total_inflow == 0: # non-slack node with only outflow in supply line
                            # adjustments to dTr_dTri
                            dTr_dTri += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                            # adjustments to dTr_dmhl
                            dTr_dmhll += n.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # add data to dTs_dmhl
                        dTs_dmhl_data.append(dTs_dmhll)
                        dTs_dmhl_row.append(ind_n)
                        dTs_dmhl_col.append(ind_hl)
                        # add data to dTs_dTshl
                        dTs_dTshl_data.append(e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        dTs_dTshl_row.append(ind_n)
                        dTs_dTshl_col.append(ind_hl)
                        # add data to dTr_dmhl
                        dTr_dmhl_data.append(dTr_dmhll)
                        dTr_dmhl_row.append(ind_n)
                        dTr_dmhl_col.append(ind_hl)
                        # add data to dTr_dTrhl
                        dTr_dTrhl_data.append(-e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        dTr_dTrhl_row.append(ind_n)
                        dTr_dTrhl_col.append(ind_hl)
                        # dphi_dmhl
                        dphi_dmhl_vec[ind_hl] = e.phi_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # dphi_dTshl
                        dphi_dTshl_vec[ind_hl] = e.phi_der_Tshl(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # dphi_dTrhl
                        dphi_dTrhl_vec[ind_hl] = e.phi_der_Trhl(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # add data to dphi_dTs
                        dphi_dTs_data.append(e.phi_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        dphi_dTs_row.append(ind_hl)
                        dphi_dTs_col.append(ind_n)
                        # add data to dphi_dTr
                        dphi_dTr_data.append(e.phi_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        dphi_dTr_row.append(ind_hl)
                        dphi_dTr_col.append(ind_n)
                        # ddT_dTshl
                        ddT_dTshl_vec[ind_hl] = e.ddT_der_Tshl(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # ddT_dTrhl
                        ddT_dTrhl_vec[ind_hl] = e.ddT_der_Trhl(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # add data to ddT_dTs
                        ddT_dTs_data.append(e.ddT_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        ddT_dTs_row.append(ind_hl)
                        ddT_dTs_col.append(ind_n)
                        # add data to ddT_dTr
                        ddT_dTr_data.append(e.ddT_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        ddT_dTr_row.append(ind_hl)
                        ddT_dTr_col.append(ind_n)
                for e in n.get_in_links(): # only HeatLinks
                    dTr_dm = 0.
                    dTs_dm = 0.
                    dTs_dTsj = 0.
                    dTr_dTrj = 0.
                    ind_e = list(self.heatnetwork.get_links()).index(e)
                    if e.start_node in list(self.heatnetwork.get_nodes()):
                        ind_nj = list(self.heatnetwork.get_nodes()).index(e.start_node)
                    else:
                        ind_nj = None
                    # link goes from node j to node i
                    dTsji_dTsj,dTsji_dTsi = e.supply_temp_end_der_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    dTrji_dTrj,dTrji_dTri = e.return_temp_end_der_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    #dTs_dm
                    dTs_dm -= e.supply_temp_end(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.supply_temp_end_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    #dTs_dTs
                    dTs_dTsi -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsji_dTsi
                    dTs_dTsj -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTsji_dTsj
                    #dTr_dm
                    dTr_dm += e.return_temp_end(scale_var=self.scale_var,scale_var_params=self.scale_var_params) + e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*e.return_temp_end_der_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    #dTr_dTr
                    dTr_dTri += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrji_dTri
                    dTr_dTrj += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)*dTrji_dTrj
                    # adjustments
                    if ind_n in self.all_slack_ind:
                        dmhl_dm = 1
                        hl = n.half_links[0]
                        # adjustments to dm_dm
                        dm_dm[ind_n,ind_e] -= -dmhl_dm
                        # adjustments to dTs_dm
                        dTs_dm += dmhl_dm*hl.supply_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments to dTr_dm
                        dTr_dm -= dmhl_dm*hl.return_temp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    elif (ind_n in self.all_source_ind or ind_n in self.all_junction_ind) and total_outflow == 0: # non-slack node with only inflow in supply line
                        # adjustments to dTs_dm
                        dTs_dm += n.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments to dTs_dTsi
                        dTs_dTsi += e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    elif (ind_n in self.all_sink_ind or ind_n in self.all_junction_ind) and total_inflow == 0: # non-slack node with only outflow in supply line
                        # adjustments to dTr_dm
                        dTr_dm -= n.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        # adjustments to dTr_dTri
                        dTr_dTri -= e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    # add data to dTs_dm
                    dTs_dm_data.append(dTs_dm)
                    dTs_dm_row.append(ind_n)
                    dTs_dm_col.append(ind_e)
                    # add data to dTr_dm
                    dTr_dm_data.append(dTr_dm)
                    dTr_dm_row.append(ind_n)
                    dTr_dm_col.append(ind_e)
                    # add data to dTs_dTs
                    if ind_nj != None: # Need to check if not None, because ind_nj = 0 needs to be taken into account
                        dTs_dTs_data.append(dTs_dTsj)
                        dTs_dTs_row.append(ind_n)
                        dTs_dTs_col.append(ind_nj)
                        # add data to dTr_dTr
                        dTr_dTr_data.append(dTr_dTrj)
                        dTr_dTr_row.append(ind_n)
                        dTr_dTr_col.append(ind_nj)
                # add data to dTs_dTs
                dTs_dTs_data.append(dTs_dTsi)
                dTs_dTs_row.append(ind_n)
                dTs_dTs_col.append(ind_n)
                # add data to dTr_dTr
                dTr_dTr_data.append(dTr_dTri)
                dTr_dTr_row.append(ind_n)
                dTr_dTr_col.append(ind_n)

            # collect dm_dmhl
            dm_dmhl = sps.csr_matrix((dm_dmhl_data,(dm_dmhl_row,dm_dmhl_col)),shape=(N,NHL))
            # collect dTs_dm
            dTs_dm = sps.csr_matrix((dTs_dm_data,(dTs_dm_row,dTs_dm_col)),shape=(N,NE))
            # collect dTs_dmhl
            dTs_dmhl = sps.csr_matrix((dTs_dmhl_data,(dTs_dmhl_row,dTs_dmhl_col)),shape=(N,NHL))
            # collect dTs_dTs
            dTs_dTs = sps.csr_matrix((dTs_dTs_data,(dTs_dTs_row,dTs_dTs_col)),shape=(N,N))
            # collect dTs_dTshl
            dTs_dTshl = sps.csr_matrix((dTs_dTshl_data,(dTs_dTshl_row,dTs_dTshl_col)),shape=(N,NHL))
            # collect dTr_dm
            dTr_dm = sps.csr_matrix((dTr_dm_data,(dTr_dm_row,dTr_dm_col)),shape=(N,NE))
            # collect dTr_dmhl
            dTr_dmhl = sps.csr_matrix((dTr_dmhl_data,(dTr_dmhl_row,dTr_dmhl_col)),shape=(N,NHL))
            # collect dTr_dTr
            dTr_dTr = sps.csr_matrix((dTr_dTr_data,(dTr_dTr_row,dTr_dTr_col)),shape=(N,N))
            # collect dTr_dTrhl
            dTr_dTrhl = sps.csr_matrix((dTr_dTrhl_data,(dTr_dTrhl_row,dTr_dTrhl_col)),shape=(N,NHL))
            # collect dphi_dmhl
            dphi_dmhl =  sps.diags(dphi_dmhl_vec).tocsr()
            # collect dphi_dTs
            dphi_dTs = sps.csr_matrix((dphi_dTs_data,(dphi_dTs_row,dphi_dTs_col)),shape=(NHL,N))
            # collect dphi_dTr
            dphi_dTr = sps.csr_matrix((dphi_dTr_data,(dphi_dTr_row,dphi_dTr_col)),shape=(NHL,N))
            # collect dphi_dTshl
            dphi_dTshl = sps.diags(dphi_dTshl_vec).tocsr()
            # collect dphi_dTrhl
            dphi_dTrhl = sps.diags(dphi_dTrhl_vec).tocsr()
            # collect dddT_dTs
            ddT_dTs = sps.csr_matrix((ddT_dTs_data,(ddT_dTs_row,ddT_dTs_col)),shape=(NHL,N))
            # collect ddT_dTr
            ddT_dTr = sps.csr_matrix((ddT_dTr_data,(ddT_dTr_row,ddT_dTr_col)),shape=(NHL,N))
            # collect ddT_dTshl
            ddT_dTshl = sps.diags(ddT_dTshl_vec).tocsr()
            # collect ddT_dTrhl
            ddT_dTrhl = sps.diags(ddT_dTrhl_vec).tocsr()

            # collect Jacobian
            if return_full:
                J = sps.bmat([
                    [dm_dm, dm_dmhl, None, None, None, None, None],
                    [dFL_dm, None, dFL_dp, None, None, None, None],
                    [dTs_dm, dTs_dmhl, None, dTs_dTs, None, dTs_dTshl, None],
                    [dTr_dm, dTr_dmhl, None, None, dTr_dTr, None, dTr_dTrhl],
                    [None, dphi_dmhl, None, dphi_dTs, dphi_dTr, dphi_dTshl, dphi_dTrhl],
                    [None, None, None, ddT_dTs, ddT_dTr, ddT_dTshl, ddT_dTrhl]
                    ],format='csr')
            else:
                J = sps.bmat([
                    [dm_dm[self.Fm,:][:,self.xm], dm_dmhl[self.Fm,:][:,self.xmhl], None, None, None, None, None],
                    [dFL_dm[self.Fdeltap,:][:,self.xm], None, dFL_dp[self.Fdeltap,:][:,self.xp], None, None, None, None],
                    [dTs_dm[self.FTs,:][:,self.xm], dTs_dmhl[self.FTs,:][:,self.xmhl], None, dTs_dTs[self.FTs,:][:,self.xTs], None, dTs_dTshl[self.FTs,:][:,self.xTshl], None],
                    [dTr_dm[self.FTr,:][:,self.xm], dTr_dmhl[self.FTr,:][:,self.xmhl], None, None, dTr_dTr[self.FTr,:][:,self.xTr], None, dTr_dTrhl[self.FTr,:][:,self.xTrhl]],
                    [None, dphi_dmhl[self.Fphi,:][:,self.xmhl], None, dphi_dTs[self.Fphi,:][:,self.xTs], dphi_dTr[self.Fphi,:][:,self.xTr], dphi_dTshl[self.Fphi,:][:,self.xTshl], dphi_dTrhl[self.Fphi,:][:,self.xTrhl]],
                    [None, None, None, ddT_dTs[self.FdT,:][:,self.xTs], ddT_dTr[self.FdT,:][:,self.xTr], ddT_dTshl[self.FdT,:][:,self.xTshl], ddT_dTrhl[self.FdT,:][:,self.xTrhl]]
                    ],format='csr')
        else:
            raise ValueError("Enter a valid formulation. It must be either 'standard' or 'half_link_flow', not {}".format(self.heat_formulation))
        return J

    def Dx(self):
        """Determines the diagonal scaling matrix for the variable vector x.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        Dx : sps matrix
            Diagonal matrix to scale x
        """
        if self.scale_var_params:
            xb_m = self.scale_var_params.get('mbase')*np.ones(len(self.unknown_m_links))
            xb_mhl = self.scale_var_params.get('mbase')*np.ones(len(self.unknown_m_halflinks))
            xb_ph = self.scale_var_params.get('phbase')*np.ones(len(self.unknown_p_nodes))
            xb_Ts = self.scale_var_params.get('Tbase')*np.ones(len(self.unknown_Ts_nodes))
            xb_Tr = self.scale_var_params.get('Tbase')*np.ones(len(self.unknown_Tr_nodes))
            xb_Tshl_h = self.scale_var_params.get('Tbase')*np.ones(len(self.unknown_hTs_halflinks))
            xb_Trhl_h = self.scale_var_params.get('Tbase')*np.ones(len(self.unknown_hTr_halflinks))
            xb_h = np.concatenate((xb_m,xb_mhl,xb_ph,xb_Ts,xb_Tr,xb_Tshl_h,xb_Trhl_h))
            Dx = sps.diags(1/xb_h)
        else:
            Dx = sps.eye(len(self.x_entries))
        return Dx

    def DF(self):
        """Determines the diagonal scaling matrix for the vector of equations F.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        DF : sps matrix
            Diagonal matrix to scale F
        """
        if self.scale_var_params:
            Fb_m = self.scale_var_params.get('mbase')*np.ones(len(self.F_m_nodes))
            Fb_deltap = self.scale_var_params.get('phbase')*np.ones(len(self.F_deltap_links))
            Fb_Ts = self.scale_var_params.get('mbase')*self.scale_var_params.get('Tbase')*np.ones(len(self.F_Ts_nodes))
            Fb_Tr = self.scale_var_params.get('mbase')*self.scale_var_params.get('Tbase')*np.ones(len(self.F_Tr_nodes))
            Fb_phi = self.scale_var_params.get('phibase')*np.ones(len(self.F_phi_halflinks))
            Fb_dT = self.scale_var_params.get('Tbase')*np.ones(len(self.F_dT_halflinks))
            Fb_h = np.concatenate((Fb_m,Fb_deltap,Fb_Ts,Fb_Tr,Fb_phi,Fb_dT))
            DF = sps.diags(1/Fb_h)
        else:
            DF = sps.eye(len(self.F_entries))
        return DF

    def plot_J_overlay(self,ax,**kwargs):
        """Lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.

        Parameters
        ----------
        ax : matplotlib axes
            Axes the lines are to be plotted on.
        x : np array
            Variable vector, possible scaled.
        """
        F_len = len(self.F_entries)
        x_len = len(self.x_entries)
        xh_m_len = len(self.unknown_m_links)
        xh_mhl_len = len(self.unknown_m_halflinks)
        xh_h_len = len(self.unknown_p_nodes)
        xh_Ts_len = len(self.unknown_Ts_nodes)
        xh_Tr_len = len(self.unknown_Tr_nodes)
        xh_Tshl_len = len(self.unknown_hTs_halflinks)
        xh_Trhl_len = len(self.unknown_hTr_halflinks)
        Fh_m_len = len(self.F_m_nodes)
        Fh_dp_len = len(self.F_deltap_links)
        Fh_Ts_len = len(self.F_Ts_nodes)
        Fh_Tr_len = len(self.F_Tr_nodes)
        Fh_phi_len = len(self.F_phi_halflinks)
        Fh_dT_len = len(self.F_dT_halflinks)
        # vertical
        ax.plot((xh_m_len-0.5,xh_m_len-0.5),(-0.5,F_len-0.5),'b--')
        ax.text(xh_m_len/2-0.5,-1.5,'$m^l$')
        if xh_mhl_len:
            ax.plot((xh_m_len+xh_mhl_len-0.5,xh_m_len+xh_mhl_len-0.5),(-0.5,F_len-0.5),'b--')
            ax.text(xh_m_len+xh_mhl_len/2-0.5,-1.5,r'$m^{hl}$')
        ax.plot((xh_m_len+xh_mhl_len+xh_h_len-0.5,xh_m_len+xh_mhl_len+xh_h_len-0.5),(-0.5,F_len-0.5),'b--')
        ax.text(xh_m_len+xh_mhl_len+xh_h_len/2-0.5,-1.5,r'$p$')
        ax.plot((xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len-0.5,xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len-0.5),(-0.5,F_len-0.5),'b--')
        ax.text(xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len/2-0.5,-1.5,r'$T^s$')
        ax.text(xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len/2-0.5,-1.5,r'$T^r$')
        if xh_Tshl_len:
            ax.plot((xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len-0.5,xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len-0.5),(-0.5,F_len-0.5),'b--')
            ax.text(xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len/2-0.5,-1.5,r'$T^s_{hl}$')
        if xh_Trhl_len:
            ax.plot((xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len-0.5,xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len+xh_Trhl_len-0.5),(-0.5,F_len-0.5),'b--')
            ax.text(xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len+xh_Trhl_len/2-0.5,-1.5,r'$T^r_{hl}$')
        # horizontal
        ax.plot((-0.5,x_len-0.5),(Fh_m_len-0.5,Fh_m_len-0.5),'b--')
        ax.text(-2.5,Fh_m_len/2-0.5,r'$F^m$')
        ax.plot((-0.5,x_len-0.5),(Fh_m_len+Fh_dp_len-0.5,Fh_m_len+Fh_dp_len-0.5),'b--')
        ax.text(-2.5,Fh_m_len+Fh_dp_len/2-0.5,r'$F^l$')
        ax.plot((-0.5,x_len-0.5),(Fh_m_len+Fh_dp_len+Fh_Ts_len-0.5,Fh_m_len+Fh_dp_len+Fh_Ts_len-0.5),'b--')
        ax.text(-2.5,Fh_m_len+Fh_dp_len+Fh_Ts_len/2-0.5,r'$F^{Ts}$')
        ax.text(-2.5,Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len/2-0.5,r'$F^{Tr}$')
        if Fh_phi_len:
            ax.plot((-0.5,x_len-0.5),(Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len-0.5,Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len-0.5),'b--')
            ax.text(-2.5,Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+Fh_phi_len/2-0.5,r'$F^\varphi$')
        if Fh_dT_len:
            ax.plot((-0.5,x_len-0.5),(Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+F_phi_len-0.5,Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+F_phi_len-0.5),'b--')
            ax.text(-2.5,Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+Fh_phi_len+F_dT_len/2-0.5,r'$F^{\Delta T}$')

# ===========================================================================
class NonLinearSystemHeterogeneous(NonLinearSystem):
    """Class that creates the (non-linear) system of equations F(x)=0
    and the corresponding Jacobian matrix J(x) for a heterogeneous network.
    NB: THE ASSUMPTION RIGHT NOW IS THAT A HETEROGENEOUS NETWORK CONSIST OF ONLY ONE GAS NETWORK, ONE ELECTRICAL NETWORK AND ONE HEAT NETWORK
    """
    def __init__(self,hetnetwork,formulation={'gas':'nodal','elec':'complex_power','heat':'standard','het':None},scale_var=None,scale_var_params=None,**kwargs):
        """Creates a NonLinearSystemHeterogeneous object

        Parameters
        ----------
        hetnetwork : HeterogeneousNetwork
            Heterogeneous network for which the sytem of equations is made.
        formulation : string, optional
            formulation used to form the system of equations. Default is None
        scale_var : string, optional
            How to scale the variable. Default is no scaling
        scale_var_params : dict, optional
            Dictionary with parameters needed for scaling.
        """
        self.hetnetwork = hetnetwork
        self.formulation = formulation
        self.scale_var = scale_var
        self.scale_var_params = scale_var_params
        self.x_entries, self.xg_entries, self.xe_entries, self.unknown_delta_nodes, self.unknown_V_nodes, self.xh_entries, self.unknown_m_links, self.unknown_m_halflinks, self.unknown_p_nodes, self.unknown_Ts_nodes, self.unknown_Tr_nodes, self.unknown_Ts_halflinks, self.unknown_Tr_halflinks, self.xc_entries, self.unknown_qc_links, self.unknown_qc_halflinks, self.unknown_Pc_links, self.unknown_Pc_halflinks, self.unknown_Qc_links, self.unknown_Qc_halflinks, self.unknown_mc_links, self.unknown_mc_halflinks, self.unknown_dphi_links, self.unknown_dphic_halflinks, self.unknown_Ts_links, self.unknown_Tsc_halflinks, self.unknown_Tr_links, self.unknown_Trc_halflinks  = self.hetnetwork.get_x_entries(formulation=self.formulation)
        self.F_entries, self.Fg_entries, self.Fe_entries, self.known_P_nodes, self.known_Q_nodes, self.Fh_entries, self.F_m_nodes, self.F_deltap_links, self.F_Ts_nodes, self.F_Tr_nodes, self.F_phi_halflinks, self.F_dT_halflinks, self.Fc_entries, self.F_fc_nodes, self.F_fc_amount, self.F_phi_nodes, self.F_dT_nodes = self.hetnetwork.get_F_entries(formulation=self.formulation)
        self.nlsystemsg = list()
        self.nlsystemse = list()
        self.nlsystemsh = list()
        for net in self.hetnetwork.get_networks('gas'):
            nlsysg = NonLinearSystemGas(net,formulation=self.formulation['gas'],scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            self.nlsystemsg.append(nlsysg)
            self.ind_xg_p = nlsysg.ind_p
            self.ind_xg_q = nlsysg.ind_q
            self.ind_Fg_node = nlsysg.ind_Fn
            self.ind_Fg_link= nlsysg.ind_Fl
        for net in self.hetnetwork.get_networks('elec'):
            nlsyse = NonLinearSystemElectrical(net,formulation=self.formulation['elec'],scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            self.nlsystemse.append(nlsyse)
        for net in self.hetnetwork.get_networks('heat'):
            nlsysh = NonLinearSystemHeat(net,formulation=self.formulation['heat'],scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            self.nlsystemsh.append(nlsysh)

    def F(self,x):
        """Determines F(x) for a heterogeneous network.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        F : np array
            (non-linear) system of equations evaluated at x
        """
        self.hetnetwork.update(x,formulation=self.formulation,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        F = np.zeros(len(self.Fg_entries+self.Fe_entries+self.Fh_entries+self.F_phi_nodes+self.F_dT_nodes)+np.sum(self.F_fc_amount))
        # homogeneous part
        for nlsysg in self.nlsystemsg:
            F[0:len(self.Fg_entries)] = nlsysg.F(x[0:len(self.xg_entries)])
        for nlsyse in self.nlsystemse:
            F[len(self.Fg_entries):len(self.Fg_entries)+len(self.Fe_entries)] = nlsyse.F(x[len(self.xg_entries):len(self.xg_entries)+len(self.xe_entries)])
        for nlsysh in self.nlsystemsh:
            F[len(self.Fg_entries)+len(self.Fe_entries):len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries)] = nlsysh.F(x[len(self.xg_entries)+len(self.xe_entries):len(self.xg_entries)+len(self.xe_entries)+len(self.xh_entries)])
        # heterogeneous (coupling) part
        for ind_el,el in enumerate(self.F_fc_nodes):
            ind = ind_el+len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries)
            if ind_el>0:
                ind += np.sum(self.F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has increased 1
            fc_len = self.F_fc_amount[ind_el]
            F[ind:ind+fc_len] = el.node_law(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        for ind_el,el in enumerate(self.F_phi_nodes):
            F[ind_el+len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries)+np.sum(self.F_fc_amount)] = el.heat_power_eq(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        for ind_el,el in enumerate(self.F_dT_nodes):
            F[ind_el+len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries)+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)] = el.temp_diff_equation(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        return F

    def J(self,x,return_full=False):
        """Determines J(x) for a heterogeneous network, based on analytical expressions.

        Parameters
        ----------
        x : np array
            Variable vector, possible scaled.

        Returns
        -------
        J : np array
            Jacobian matrix evaluated at x
        """
        self.hetnetwork.update(x,formulation=self.formulation,scale_var=self.scale_var,scale_var_params=self.scale_var_params)
        # homogeneous part (diagonal blocks in Jacobian)
        Jgg = None
        Jee = None
        Jhh = None
        Jcc = None
        for nlsysg in self.nlsystemsg:
            Jgg = nlsysg.J(x[0:len(self.xg_entries)])
        for nlsyse in self.nlsystemse:
            Jee = nlsyse.J(x[len(self.xg_entries):len(self.xg_entries)+len(self.xe_entries)])
        for nlsysh in self.nlsystemsh:
            T_shift = None#net.get_Ta(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            Jhh = nlsysh.J(x[len(self.xg_entries)+len(self.xe_entries):len(self.xg_entries)+len(self.xe_entries)+len(self.xh_entries)])

        # coupling part (diagonal block in Jacobian)
        Jcc_row = list()
        Jcc_col = list()
        Jcc_data = list()
        for ind_n, n in enumerate(self.F_fc_nodes):
            if self.F_fc_amount[ind_n]>1: # more than one coupling equation for this node
                dfc_dE = n.der_node_law_dE(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                dfc_dq = dfc_dE[:,0]
                dfc_dP = dfc_dE[:,1]
                dfc_dphi = dfc_dE[:,2]
            else:
                dfc_dq,dfc_dP,dfc_dphi = n.der_node_law_dE(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                dfc_dq = np.array([dfc_dq])
                dfc_dP = np.array([dfc_dP])
                dfc_dphi = np.array([dfc_dphi])
            for e in n.get_links():
                if e in self.unknown_qc_links:
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(self.unknown_qc_links.index(e))
                        Jcc_data.append(dfc_dq[ind_fc])
                elif e in self.unknown_qc_halflinks:
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+self.unknown_qc_halflinks.index(e))
                        Jcc_data.append(dfc_dq[ind_fc])
                elif e in self.unknown_Pc_links:
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+self.unknown_Pc_links.index(e))
                        Jcc_data.append(dfc_dP[ind_fc])
                elif e in self.unknown_Pc_halflinks:
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+self.unknown_Pc_halflinks.index(e))
                        Jcc_data.append(dfc_dP[ind_fc])
                elif e in self.unknown_dphi_links:
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+self.unknown_dphi_links.index(e))
                        Jcc_data.append(dfc_dphi[ind_fc])
                elif e in self.unknown_dphic_halflinks:
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+self.unknown_dphic_halflinks.index(e))
                        Jcc_data.append(dfc_dphi[ind_fc])
                if e in self.unknown_Ts_links:
                    dfc_dTo = n.der_node_law_dTs(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    if self.F_fc_amount[ind_n]==1: # only one coupling equation
                        dfc_dTo = np.array([dfc_dTo])
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+self.unknown_Ts_links.index(e))
                        Jcc_data.append(dfc_dTo[ind_fc])
                elif e in self.unknown_Tsc_halflinks:
                    dfc_dTo = n.der_node_law_dTs(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                    if self.F_fc_amount[ind_n]==1: # only one coupling equation
                        dfc_dTo = np.array([dfc_dTo])
                    for ind_fc in range(self.F_fc_amount[ind_n]):
                        Jcc_row.append(ind_fc+np.sum(self.F_fc_amount[0:ind_n]))
                        Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+self.unknown_Tsc_halflinks.index(e))
                        Jcc_data.append(dfc_dTo[ind_fc])
        for ind_n, n in enumerate(self.F_phi_nodes):
            ind_F = ind_n+np.sum(self.F_fc_amount)
            dfphi_dm,dfphi_dphi,dfphi_dT = n.der_heat_power_eq_dE(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
            for e in n.get_links():
                if e in self.unknown_mc_links:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+self.unknown_mc_links.index(e))
                    Jcc_data.append(dfphi_dm)
                elif e in self.unknown_mc_halflinks:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+self.unknown_mc_halflinks.index(e))
                    Jcc_data.append(dfphi_dm)
                if e in self.unknown_dphi_links:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+self.unknown_dphi_links.index(e))
                    Jcc_data.append(dfphi_dphi)
                elif e in self.unknown_dphic_halflinks:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+self.unknown_dphic_halflinks.index(e))
                    Jcc_data.append(dfphi_dphi)
                if e in self.unknown_Ts_links:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+self.unknown_Ts_links.index(e))
                    Jcc_data.append(dfphi_dT)
                elif e in self.unknown_Tsc_halflinks:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+self.unknown_Tsc_halflinks.index(e))
                    Jcc_data.append(dfphi_dT)
        for ind_n, n in enumerate(self.F_dT_nodes):
            ind_F = ind_n+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)
            for e in n.get_out_links():
                if e in self.unknown_Ts_links:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+self.unknown_Ts_links.index(e))
                    Jcc_data.append(1)
                elif e in self.unknown_Tsc_halflinks:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+self.unknown_Tsc_halflinks.index(e))
                    Jcc_data.append(1)
                if e in self.unknown_Trc_halflinks:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)+len(self.unknown_Tr_links)+self.unknown_Trc_halflinks.index(e))
                    Jcc_data.append(-1)
                elif e in self.unknown_Tr_links:
                    Jcc_row.append(ind_F)
                    Jcc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)+self.unknown_Tr_links.index(e))
                    Jcc_data.append(-1)
        if Jcc_data:
            Jcc = sps.csr_matrix((Jcc_data,(Jcc_row,Jcc_col)),shape=(np.sum(self.F_fc_amount)+len(self.F_phi_nodes)+len(self.F_dT_nodes),len(self.xc_entries)))

        # off-diagonal matrices
        if Jcc_data:
            Jgc_row = list()
            Jgc_col = list()
            Jgc_data = list()
            Jec_row = list()
            Jec_col = list()
            Jec_data = list()
            Jhc_row = list()
            Jhc_col = list()
            Jhc_data = list()
            Jch_row = list()
            Jch_col = list()
            Jch_data = list()
        else:
            Jgc_data = None
            Jec_data = None
            Jhc_data = None
            Jch_data = None
        for ind_n,n in enumerate(self.F_fc_nodes):
            for e in n.get_out_links():
                if not 'Half' in type(e).__name__: # half links don't have end nodes
                    if (e.end_node in self.Fg_entries and e in self.unknown_qc_links):
                        # dFg_node/dq
                        Jgc_row.append(self.Fg_entries.index(e.end_node))
                        Jgc_col.append(self.unknown_qc_links.index(e))
                        Jgc_data.append(1.)
                    elif e.end_node in self.Fe_entries:
                        if (e.end_node in self.known_P_nodes and e in self.unknown_Pc_links):
                            # dFP/dPc
                            Jec_row.append(self.known_P_nodes.index(e.end_node))
                            Jec_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+self.unknown_Pc_links.index(e))
                            Jec_data.append(-1.)
                        if (e.end_node in self.known_Q_nodes and e in self.unknown_Qc_links):
                            # dFQ/dQc
                            Jec_row.append(len(self.known_P_nodes)+self.known_Q_nodes.index(e.end_node))
                            Jec_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+self.unknown_Qc_links.index(e))
                            Jec_data.append(-1.)
                    elif e.end_node in self.Fh_entries:
                        if e.end_node in self.F_m_nodes:
                            # dFm/dmc
                            Jhc_row.append(self.F_m_nodes.index(e.end_node))
                            Jhc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+
                                           len(self.unknown_Pc_links)+
                                           len(self.unknown_Qc_links)+
                                           self.unknown_mc_links.index(e))
                            Jhc_data.append(1.)
                        if e.end_node in self.F_Ts_nodes:
                            # dFTs/dmc
                            Jhc_row.append(len(self.F_m_nodes)+len(self.F_deltap_links)+self.F_Ts_nodes.index(e.end_node))
                            Jhc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+
                                           len(self.unknown_Pc_links)+
                                           len(self.unknown_Qc_links)+
                                           self.unknown_mc_links.index(e))
                            if e.end_node.get_outflow(scale_var=self.scale_var,scale_var_params=self.scale_var_params) == 0 and (e.end_node.node_type in [2,5,6,7] or e.end_node.half_links[0].source):# only inflow from links, and a source or junction node
                                Jhc_data.append(e.end_node.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params) - e.get_Tsstart(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                            elif e.end_node.node_type == 8: # heat load slack node
                                Jhc_data.append(e.end_node.get_Ts(scale_var=self.scale_var,scale_var_params=self.scale_var_params,T_shift=T_shift))
                            else:
                                Jhc_data.append(- e.get_Tsstart(scale_var=self.scale_var,scale_var_params=self.scale_var_params,T_shift=T_shift))
                            if e in self.unknown_Ts_links:
                                #dFTs/dToc
                                Jhc_row.append(len(self.F_m_nodes)+len(self.F_deltap_links)+self.F_Ts_nodes.index(e.end_node))
                                Jhc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Qc_links)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+self.unknown_Ts_links.index(e))
                                Jhc_data.append(-e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                            for hl in n.get_half_links():
                                if hl in self.unknown_Tsc_halflinks:
                                    #dFTs/dToc
                                    Jhc_row.append(len(self.F_m_nodes)+len(self.F_deltap_links)+self.F_Ts_nodes.index(e.end_node))
                                    Jhc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Qc_links)+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+self.unknown_Tsc_halflinks.index(hl))
                                    Jhc_data.append(-e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params))
                        if e.end_node in self.F_Tr_nodes:
                            # dFTr/dmc
                            Jhc_row.append(len(self.F_m_nodes)+len(self.F_deltap_links)+len(self.F_Ts_nodes)+self.F_Tr_nodes.index(e.end_node))
                            Jhc_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+len(self.unknown_Qc_links)+self.unknown_mc_links.index(e))
                            if e.end_node.node_type == 0: # heat source slack node
                                Jhc_data.append(0.)
                            else:
                                Jhc_data.append(e.end_node.get_Tr(scale_var=self.scale_var,scale_var_params=self.scale_var_params,T_shift=T_shift))
            for e in n.get_in_links():
                if (e.start_node in self.Fg_entries and e in self.unknown_qc_links):
                    # dFg_node/dq
                    Jgc_row.append(self.Fg_entries.index(e.start_node))
                    Jgc_col.append(self.unknown_qc_links.index(e))
                    Jgc_data.append(-1.)
                elif e.start_node in self.Fe_entries:
                    if (e.start_node in self.known_P_nodes and e in self.unknown_Pc_links):
                        # dFP/dPc
                        Jec_row.append(self.known_P_nodes.index(e.start_node))
                        Jec_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+self.unknown_Pc_links.index(e))
                        Jec_data.append(1.)
                    if (e.start_node in self.known_Q_nodes and e in self.unknown_Qc_links):
                        # dFQ/dQc
                        Jec_row.append(len(self.known_P_nodes)+self.known_Q_nodes.index(e.start_node))
                        Jec_col.append(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)+len(self.unknown_Pc_links)+self.unknown_Qc_links.index(e))
                        Jec_data.append(1.)
                elif e.start_node in self.Fh_entries:
                    raise NotImplementedError('Jhc not implemented for incoming heat links!')
        for ind_n,n in enumerate(self.F_phi_nodes):
            for e in n.get_out_links():
                if not 'Half' in type(e).__name__: # half links don't have end nodes
                    if e.end_node in self.unknown_Tr_nodes:
                        #dFphi/dTr
                        Eout = -e.get_dphistart(scale_var=self.scale_var,scale_var_params=self.scale_var_params) # -phi because it is a source, so phi and m will be <0 by definition.
                        Cp = e.link_params.get('carrier').get_Cp(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        Ts = e.get_Tsstart(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        m = e.get_m(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        Tr = e.get_Trstart(scale_var=self.scale_var,scale_var_params=self.scale_var_params)
                        Jch_row.append(ind_n+np.sum(self.F_fc_amount))
                        Jch_col.append(len(self.unknown_m_links)+len(self.unknown_m_halflinks)+len(self.unknown_p_nodes)+len(self.unknown_Ts_nodes)+self.unknown_Tr_nodes.index(e.end_node))
                        Jch_data.append(n.dfphi_dTr(Eout,m,Ts,Tr,Cp))
        for ind_n,n in enumerate(self.F_dT_nodes):
            for e in n.get_out_links():
                if not 'Half' in type(e).__name__: # half links don't have end nodes
                    if e.end_node in self.unknown_Tr_nodes:
                        #dFTo/dTr
                        Jch_row.append(ind_n+np.sum(self.F_fc_amount)+len(self.F_phi_nodes))
                        Jch_col.append(len(self.unknown_m_links)+len(self.unknown_m_halflinks)+len(self.unknown_p_nodes)+len(self.unknown_Ts_nodes)+self.unknown_Tr_nodes.index(e.end_node))
                        Jch_data.append(-1)
        if Jgc_data:
            Jgc = sps.csr_matrix((Jgc_data,(Jgc_row,Jgc_col)),shape=(Jgg.shape[0],Jcc.shape[1]))
        else:
            Jgc = None
        if Jec_data:
            Jec = sps.csr_matrix((Jec_data,(Jec_row,Jec_col)),shape=(Jee.shape[0],Jcc.shape[1]))
        else:
            Jec = None
        if Jhc_data:
            Jhc = sps.csr_matrix((Jhc_data,(Jhc_row,Jhc_col)),shape=(Jhh.shape[0],Jcc.shape[1]))
            Jch = sps.csr_matrix((Jch_data,(Jch_row,Jch_col)),shape=(Jcc.shape[0],Jhh.shape[1]))
        else:
            Jhc = None
            Jch = None

        # construct Jacobian matrix
        J = sps.bmat([[Jgg,None,None,Jgc],
                      [None,Jee,None,Jec],
                      [None,None,Jhh,Jhc],
                      [None,None,Jch,Jcc]],format='csr')
        return J

    def Dx(self):
        """Determines the diagonal scaling matrix for the variable vector x.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        Dx : sps matrix
            Diagonal matrix to scale x
        """
        if self.scale_var_params:
            xb = np.zeros(len(self.x_entries))
            # homogeneous part
            for nlsysg in self.nlsystemsg:
                xb[0:len(self.xg_entries)] = 1/nlsysg.Dx().data[0]
            for nlsyse in self.nlsystemse:
                xb[len(self.xg_entries):len(self.xg_entries)+len(self.xe_entries)] = 1/nlsyse.Dx().data[0]
            for nlsysh in self.nlsystemsh:
                xb[len(self.xg_entries)+len(self.xe_entries):len(self.xg_entries)+len(self.xe_entries)+len(self.xh_entries)] = 1/nlsysh.Dx().data[0]
            # heterogeneous (coupling) gas part
            len_xsc = len(self.xg_entries)+len(self.xe_entries)+len(self.xh_entries)
            xb[len_xsc:len_xsc+len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)] = self.scale_var_params.get('qbase')*np.ones(len(self.unknown_qc_links)+len(self.unknown_qc_halflinks))
            len_xsc += len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)
            # heterogeneous (coupling) electrical part
            xb[len_xsc:len_xsc+len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)] = self.scale_var_params.get('Sbase')*np.ones(len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks))
            len_xsc += +len(self.unknown_Pc_links)+len(self.unknown_Pc_halflinks)
            xb[len_xsc:len_xsc+len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)] = self.scale_var_params.get('Sbase')*np.ones(len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks))
            len_xsc += len(self.unknown_Qc_links)+len(self.unknown_Qc_halflinks)
            # heterogeneous (coupling) heat part
            xb[len_xsc:len_xsc+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)] = self.scale_var_params.get('mbase')*np.ones(len(self.unknown_mc_links)+len(self.unknown_mc_halflinks))
            len_xsc += len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)
            xb[len_xsc:len_xsc+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)] = self.scale_var_params.get('phibase')*np.ones(len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks))
            len_xsc += len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)
            xb[len_xsc:len_xsc+len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)+len(self.unknown_Tr_links)+len(self.unknown_Trc_halflinks)] = self.scale_var_params.get('Tbase')*np.ones(len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)+len(self.unknown_Tr_links)+len(self.unknown_Trc_halflinks))
            Dx = sps.diags(1/xb)
        else:
            Dx = sps.eye(len(self.x_entries))
        return Dx

    def DF(self):
        """Determines the diagonal scaling matrix for the vector of equations F.
        If no scaling parameters are provided, the identity matrix is returned.

        Returns
        -------
        DF : sps matrix
            Diagonal matrix to scale F
        """
        if self.scale_var_params:
            Fb = np.zeros(len(self.Fg_entries+self.Fe_entries+self.Fh_entries+self.F_phi_nodes+self.F_dT_nodes)+np.sum(self.F_fc_amount))
            # homogeneous part
            for nlsysg in self.nlsystemsg:
                Fb[0:len(self.Fg_entries)] = 1/nlsysg.DF().data[0]
            for nlsyse in self.nlsystemse:
                Fb[len(self.Fg_entries):len(self.Fg_entries)+len(self.Fe_entries)] = 1/nlsyse.DF().data[0]
            for nlsysh in self.nlsystemsh:
                Fb[len(self.Fg_entries)+len(self.Fe_entries):len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries)] = 1/nlsysh.DF().data[0]
            # heterogeneous (coupling) part
            len_Fsc = len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries)
            for ind_el,el in enumerate(self.F_fc_nodes):
                ind = ind_el+len_Fsc
                if ind_el>0:
                    ind += np.sum(self.F_fc_amount[0:ind_el])-ind_el # -ind_el, because index is already shifted by one with respect to previous element because ind_el has
                if el.unit_type == 'gh_gas_boiler':
                    Fb[ind] = self.scale_var_params.get('phibase')
                elif el.unit_type == 'gh_gas_boiler_part_load':
                    Fb[ind] = self.scale_var_params.get('phibase')
                elif el.unit_type == 'ge_gas_fired_gen':
                    Fb[ind] = self.scale_var_params.get('Sbase')
                elif el.unit_type == 'ge_gas_fired_gen_valve_point':
                    Fb[ind] = self.scale_var_params.get('Ebase')
                elif el.unit_type == 'geh_CHP':
                    Fb[ind] = self.scale_var_params.get('Ebase')
                elif el.unit_type == 'geh_CHP_part_load':
                    Fb[ind] = self.scale_var_params.get('Ebase')
                    Fb[ind+1] = self.scale_var_params.get('Sbase')
                elif el.unit_type == 'EH':
                    Fb[ind] = self.scale_var_params.get('Sbase')
                    Fb[ind+1] = self.scale_var_params.get('phibase')
                else:
                    raise ValueError('Encountered a node type for which the function base is not specified.')
            Fb[len_Fsc+np.sum(self.F_fc_amount):len_Fsc+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)] = self.scale_var_params.get('phibase')*np.ones(len(self.F_phi_nodes))
            Fb[len_Fsc+np.sum(self.F_fc_amount)+len(self.F_phi_nodes):len_Fsc+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)+len(self.F_dT_nodes)] = self.scale_var_params.get('Tbase')*np.ones(len(self.F_dT_nodes))
            DF = sps.diags(1/Fb)
        else:
            DF = sps.eye(len(self.Fg_entries+self.Fe_entries+self.Fh_entries+self.F_phi_nodes+self.F_dT_nodes)+np.sum(self.F_fc_amount))
        return DF

    def plot_J_overlay(self,ax,P_F=np.array([]),P_x=np.array([])):
        """Lines to indicate the separate blocks.
        Horizontal lines indicate the end of a block with the derivative of F to that part of x.
        Vertical lines indicate the end of block with the derivative of that part of F to x.

        Parameters
        ----------
        ax : matplotlib axes
            Axes the lines are to be plotted on.
        P_F : array, optional
            Permutation matrix :math:`P_F`for the vector of equations :math:`F(x)`. This matrix is assumed to be an orthogonal binary matrix.
        P_x : array, optional
            Permutation matrix :math:`P_x`for the vector of variables :math:`x`. This matrix is assumed to be an orthogonal binary matrix.
        """
        Fg_len = len(self.Fg_entries)
        Fe_len = len(self.Fe_entries)
        Fh_len = len(self.Fh_entries)
        Fc_len = len(self.Fc_entries)
        F_len = Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)+len(self.F_dT_nodes)
        x_len = len(self.x_entries)
        xg_len = len(self.xg_entries)
        xe_len = len(self.xe_entries)
        xh_len = len(self.xh_entries)
        xc_len = len(self.xc_entries)

        # make adjustments based on permutation matrices
        if sps.issparse(P_F):
            P_F_len = P_F.shape[0]
        else:
            P_F_len = len(P_F)
        if P_F_len:
            PF_col = [col for row, col in zip(*P_F.nonzero())]
            Fg_len = F_len.copy()
            Fe_len = F_len.copy()
            Fh_len = F_len.copy()
            Fc_len = F_len.copy()
            F_diffs = np.where(np.array([j-i for i,j in zip(PF_col[:-1],PF_col[1:])])<0)
            for ind_F,F_diff in enumerate(F_diffs[0]):
                if ind_F == 0 and PF_col[F_diff+1] == len(self.Fg_entries):
                    Fg_len = F_diff + 1
                elif ind_F == 0:
                    Fg_len = len(self.Fg_entries)

                if ind_F <= 1 and PF_col[F_diff+1] == len(self.Fg_entries)+len(self.Fe_entries):
                    Fe_len = F_diff + 1
                elif ind_F <= 1 and len(self.Fh_entries): # there are also heat entries
                    Fe_len = Fg_len + len(self.Fe_entries)

                if ind_F <= 2 and PF_col[F_diff+1] == len(self.Fg_entries)+len(self.Fe_entries)+len(self.Fh_entries):
                    Fh_len = F_diff + 1
                elif ind_F <= 2:
                    Fh_len = F_len.copy()# Assuming permutation is done to get rid of coupling part. Otherwise: Fe_len + len(self.Fh_entries)
            Fe_len -= Fg_len
            Fh_len -= Fe_len + Fg_len
            Fc_len = F_len - Fg_len - Fe_len - Fh_len
        if sps.issparse(P_x):
            P_x_len = P_x.shape[0]
        else:
            P_x_len = len(P_x)
        if P_x_len:
            Px_col = [col for row, col in zip(*P_x.nonzero())]
            xg_len = len(self.x_entries)
            xe_len = len(self.x_entries)
            xh_len = len(self.x_entries)
            xc_len = len(self.x_entries)
            x_diffs = np.where(np.array([j-i for i,j in zip(Px_col[:-1],Px_col[1:])])<0)
            for ind,x_diff in enumerate(x_diffs[0]):
                if ind == 0 and Px_col[x_diff+1] == len(self.xg_entries):
                    xg_len = x_diff + 1
                elif ind == 0:
                    xg_len = len(self.xg_entries)

                if ind <= 1 and Px_col[x_diff+1] == len(self.xg_entries) + len(self.xe_entries):
                    xe_len = x_diff + 1
                elif ind <= 1 and len(self.xh_entries): # there are also heat entries
                    xe_len = xg_len + len(self.xe_entries)

                if ind <= 2 and Px_col[x_diff+1] == len(self.xg_entries) + len(self.xe_entries) + len(self.xh_entries):
                    xh_len = x_diff + 1
                elif ind <= 2:
                    xh_len = len(self.x_entries)# Assuming permutation is done to get rid of coupling part. Otherwise: xe_len + len(self.xh_entries)
            xe_len -= xg_len
            xh_len -= xe_len + xg_len
            xc_len = x_len - xg_len - xe_len - xh_len
        # vertical lines, which indicates the derivatives to x's
        if self.xg_entries:
            ax.plot((xg_len-0.5,xg_len-0.5),(0-0.5,F_len-0.5),'k-')
            ax.text(xg_len/2-0.5,-2.5,r'$x^g$', horizontalalignment='center',verticalalignment='center',color='g')
        if self.xe_entries:
            ax.plot((xg_len+xe_len-0.5,xg_len+xe_len-0.5),(0-0.5,F_len-0.5),'k-')
            ax.text(xg_len+xe_len/2-0.5,-1.5,r'$x^e$', horizontalalignment='center',verticalalignment='center',color='r')
        if self.xh_entries:
            ax.plot((xg_len+xe_len+xh_len-0.5,xg_len+xe_len+xh_len-0.5),(0-0.5,F_len-0.5),'k-')
            ax.text(xg_len+xe_len+xh_len/2-0.5,-1.5,r'$x^h$', horizontalalignment='center',verticalalignment='center',color='b')
        if self.xc_entries:
            ax.plot((xg_len+xe_len+xh_len+xc_len-0.5,xg_len+xe_len+xh_len+xc_len-0.5),(0-0.5,F_len-0.5),'k-')
            ax.text(xg_len+xe_len+xh_len+xc_len/2-0.5,-2.5,r'$x^c$', horizontalalignment='center',verticalalignment='center',color='k')
        # horizontal lines, which indicates the derivatives of F's
        if self.Fg_entries:
            ax.plot((0-0.5,x_len-0.5),(Fg_len-0.5,Fg_len-0.5),'k-')
            ax.text(-2.5,Fg_len/2-0.5,r'$F^g$', horizontalalignment='center',verticalalignment='center',color='g')
        if self.Fe_entries:
            ax.plot((0-0.5,x_len-0.5),(Fg_len+Fe_len-0.5,Fg_len+Fe_len-0.5),'k-')
            ax.text(-2.5,Fg_len+Fe_len/2-0.5,r'$F^e$', horizontalalignment='center',verticalalignment='center',color='r')
        if self.Fh_entries:
            ax.plot((0-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_len-0.5,Fg_len+Fe_len+Fh_len-0.5),'k-')
            ax.text(-2.5,Fg_len+Fe_len+Fh_len/2-0.5,r'$F^h$', horizontalalignment='center',verticalalignment='center',color='b')
        if Fc_len:
            ax.plot((0-0.5,x_len-0.5),(F_len-0.5,F_len-0.5),'k-')
            ax.text(-2.5,Fg_len+Fe_len+Fh_len+Fc_len/2-0.5,r'$F^c$', horizontalalignment='center',verticalalignment='center',color='k')

        if not(P_F_len or P_x_len):
            # lines with respect to specific gas variables and equations, in J_gg, J_gc, and J_cg
            if self.Fg_entries and self.formulation['gas'] == 'full':
                xg_p_len = len(self.ind_xg_p)
                xg_q_len = len(self.ind_xg_q)
                Fg_node_len = len(self.ind_Fg_node)
                Fg_link_len = len(self.ind_Fg_link)
                ax.plot((-0.5,x_len-0.5),(Fg_node_len-0.5,Fg_node_len-0.5),'g--')
                ax.text(-1.5,Fg_node_len/2-0.5,r'$F^q$', horizontalalignment='center',verticalalignment='center',color='g')
                ax.text(-1.5,Fg_node_len+Fg_link_len/2-0.5,r'$F^l$', horizontalalignment='center',verticalalignment='center',color='g')
                ax.plot((xg_q_len-0.5,xg_q_len-0.5),(0-0.5,F_len-0.5),'g--')
                ax.text(xg_q_len/2-0.5,-1.5,r'$q$', horizontalalignment='center',verticalalignment='center',color='g')
                ax.text(xg_q_len+xg_p_len/2-0.5,-1.5,r'$p$', horizontalalignment='center',verticalalignment='center',color='g')

            # lines with respect to specific heat variables and equations, in J_hh, J_hc, and J_ch
            if self.Fe_entries:
                x_delta_len = len(self.unknown_delta_nodes)
                x_V_len = len(self.unknown_V_nodes)
                F_P_len = len(self.known_P_nodes)
                F_Q_len = len(self.known_Q_nodes)
                # vertical
                ax.plot((xg_len+x_delta_len-0.5,xg_len+x_delta_len-0.5),(Fg_len-0.5,F_len-0.5),'r--')
                if x_delta_len:
                    ax.text(xg_len+x_delta_len/2-0.5,Fg_len-1.5,r'$\delta$', horizontalalignment='center',verticalalignment='center',color='r')
                if x_V_len:
                    ax.text(xg_len+x_delta_len+x_V_len/2-0.5,Fg_len-1.5,r'$|V|$', horizontalalignment='center',verticalalignment='center',color='r')
                # horizontal
                ax.plot((xg_len-0.5,x_len-0.5),(Fg_len+F_P_len-0.5,Fg_len+F_P_len-0.5),'r--')
                ax.text(xg_len-1.5,Fg_len+F_P_len/2-0.5,r'$F^P$', horizontalalignment='center',verticalalignment='center',color='r')
                ax.text(xg_len-1.5,Fg_len+F_P_len+F_Q_len/2-0.5,r'$F^Q$', horizontalalignment='center',verticalalignment='center',color='r')

            # lines with respect to specific heat variables and equations, in J_hh, J_hc, and J_ch
            if self.Fh_entries:
                xh_m_len = len(self.unknown_m_links)
                xh_mhl_len = len(self.unknown_m_halflinks)
                xh_h_len = len(self.unknown_p_nodes)
                xh_Ts_len = len(self.unknown_Ts_nodes)
                xh_Tr_len = len(self.unknown_Tr_nodes)
                xh_Tshl_len = len(self.unknown_Ts_halflinks)
                xh_Trhl_len = len(self.unknown_Tr_halflinks)
                Fh_m_len = len(self.F_m_nodes)
                Fh_dp_len = len(self.F_deltap_links)
                Fh_Ts_len = len(self.F_Ts_nodes)
                Fh_Tr_len = len(self.F_Tr_nodes)
                Fh_phi_len = len(self.F_phi_halflinks)
                Fh_dT_len = len(self.F_dT_halflinks)
                # vertical
                ax.plot((xg_len+xe_len+xh_m_len-0.5,xg_len+xe_len+xh_m_len-0.5),(Fg_len+Fe_len-0.5,F_len-0.5),'b--')
                ax.text(xg_len+xe_len+xh_m_len/2-0.5,Fg_len+Fe_len-1.5,'$m^l$', horizontalalignment='center',verticalalignment='center',color='b')
                if xh_mhl_len:
                    ax.plot((xg_len+xe_len+xh_m_len+xh_mhl_len-0.5,xg_len+xe_len+xh_m_len+xh_mhl_len-0.5),(Fg_len+Fe_len-0.5,F_len-0.5),'b--')
                    ax.text(xg_len+xe_len+xh_m_len+xh_mhl_len/2-0.5,Fg_len+Fe_len-1.5,r'$m^{hl}$', horizontalalignment='center',verticalalignment='center',color='b')
                ax.plot((xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len-0.5,xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len-0.5),(Fg_len+Fe_len-0.5,F_len-0.5),'b--')
                ax.text(xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len/2-0.5,Fg_len+Fe_len-1.5,r'$p$', horizontalalignment='center',verticalalignment='center',color='b')
                ax.plot((xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len-0.5,xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len-0.5),(Fg_len+Fe_len-0.5,F_len-0.5),'b--')
                ax.text(xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len/2-0.5,Fg_len+Fe_len-1.5,r'$T^s$', horizontalalignment='center',verticalalignment='center',color='b')
                ax.text(xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len/2-0.5,Fg_len+Fe_len-1.5,r'$T^r$', horizontalalignment='center',verticalalignment='center',color='b')
                if xh_Tshl_len:
                    ax.plot((xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len-0.5,xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len-0.5),(Fg_len+Fe_len-0.5,F_len-0.5),'b--')
                    ax.text(xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len/2-0.5,Fg_len+Fe_len-1.5,r'$T^s_{hl}$', horizontalalignment='center',verticalalignment='center',color='b')
                if xh_Trhl_len:
                    ax.plot((xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len-0.5,xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len+xh_Trhl_len-0.5),(Fg_len+Fe_len-0.5,F_len-0.5),'b--')
                    ax.text(xg_len+xe_len+xh_m_len+xh_mhl_len+xh_h_len+xh_Ts_len+xh_Tr_len+xh_Tshl_len+xh_Trhl_len/2-0.5,Fg_len+Fe_len-1.5,r'$T^s_{hl}$', horizontalalignment='center',verticalalignment='center',color='b')
                # horizontal
                ax.plot((xg_len+xe_len-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_m_len-0.5,Fg_len+Fe_len+Fh_m_len-0.5),'b--')
                ax.text(xg_len+xe_len-1.5,Fg_len+Fe_len+Fh_m_len/2-0.5,r'$F^m$', horizontalalignment='center',verticalalignment='center',color='b')
                ax.plot((xg_len+xe_len-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_m_len+Fh_dp_len-0.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len-0.5),'b--')
                ax.text(xg_len+xe_len-1.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len/2-0.5,r'$F^l$', horizontalalignment='center',verticalalignment='center',color='b')
                ax.plot((xg_len+xe_len-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len-0.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len-0.5),'b--')
                ax.text(xg_len+xe_len-1.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len/2-0.5,r'$F^{Ts}$', horizontalalignment='center',verticalalignment='center',color='b')
                ax.text(xg_len+xe_len-1.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len/2-0.5,r'$F^{Tr}$', horizontalalignment='center',verticalalignment='center',color='b')
                if Fh_phi_len:
                    ax.plot((xg_len+xe_len-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len-0.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len-0.5),'b--')
                    ax.text(xg_len+xe_len-1.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+Fh_phi_len/2-0.5,r'$F^\varphi$', horizontalalignment='center',verticalalignment='center',color='b')
                if Fh_dT_len:
                    ax.plot((xg_len+xe_len-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+Fh_phi_len-0.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+Fh_phi_len-0.5),'b--')
                    ax.text(xg_len+xe_len-1.5,Fg_len+Fe_len+Fh_m_len+Fh_dp_len+Fh_Ts_len+Fh_Tr_len+Fh_phi_len+Fh_dT_len/2-0.5,r'$F^{\Delta T}$', horizontalalignment='center',verticalalignment='center',color='b')

            # lines with respect to specific coupling variables and equations
            if self.Fc_entries:
                xc_g_len = len(self.unknown_qc_links)+len(self.unknown_qc_halflinks)
                xc_e_len = len(self.unknown_Pc_links)+len(self.unknown_Qc_links)
                # vertical
                if self.unknown_qc_links or self.unknown_qc_halflinks:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len-0.5,xg_len+xe_len+xh_len+xc_g_len-0.5),(0-0.5,F_len-0.5),'--g')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len/2-0.5,-1.5,r'$q^c$', horizontalalignment='center',verticalalignment='center',color='g')
                if self.unknown_Pc_links:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len+len(self.unknown_Pc_links)-0.5,xg_len+xe_len+xh_len+xc_g_len+len(self.unknown_Pc_links)-0.5),(0-0.5,F_len-0.5),'--r')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+len(self.unknown_Pc_links)/2-0.5,-1.5,r'$P^c$', horizontalalignment='center',verticalalignment='center',color='r')
                if self.unknown_Qc_links:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len+xc_e_len-0.5,xg_len+xe_len+xh_len+xc_g_len+xc_e_len-0.5),(0-0.5,F_len-0.5),'--r')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+len(self.unknown_Pc_links)+len(self.unknown_Qc_links)/2-0.5,-1.5,r'$Q^c$', horizontalalignment='center',verticalalignment='center',color='r')
                if self.unknown_mc_links:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)-0.5,xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)-0.5),(0-0.5,F_len-0.5),'--b')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)/2-0.5,-1.5,r'$m^c$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_mc_halflinks:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)-0.5,xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)-0.5),(0-0.5,F_len-0.5),'--b')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)/2-0.5,-1.5,r'$m^c_{hl}$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_dphi_links:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)-0.5,xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)-0.5),(0-0.5,F_len-0.5),'--b')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)/2-0.5,-1.5,r'$\varphi^c$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_dphic_halflinks:
                    ax.plot((xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)-0.5,xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)-0.5),(0-0.5,F_len-0.5),'--b')
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)/2-0.5,-1.5,r'$\varphi^c$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_Ts_links:
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)/2-0.5,-1.5,r'$T^{s,c}$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_Tsc_halflinks:
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)+len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)/2-0.5,-1.5,r'$T^{s,c}_{hl}$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_Tr_links:
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)++len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)+len(self.unknown_Tr_links)/2-0.5,-1.5,r'$T^{r,c}$', horizontalalignment='center',verticalalignment='center',color='b')
                if self.unknown_Trc_halflinks:
                    ax.text(xg_len+xe_len+xh_len+xc_g_len+xc_e_len+len(self.unknown_mc_links)+len(self.unknown_mc_halflinks)+len(self.unknown_dphi_links)+len(self.unknown_dphic_halflinks)++len(self.unknown_Ts_links)+len(self.unknown_Tsc_halflinks)+len(self.unknown_Trc_halflinks)/2-0.5,-1.5,r'$T^{r,c}_{hl}$', horizontalalignment='center',verticalalignment='center',color='b')
                # horizontal
                if self.F_fc_nodes:
                    ax.plot((0-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)-0.5,Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)-0.5),'--k')
                    ax.plot((0-0.5,x_len-0.5),(Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)-0.5,Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)-0.5),'--k')
                    ax.text(-1.5,Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)/2-0.5,r'$F^c$', horizontalalignment='center',verticalalignment='center',color='k')
                    ax.text(-1.5,Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)/2-0.5,r'$F^{\varphi^c}$', horizontalalignment='center',verticalalignment='center',color='k')
                    ax.text(-1.5,Fg_len+Fe_len+Fh_len+np.sum(self.F_fc_amount)+len(self.F_phi_nodes)+len(self.F_dT_nodes)/2-0.5,r'$F^{\Delta T^c}$', horizontalalignment='center',verticalalignment='center',color='k')

        # set ticks along axes
        ax.set_xticks([xg_len-1,xg_len+xe_len-1,xg_len+xe_len+xh_len-1,x_len-1])
        ax.set_yticks([Fg_len-1,Fg_len+Fe_len-1,Fg_len+Fe_len+Fh_len-1,F_len-1])
