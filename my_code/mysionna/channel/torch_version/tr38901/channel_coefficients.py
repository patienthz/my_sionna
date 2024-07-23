
"""
Class for sampling channel impulse responses following 3GPP TR38.901
specifications and giving LSPs and rays.
"""

import torch

from my_code.mysionna import PI, SPEED_OF_LIGHT
from torch import sin, cos, acos

class Topology:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------

    velocities : [batch size, number of UTs], torch.float
        UT velocities

    moving_end : str
        Indicated which end of the channel (TX or RX) is moving. Either "tx" or
        "rx".

    los_aoa : [batch size, number of BSs, number of UTs], torch.float
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], torch.float
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], torch.float
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], torch.float
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], torch.bool
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], torch.float
        Distance between the UTs in X-Y-Z space (not only X-Y plan).

    tx_orientations : [batch size, number of TXs, 3], torch.float
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian].

    rx_orientations : [batch size, number of RXs, 3], torch.float
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian].
    """

    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d


class ChannelCoefficientsGenerator:
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays.

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    tx_array : PanelArray
        Panel array used by the transmitters.
        All transmitters share the same antenna array configuration.

    rx_array : PanalArray
        Panel array used by the receivers.
        All transmitters share the same antenna array configuration.

    subclustering : bool
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (UMa,
        UMi, RMa) do.

    dtype : Complex torch.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `torch.complex64`.

    Input
    -----
    num_time_samples : int
        Number of samples

    sampling_frequency : float
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX]
        K-factor

    rays : Rays
        Rays from which to compute thr CIR

    topology : Topology
        Topology of the network

    c_ds : [batch size, number of TX, number of RX]
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : bool
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], torch.complex
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], torch.real
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], torch.real
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], torch.float
        Sampling time steps
    """

    def __init__(self,  carrier_frequency,
                        tx_array, rx_array,
                        subclustering,
                        dtype=torch.complex64):
        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        
        if dtype == torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("real_dtype must be in [torch.float16,torch.float32,torch.float64]")
        
        self._real_dtype = real_dtype
        
        # Wavelength (m)
        self._lambda_0 = torch.tensor(SPEED_OF_LIGHT/carrier_frequency,dtype=real_dtype)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = torch.tensor([0,1,2,3,4,5,6,7,18,19], torch.int32)
        self._sub_cl_2_ind = torch.tensor([8,9,10,11,16,17], torch.int32)
        self._sub_cl_3_ind = torch.tensor([12,13,14,15], torch.int32)
        self._sub_cl_delay_offsets = torch.tensor([0, 1.28, 2.56],
                                                    real_dtype)
        

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 topology, c_ds=None, debug=False):
        # Sample times
        sample_times = (torch.arange(num_time_samples,
                dtype=self._real_dtype)/sampling_frequency)

        if not isinstance(rays.aoa,torch.Tensor):
            rays_aoa =torch.tensor(rays.aoa)
        # Step 10
        phi = self._step_10(rays_aoa.shape)

        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times,
                                                                        c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, torch.float
            Zenith [radian]

        phi : Same shape as ``theta``, torch.float
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = torch.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], dim=-1)
        return torch.unsqueeze(rho_hat, dim=-1)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], torch.float
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], torch.float
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = torch.stack([cos(a)*cos(b),
            cos(a)*sin(b)*sin(c)-sin(a)*cos(c),
            cos(a)*sin(b)*cos(c)+sin(a)*sin(c)], dim=-1)

        row_2 = torch.stack([sin(a)*cos(b),
            sin(a)*sin(b)*sin(c)+cos(a)*cos(c),
            sin(a)*sin(b)*cos(c)-cos(a)*sin(c)], dim=-1)

        row_3 = torch.stack([-sin(b),
            cos(b)*sin(c),
            cos(b)*cos(c)], dim=-1)

        rot_mat = torch.stack([row_1, row_2, row_3], dim=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], torch.float
            Orientation to which to rotate [radian]

        positions : [...,3,1], torch.float
            Positions to rotate

        Output
        -------
        : [...,3,1], torch.float
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return torch.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], torch.float
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], torch.float
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = rot_mat.T
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, torch.float
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, torch.float
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, torch.float
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, torch.float
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, torch.float
            Rotated azimuth
        """

        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = torch.matmul(rot_inv, rho_hat)
        v1 = torch.tensor([0,0,1], self._real_dtype)
        v1 = torch.reshape(v1, [1]*(rot_rho.dim()-1)+[3])
        v2 = torch.tensor([1+0j,1j,0], self._dtype)
        v2 = torch.reshape(v2, [1]*(rot_rho.dim()-1)+[3])
        z = torch.matmul(v1, rot_rho)
        z = torch.clamp(z, torch.tensor(-1., self._real_dtype),
                             torch.tensor(1., self._real_dtype))
        theta_prime = acos(z)
        phi_prime = torch.angle((torch.matmul(v2, rot_rho.to(dtype=self._real_dtype))))
        theta_prime = torch.squeeze(theta_prime, dim=[phi.dim(),
            phi.dim()+1])
        phi_prime = torch.squeeze(phi_prime, dim=[phi.dim(),
            phi.dim()+1])

        return (theta_prime, phi_prime)
    
    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], torch.float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, torch.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, torch.float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, torch.float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = torch.angle(torch.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], torch.float
            Field components

        orientations : K-Dim Tensor of shape [...,3], torch.float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, torch.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, torch.float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], torch.float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = torch.stack([cos(psi), -sin(psi)], dim=-1)
        row2 = torch.stack([sin(psi), cos(psi)], dim=-1)
        mat = torch.stack([row1, row2], dim=-2)
        f = torch.matmul(mat, torch.unsqueeze(f_prime, -1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = torch.unsqueeze(tx_orientations, dim = 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = torch.reshape(tx_ant_pos_lcs,
            [1,1]+tx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = torch.reshape(tx_ant_pos_gcs,
            tx_ant_pos_gcs.shape[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = torch.unsqueeze(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = torch.reshape(rx_ant_pos_lcs,
            [1,1]+rx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = torch.reshape(rx_ant_pos_gcs,
            rx_ant_pos_gcs.shape[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], torch.float
            Phases for all polarization combinations
        """
        final_shape = list(shape) + [4]
        phi = (torch.rand(final_shape, dtype=self._real_dtype) * 2 * PI) - PI

        return phi

    def _step_11_phase_matrix(self, phi, rays):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with random phases in (7.5-22)

        Input
        -----
        phi : [batch size, num TXs, num RXs, num clusters, num rays, 4], torch.float
            Initial phases for all combinations of polarization

        rays : Rays
            Rays

        Output
        ------
        h_phase : [batch size, num TXs, num RXs, num clusters, num rays, 2, 2], torch.complex
            Matrix with random phases in (7.5-22)
        """
        xpr = rays.xpr

        xpr_scaling = torch.complex(torch.sqrt(1/xpr),
            torch.tensor(0., self._real_dtype))
        e0 = torch.exp(torch.complex(torch.tensor(0., self._real_dtype),
            phi[...,0]))
        e3 = torch.exp(torch.complex(torch.tensor(0., self._real_dtype),
            phi[...,3]))
        e1 = xpr_scaling*torch.exp(torch.complex(torch.tensor(0.,
                                self._real_dtype), phi[...,1]))
        e2 = xpr_scaling*torch.exp(torch.complex(torch.tensor(0.,
                                self._real_dtype), phi[...,2]))
        shape = torch.cat([e0.shape, [2,2]], dim=-1)
        h_phase = torch.reshape(torch.stack([e0, e1, e2, e3], dim=-1), shape)

        return h_phase