import torch

from torch import log10
from my_code.mysionna.channel.torch_version.utils import deg_2_rad,wrap_angle_0_360


class Rays:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing rays

    Parameters
    -----------

    delays : [batch size, number of BSs, number of UTs, number of clusters], torch.float
        Paths delays [s]

    powers : [batch size, number of BSs, number of UTs, number of clusters], torch.float
        Normalized path powers

    aoa : (batch size, number of BSs, number of UTs, number of clusters, number of rays], torch.float
        Azimuth angles of arrival [radian]

    aod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], torch.float
        Azimuth angles of departure [radian]

    zoa : [batch size, number of BSs, number of UTs, number of clusters, number of rays], torch.float
        Zenith angles of arrival [radian]

    zod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], torch.float
        Zenith angles of departure [radian]

    xpr [batch size, number of BSs, number of UTs, number of clusters, number of rays], torch.float
        Coss-polarization power ratios.
    """

    def __init__(self, delays, powers, aoa, aod, zoa, zod, xpr):
        self.delays = delays
        self.powers = powers
        self.aoa = aoa
        self.aod = aod
        self.zoa = zoa
        self.zod = zod
        self.xpr = xpr


class RaysGenerator:
    """
    Sample rays according to a given channel scenario and large scale
    parameters (LSP).

    This class implements steps 6 to 9 from the TR 38.901 specifications,
    (section 7.5).

    Note that a global scenario is set for the entire batches when instantiating
    this class (UMa, UMi, or RMa). However, each UT-BS link can have its
    specific state (LoS, NLoS, or indoor).

    The batch size is set by the ``scenario`` given as argument when
    constructing the class.

    Parameters
    ----------
    scenario : :class:`~sionna.channel.tr38901.SystemLevelScenario``
        Scenario used to generate LSPs

    Input
    -----
    lsp : :class:`~sionna.channel.tr38901.LSP`
        LSPs samples

    Output
    ------
    rays : :class:`~sionna.channel.tr38901.Rays`
        Rays samples
    """

    def __init__(self, scenario):
        # Scenario
        if not isinstance(scenario,torch.Tensor):
            scenario = torch.tensor(scenario)
        self._scenario = scenario

        dtype = scenario.dtype
        # For AoA, AoD, ZoA, and ZoD, offset to add to cluster angles to get ray
        # angles. This is hardcoded from table 7.5-3 for 3GPP 38.901
        # specification.
        if dtype == torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("real_dtype must be in [torch.float16,torch.float32,torch.float64]")
        
        self._real_dtype = real_dtype
        self._ray_offsets = torch.tensor([0.0447, -0.0447,
                                         0.1413, -0.1413,
                                         0.2492, -0.2492,
                                         0.3715, -0.3715,
                                         0.5129, -0.5129,
                                         0.6797, -0.6797,
                                         0.8844, -0.8844,
                                         1.1481, -0.1481,
                                         1.5195, -1.5195,
                                         2.1551, -2.1551],
                                         self._real_dtype)

    #########################################
    # Public methods and properties
    #########################################

    def __call__(self, lsp):
        # Sample cluster delays
        delays, delays_unscaled = self._cluster_delays(lsp.ds, lsp.k_factor)

        # Sample cluster powers
        powers, powers_for_angles_gen = self._cluster_powers(lsp.ds,
                                            lsp.k_factor, delays_unscaled)

        # Sample AoA
        aoa = self._azimuth_angles_of_arrival(lsp.asa, lsp.k_factor,
                                                powers_for_angles_gen)

        # Sample AoD
        aod = self._azimuth_angles_of_departure(lsp.asd, lsp.k_factor,
                                                powers_for_angles_gen)

        # Sample ZoA
        zoa = self._zenith_angles_of_arrival(lsp.zsa, lsp.k_factor,
                                                powers_for_angles_gen)

        # Sample ZoD
        zod = self._zenith_angles_of_departure(lsp.zsd, lsp.k_factor,
                                                powers_for_angles_gen)

        # XPRs
        xpr = self._cross_polarization_power_ratios()

        # Random coupling
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        # Convert angles of arrival and departure from degree to radian
        aoa = deg_2_rad(aoa)
        aod = deg_2_rad(aod)
        zoa = deg_2_rad(zoa)
        zod = deg_2_rad(zod)

        # Storing and returning rays
        rays = Rays(delays = delays,
                    powers = powers,
                    aoa    = aoa,
                    aod    = aod,
                    zoa    = zoa,
                    zod    = zod,
                    xpr    = xpr)

        return rays


    def topology_updated_callback(self):
        """
        Updates internal quantities. Must be called at every update of the
        scenario that changes the state of UTs or their locations.

        Input
        ------
        None

        Output
        ------
        None
        """
        self._compute_clusters_mask()

    ########################################
    # Internal utility methods
    ########################################

    def _compute_clusters_mask(self):
        """
        Given a scenario (UMi, UMa, RMa), the number of clusters is different
        for different state of UT-BS links (LoS, NLoS, indoor).

        Because we use tensors with predefined dimension size (not ragged), the
        cluster dimension is always set to the maximum number of clusters the
        scenario requires. A mask is then used to discard not required tensors,
        depending on the state of each UT-BS link.

        This function computes and stores this mask of size
        [batch size, number of BSs, number of UTs, maximum number of cluster]
        where an element equals 0 if the cluster is used, 1 otherwise.
        """

        scenario = self._scenario
        num_clusters_los = scenario.num_clusters_los
        num_clusters_nlos = scenario.num_clusters_nlos
        num_clusters_o2i = scenario.num_clusters_indoor
        num_clusters_max = torch.max([num_clusters_los, num_clusters_nlos,
            num_clusters_o2i])


        # Initialize an empty mask
        mask = torch.zeros(shape=[scenario.batch_size, scenario.num_bs,
            scenario.num_ut, num_clusters_max],
            dtype=self._real_dtype)

        # Indoor mask
        mask_indoor = torch.cat((torch.zeros([num_clusters_o2i],
                                          self._real_dtype),
                                 torch.ones([num_clusters_max-num_clusters_o2i],
                                    self._real_dtype)), dim=0)
        mask_indoor = torch.reshape(mask_indoor, [1, 1, 1, num_clusters_max])
        indoor = torch.unsqueeze(scenario.indoor, dim=1) # Broadcasting with BS
        o2i_slice_mask = indoor.to(self._real_dtype)
        o2i_slice_mask = torch.unsqueeze(o2i_slice_mask, dim=3)
        mask = mask + o2i_slice_mask*mask_indoor

        # LoS
        mask_los = torch.cat([torch.zeros([num_clusters_los],
            self._real_dtype),
            torch.ones([num_clusters_max-num_clusters_los],
            self._real_dtype)], dim=0)
        mask_los = torch.reshape(mask_los, [1, 1, 1, num_clusters_max])
        los_slice_mask = scenario.los
        los_slice_mask = los_slice_mask.to(self._real_dtype)
        los_slice_mask = torch.unsqueeze(los_slice_mask, dim=3)
        mask = mask + los_slice_mask*mask_los

        # NLoS
        mask_nlos = torch.cat([torch.zeros([num_clusters_nlos],
            self._real_dtype),
            torch.ones([num_clusters_max-num_clusters_nlos],
            self._real_dtype)], dim=0)
        mask_nlos = torch.reshape(mask_nlos, [1, 1, 1, num_clusters_max])
        nlos_slice_mask = torch.logical_and(torch.logical_not(scenario.los),
            torch.logical_not(indoor))
        nlos_slice_mask = nlos_slice_mask.to(self._real_dtype)
        nlos_slice_mask = torch.unsqueeze(nlos_slice_mask, dim=3)
        mask = mask + nlos_slice_mask*mask_nlos

        # Save the mask
        self._cluster_mask = mask

    def _cluster_delays(self, delay_spread, rician_k_factor):
        # pylint: disable=line-too-long
        """
        Generate cluster delays.
        See step 5 of section 7.5 from TR 38.901 specification.

        Input
        ------
        delay_spread : [batch size, num of BSs, num of UTs], torch.float
            RMS delay spread of each BS-UT link.

        rician_k_factor : [batch size, num of BSs, num of UTs], torch.float
            Rician K-factor of each BS-UT link. Used only for LoS links.

        Output
        -------
        delays : [batch size, num of BSs, num of UTs, maximum number of clusters], torch.float
            Path delays [s]

        unscaled_delays [batch size, num of BSs, num of UTs, maximum number of clusters], torch.float
            Unscaled path delays [s]
        """

        scenario = self._scenario

        batch_size = scenario.batch_size
        num_bs = scenario.num_bs
        num_ut = scenario.num_ut

        num_clusters_max = scenario.num_clusters_max

        # Getting scaling parameter according to each BS-UT link scenario
        delay_scaling_parameter = scenario.get_param("rTau")
        delay_scaling_parameter = torch.unsqueeze(delay_scaling_parameter,
            dim=3)

        # Generating random cluster delays
        # We don't start at 0 to avoid numerical errors
        delay_spread = torch.unsqueeze(delay_spread, dim=3)
        x = torch.rand(size=[batch_size, num_bs, num_ut,
            num_clusters_max],dtype=self._real_dtype)*(1.0-1e-6)+1e-6

        # Moving to linear domain
        unscaled_delays = -delay_scaling_parameter*delay_spread*torch.math.log(x)
        # Forcing the cluster that should not exist to huge delays (1s)
        unscaled_delays = (unscaled_delays*(1.-self._cluster_mask)
            + self._cluster_mask)

        # Normalizing and sorting the delays
        unscaled_delays = unscaled_delays - torch.min(unscaled_delays,
            dim=3, keepdims=True)
        unscaled_delays = torch.sort(unscaled_delays, dim=3)

        # Additional scaling applied to LoS links
        rician_k_factor_db = 10.0*log10(rician_k_factor) # to dB
        scaling_factor = (0.7705 - 0.0433*rician_k_factor_db
            + 0.0002*torch.square(rician_k_factor_db)
            + 0.000017*torch.math.pow(rician_k_factor_db, torch.tensor(3.,
            self._real_dtype)))
        scaling_factor = torch.unsqueeze(scaling_factor, dim=3)
        delays = torch.where(torch.unsqueeze(scenario.los, dim=3),
            unscaled_delays / scaling_factor, unscaled_delays)

        return delays, unscaled_delays

