import torch

try:
    import my_code.mysionna
except ImportError as e:
    import sys
    sys.path.append("../")

# Check if GPU is available
if torch.cuda.is_available():
    # Number of GPUs available
    num_gpus = torch.cuda.device_count()
    print('Number of GPUs available:', num_gpus)

    # Specify which GPU to use
    gpu_num = 0 # Number of the GPU to be used

    if gpu_num < num_gpus:
        device = torch.device(f'cuda:{gpu_num}')
        print(f'Using GPU number {gpu_num}.')

        # Set the specified GPU as the current device
        torch.cuda.set_device(device)

        # This functionality is built-in in PyTorch and does not need to be set manually.
        # Memory growth equivalent in PyTorch is handled by default.
    else:
        print(f'GPU number {gpu_num} is not available. Using CPU.')
        device = torch.device('cpu')
else:
    print('No GPU available. Using CPU.')
    device = torch.device('cpu')

import unittest
import numpy as np
import my_code.mysionna as sionna
from mytest.unit.channel.channel_test_utils import *
from scipy.stats import kstest

from  my_code.mysionna.channel.torch_version.tr38901 import antenna
from  my_code.mysionna.channel.torch_version.tr38901.lsp import LSP,LSPGenerator
from  my_code.mysionna.channel.torch_version.tr38901.rma import RMa,RMaScenario
from  my_code.mysionna.channel.torch_version.tr38901.rays import Rays,RaysGenerator
from my_code.mysionna.channel.torch_version.tr38901.umi import UMi,UMiScenario
from my_code.mysionna.channel.torch_version.tr38901.uma import UMa,UMaScenario
class TestRays(unittest.TestCase):
    r"""Test the rays generated for 3GPP system level simulations
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 3e-2

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 35.0


    def setUpClass():
        r"""Sample rays from all LoS and NLoS channel models for testing"""

        # Forcing the seed to make the tests deterministic
        torch.manual_seed(42)
        np.random.seed(42)

        batch_size = TestRays.BATCH_SIZE
        fc = TestRays.CARRIER_FREQUENCY

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        bs_array = antenna.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='single',
                                                    polarization_type='V',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=torch.complex128)
        ut_array = antenna.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='single',
                                                    polarization_type='V',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=torch.complex128)
        

        # The following quantities have no impact on the rays, but are
        # required to instantiate models
        ut_orientations = torch.rand([batch_size, 1, 3],
                                            dtype=torch.float64)*(2*sionna.PI)+sionna.PI
        bs_orientations = torch.rand([batch_size, 1, 3],
                                            dtype=torch.float64)*(2*sionna.PI)+sionna.PI
        ut_velocities = torch.rand([batch_size, 1, 3], 
                                            dtype=torch.float64)*2+1

        # 1 UT and 1 BS
        ut_loc = generate_random_loc(batch_size, 1, (100,2000), (100,2000),
                                        (1.5, 1.5), share_loc=True,
                                        dtype=torch.float64)
        bs_loc = generate_random_loc(batch_size, 1, (0,100), (0,100),
                                        (35.0, 35.0), share_loc=True,
                                        dtype=torch.float64)
        
        # Force the LSPs
        TestRays.ds = np.power(10.0, -7.49)
        ds_ = torch.full([batch_size, 1, 1], TestRays.ds, dtype=torch.float64)

        TestRays.asd = np.power(10.0, 0.90)
        asd_ = torch.full([batch_size, 1, 1], TestRays.asd, dtype=torch.float64)

        TestRays.asa = np.power(10.0, 1.52)
        asa_ = torch.full([batch_size, 1, 1], TestRays.asa, dtype=torch.float64)

        TestRays.zsa = np.power(10.0, 0.47)
        zsa_ = torch.full([batch_size, 1, 1], TestRays.zsa, dtype=torch.float64)

        TestRays.zsd = np.power(10.0, -0.29)
        zsd_ = torch.full([batch_size, 1, 1], TestRays.zsd, dtype=torch.float64)

        TestRays.k = np.power(10.0, 7. / 10.)
        k_ = torch.full([batch_size, 1, 1], TestRays.k, dtype=torch.float64)

        sf_ = torch.zeros([batch_size, 1, 1], dtype=torch.float64)

        # 创建 LSP
        lsp = LSP(ds_, asd_, asa_, sf_, k_, zsa_, zsd_)

        # Store the sampled rays
        TestRays.delays = {}
        TestRays.powers = {}
        TestRays.aoa = {}
        TestRays.aod = {}
        TestRays.zoa = {}
        TestRays.zod = {}
        TestRays.xpr = {}
        TestRays.num_clusters = {}
        TestRays.los_aoa = {}
        TestRays.los_aod = {}
        TestRays.los_zoa = {}
        TestRays.los_zod = {}
        TestRays.mu_log_zsd = {}

        #################### RMa
        TestRays.delays['rma'] = {}
        TestRays.powers['rma'] = {}
        TestRays.aoa['rma'] = {}
        TestRays.aod['rma'] = {}
        TestRays.zoa['rma'] = {}
        TestRays.zod['rma'] = {}
        TestRays.xpr['rma'] = {}
        TestRays.num_clusters['rma'] = {}
        TestRays.los_aoa['rma'] = {}
        TestRays.los_aod['rma'] = {}
        TestRays.los_zoa['rma'] = {}
        TestRays.los_zod['rma'] = {}
        TestRays.mu_log_zsd['rma'] = {}
        scenario = RMaScenario(fc, ut_array, bs_array,
                                    "downlink",
                                    dtype=torch.complex128)
        ray_sampler = RaysGenerator(scenario)

        #### LoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                                ut_velocities, in_state, los=True)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['rma']['los'] = rays.delays.squeeze().numpy()
        TestRays.powers['rma']['los'] = rays.powers.squeeze().numpy()
        TestRays.aoa['rma']['los'] = rays.aoa.squeeze().numpy()
        TestRays.aod['rma']['los'] = rays.aod.squeeze().numpy()
        TestRays.zoa['rma']['los'] = rays.zoa.squeeze().numpy()
        TestRays.zod['rma']['los'] = rays.zod.squeeze().numpy()
        TestRays.xpr['rma']['los'] = rays.xpr.squeeze().numpy()
        TestRays.num_clusters['rma']['los'] = 11
        TestRays.los_aoa['rma']['los'] = scenario.los_aoa.numpy()
        TestRays.los_aod['rma']['los'] = scenario.los_aod.numpy()
        TestRays.los_zoa['rma']['los'] = scenario.los_zoa.numpy()
        TestRays.los_zod['rma']['los'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['rma']['los'] = scenario.lsp_log_mean[:,0,0,6].numpy()


        #################### UMi
        TestRays.delays['umi'] = {}
        TestRays.powers['umi'] = {}
        TestRays.aoa['umi'] = {}
        TestRays.aod['umi'] = {}
        TestRays.zoa['umi'] = {}
        TestRays.zod['umi'] = {}
        TestRays.xpr['umi'] = {}
        TestRays.num_clusters['umi'] = {}
        TestRays.los_aoa['umi'] = {}
        TestRays.los_aod['umi'] = {}
        TestRays.los_zoa['umi'] = {}
        TestRays.los_zod['umi'] = {}
        TestRays.mu_log_zsd['umi'] = {}
        scenario = UMiScenario(  fc, 'low',
                                    ut_array, bs_array,
                                    "downlink",
                                    dtype=torch.complex128)
        ray_sampler = RaysGenerator(scenario)       

        # LoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                            ut_velocities, in_state, los=True)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['umi']['los'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['umi']['los'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['umi']['los'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['umi']['los'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['umi']['los'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['umi']['los'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['umi']['los'] = torch.squeeze(rays.xpr).numpy()

        TestRays.num_clusters['umi']['los'] = 12
        TestRays.los_aoa['umi']['los'] = scenario.los_aoa.numpy()
        TestRays.los_aod['umi']['los'] = scenario.los_aod.numpy()
        TestRays.los_zoa['umi']['los'] = scenario.los_zoa.numpy()
        TestRays.los_zod['umi']['los'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['umi']['los'] = scenario.lsp_log_mean[:, 0, 0, 6].numpy()


        # NLoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                            ut_velocities, in_state, los=False)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['umi']['nlos'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['umi']['nlos'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['umi']['nlos'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['umi']['nlos'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['umi']['nlos'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['umi']['nlos'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['umi']['nlos'] = torch.squeeze(rays.xpr).numpy()

        TestRays.num_clusters['umi']['nlos'] = 19
        TestRays.los_aoa['umi']['nlos'] = scenario.los_aoa.numpy()
        TestRays.los_aod['umi']['nlos'] = scenario.los_aod.numpy()
        TestRays.los_zoa['umi']['nlos'] = scenario.los_zoa.numpy()
        TestRays.los_zod['umi']['nlos'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['umi']['nlos'] = scenario.lsp_log_mean[:, 0, 0, 6].numpy()

        #### O2I
        in_state = generate_random_bool(batch_size, 1, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['umi']['o2i'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['umi']['o2i'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['umi']['o2i'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['umi']['o2i'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['umi']['o2i'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['umi']['o2i'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['umi']['o2i'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['umi']['o2i'] = 12
        TestRays.los_aoa['umi']['o2i'] = scenario.los_aoa.numpy()
        TestRays.los_aod['umi']['o2i'] = scenario.los_aod.numpy()
        TestRays.los_zoa['umi']['o2i'] = scenario.los_zoa.numpy()
        TestRays.los_zod['umi']['o2i'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['umi']['o2i'] = scenario.lsp_log_mean[:,0,0,6].numpy()


        #################### UMa
        TestRays.delays['uma'] = {}
        TestRays.powers['uma'] = {}
        TestRays.aoa['uma'] = {}
        TestRays.aod['uma'] = {}
        TestRays.zoa['uma'] = {}
        TestRays.zod['uma'] = {}
        TestRays.xpr['uma'] = {}
        TestRays.num_clusters['uma'] = {}
        TestRays.los_aoa['uma'] = {}
        TestRays.los_aod['uma'] = {}
        TestRays.los_zoa['uma'] = {}
        TestRays.los_zod['uma'] = {}
        TestRays.mu_log_zsd['uma'] = {}
        scenario = UMaScenario(  fc, 'low',
                                    ut_array, bs_array,
                                    "downlink",
                                    dtype=torch.complex128)
        ray_sampler = RaysGenerator(scenario)

        #### LoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=True)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['uma']['los'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['los'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['los'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['los'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['los'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['los'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['los'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['los'] = 12
        TestRays.los_aoa['uma']['los'] = scenario.los_aoa.numpy()
        TestRays.los_aod['uma']['los'] = scenario.los_aod.numpy()
        TestRays.los_zoa['uma']['los'] = scenario.los_zoa.numpy()
        TestRays.los_zod['uma']['los'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['uma']['los'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### NLoS
        in_state = generate_random_bool(batch_size, 1, 0.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state, los=False)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['uma']['nlos'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['nlos'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['nlos'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['nlos'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['nlos'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['nlos'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['nlos'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['nlos'] = 20
        TestRays.los_aoa['uma']['nlos'] = scenario.los_aoa.numpy()
        TestRays.los_aod['uma']['nlos'] = scenario.los_aod.numpy()
        TestRays.los_zoa['uma']['nlos'] = scenario.los_zoa.numpy()
        TestRays.los_zod['uma']['nlos'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['uma']['nlos'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        #### O2I
        in_state = generate_random_bool(batch_size, 1, 1.0)
        scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                            ut_velocities, in_state)
        ray_sampler.topology_updated_callback()
        rays = ray_sampler(lsp)
        TestRays.delays['uma']['o2i'] = torch.squeeze(rays.delays).numpy()
        TestRays.powers['uma']['o2i'] = torch.squeeze(rays.powers).numpy()
        TestRays.aoa['uma']['o2i'] = torch.squeeze(rays.aoa).numpy()
        TestRays.aod['uma']['o2i'] = torch.squeeze(rays.aod).numpy()
        TestRays.zoa['uma']['o2i'] = torch.squeeze(rays.zoa).numpy()
        TestRays.zod['uma']['o2i'] = torch.squeeze(rays.zod).numpy()
        TestRays.xpr['uma']['o2i'] = torch.squeeze(rays.xpr).numpy()
        TestRays.num_clusters['uma']['o2i'] = 12
        TestRays.los_aoa['uma']['o2i'] = scenario.los_aoa.numpy()
        TestRays.los_aod['uma']['o2i'] = scenario.los_aod.numpy()
        TestRays.los_zoa['uma']['o2i'] = scenario.los_zoa.numpy()
        TestRays.los_zod['uma']['o2i'] = scenario.los_zod.numpy()
        TestRays.mu_log_zsd['uma']['o2i'] = scenario.lsp_log_mean[:,0,0,6].numpy()

        ###### General
        TestRays.d_2d = scenario.distance_2d[0,0,0].numpy()


    @channel_test_on_models(('rma', 'umi', 'uma'), ('los', 'nlos', 'o2i'))
    def test_delays(self, model, submodel):
        """Test ray generation: Delays"""
        num_clusters = TestRays.num_clusters[model][submodel]
        batch_size = TestRays.BATCH_SIZE
        tau = TestRays.delays[model][submodel][:,:num_clusters].flatten()
        _, ref_tau = delays(model, submodel, batch_size, num_clusters,
                                    TestRays.ds, TestRays.k)
        ref_tau = ref_tau[:,:num_clusters].flatten()
        D,_ = kstest(tau,ref_tau)
        self.assertLessEqual(D, TestRays.MAX_ERR, f"{model}:{submodel}")


if __name__ == '__main__':
    unittest.main()