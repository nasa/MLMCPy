import os
import sys
import imp
import pytest
import numpy as np

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.input import RandomInput

def determine_num_cpu_samples(total_num_samples, rank, size):
    """
        Borrowed from MLMC class to test parallel sampling below...
        Determines number of samples to be run on current cpu based on
        total number of samples to be run.
        :param total_num_samples: Total samples to be taken.
        :return: Samples to be taken by this cpu.
    """
    num_cpu_samples = total_num_samples // size
    num_residual_samples = total_num_samples - \
        num_cpu_samples * size

    if rank < num_residual_samples:
        num_cpu_samples += 1

    return num_cpu_samples

@pytest.fixture
def uniform_distribution_input():
    """
    Creates a RandomInput object that produces samples from a
    uniform distribution.
    """
    return RandomInput(np.random.uniform)


@pytest.fixture
def beta_distribution_input():
    """
    Creates a RandomInput object that produces samples from a
    beta distribution.
    """
    np.random.seed(1)

    def beta_distribution(shift, scale, alpha, beta, size):
        return shift + scale * np.random.beta(alpha, beta, size)

    return RandomInput(distribution_function=beta_distribution,
                       shift=1.0, scale=2.5, alpha=3., beta=2.)


@pytest.fixture
def comm():
    """
    Creates a MPI.COMM_WORLD object for working with multi-process information.
    """
    imp.find_module('mpi4py')

    from mpi4py import MPI
    return MPI.COMM_WORLD


def test_init_invalid_input():
    """
    Ensure an exception is raised if an object that is not a function is
    provided.
    """
    with pytest.raises(TypeError):
        RandomInput(1)


def test_draw_samples_expected_output(uniform_distribution_input):
    """
    Ensure outputs from draw_samples matches expected type and shape.
    """
    for num_samples in range(1, 10):

        sample = uniform_distribution_input.draw_samples(num_samples)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (num_samples, 1)


def test_exception_invalid_distribution_function():
    """
    Ensure an exception is thrown if the provided function does not return
    the expected data type.
    """
    def invalid_distribution_function():
        return [1, 2, 3]

    with pytest.raises(TypeError):
        invalid_sampler = RandomInput(invalid_distribution_function)
        invalid_sampler.draw_samples(5)


def test_extra_distribution_function_parameters():
    """
    Test ability to specify optional distribution function parameters.
    """
    np.random.seed(1)

    normal_sampler = RandomInput(np.random.normal, loc=1.)
    sample = normal_sampler.draw_samples(100)

    assert isinstance(sample, np.ndarray)
    assert sample.shape == (100, 1)
    assert np.abs(np.mean(sample) - 1.) < .2


@pytest.mark.parametrize('argument', [1., 0, 'a'])
def test_draw_samples_invalid_arguments(uniform_distribution_input, argument):
    """
    Ensure an exception occurs if invalid parameters are passed to draw_samples.
    """
    with pytest.raises(Exception):
        uniform_distribution_input.draw_samples(argument)


def test_distribution_exception_if_size_parameter_not_accepted():
    """
    Ensure an exception occurs if the distribution function does not accept
    a size parameter.
    """
    def invalid_distribution_function():
        return np.zeros(5)

    invalid_input = \
        RandomInput(distribution_function=invalid_distribution_function)

    with pytest.raises(TypeError):
        invalid_input.draw_samples(10)


@pytest.mark.parametrize('random_input', [uniform_distribution_input(),
                                          beta_distribution_input()],
                         ids=['uniform', 'beta'])
def test_multi_cpu_sampling(random_input):

    imp.find_module('mpi4py')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Get per process samples, then aggregate them to compare to single cpu
    # sampling.
    np.random.seed(1)
    this_test_sample = random_input.draw_samples(10)

    test_sample_list = comm.allgather(this_test_sample)

    test_samples = np.zeros((10 * comm.size, 1))
    for i, sample in enumerate(test_sample_list):
        test_samples[i::comm.size] = sample

    # Get samples that would be returned in single cpu environment.
    random_input._num_cpus = 1
    random_input._cpu_rank = 0

    np.random.seed(1)
    baseline_samples = random_input.draw_samples(10 * comm.size)

    assert np.array_equal(test_samples, baseline_samples)

def test_back_to_back_parallel_sampling(beta_distribution_input):
    '''
    Draw N0 samples, then N1 samples in parallel and verify that the means
    match reference values from serial sampling
    '''

    imp.find_module('mpi4py')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    N0 = 19
    N1 = 17

    mean_ref_0 = 2.6737292217016666
    mean_ref_1 = 2.528881234826408    

    #Draw samples on each proc
    num_samples_0 = determine_num_cpu_samples(N0, comm.rank, comm.size)
    samples_0 = np.array(beta_distribution_input.draw_samples(num_samples_0))
    num_samples_1 = determine_num_cpu_samples(N1, comm.rank, comm.size)
    samples_1 = np.array(beta_distribution_input.draw_samples(num_samples_1))

    #Collect samples on the root proc
    samples_recv_0 = None  
    samples_recv_1 = None  

    sendcounts = np.array(comm.gather(len(samples_0), 0))
    if comm.rank == 0:
        samples_recv_0 = np.empty(sum(sendcounts), dtype=float)
    comm.Gatherv(sendbuf=samples_0, recvbuf=(samples_recv_0, sendcounts),root=0)

    sendcounts = np.array(comm.gather(len(samples_1), 0))
    if comm.rank == 0:
        samples_recv_1 = np.empty(sum(sendcounts), dtype=float)
    comm.Gatherv(sendbuf=samples_1, recvbuf=(samples_recv_1, sendcounts),root=0)

    #Check means vs reference:
    if comm.rank == 0:
        assert np.mean(samples_recv_0) == mean_ref_0
        assert np.mean(samples_recv_1) == mean_ref_1

