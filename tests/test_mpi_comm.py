import pytest
from commi import MPICommunicator, Status

@pytest.mark.mpi
@pytest.mark.parametrize("use_status", [False, True])
def test_mpi_sendrecv(use_status):
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    assert mpi_comm.Get_size() == 2
    commi_comm = MPICommunicator(mpi_comm)
    rank = commi_comm.Get_rank()
    partner_rank = int(not rank)

    if use_status:
        st = Status()
    else:
        st = None

    if rank == 0:
        commi_comm.send(1, dest=partner_rank, tag=0)
        val = commi_comm.recv(source=partner_rank, tag=1, status=st)
        assert val == 12345
        if use_status:
            assert st.COMMI_TAG == 1
            assert st.COMMI_SOURCE == partner_rank
            assert st.count == 15
            assert st.cancelled == 0

    else:
        val = commi_comm.recv(source=partner_rank, tag=0, status=st)
        commi_comm.send(12345, dest=partner_rank, tag=1)
        assert val == 1
        if use_status:
            assert st.COMMI_TAG == 0
            assert st.COMMI_SOURCE == partner_rank
            assert st.count == 5
            assert st.cancelled == 0


@pytest.mark.mpi
@pytest.mark.parametrize("use_status", [False, True])
def test_mpi_sendrecv_np(use_status):
    pytest.importorskip("numpy")
    import numpy as np
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    commi_comm = MPICommunicator(mpi_comm)
    if use_status:
        st = Status()
    else:
        st = None

    rank = commi_comm.Get_rank()
    partner_rank = int(not rank)
    data_1 = np.arange(0, 10)
    data_2 = np.arange(10, 0, -1)
    if rank == 0:
        commi_comm.Send(data_1, dest=partner_rank, tag=0)
        commi_comm.Recv(data_2, source=partner_rank, tag=1, status=st)
        assert np.allclose(data_1, data_2)
        if use_status:
            assert st.COMMI_TAG == 1
            assert st.COMMI_SOURCE == partner_rank
            # 80, not ten because unit is bytes
            assert st.count == 80
            assert st.cancelled == 0
    else:
        data_recv = np.arange(20, 30)
        commi_comm.Recv(data_recv, source=partner_rank, tag=0, status=st)
        commi_comm.Send(data_1, dest=partner_rank, tag=1)
        assert np.allclose(data_recv, data_1)
        if use_status:
            assert st.COMMI_TAG == 0
            assert st.COMMI_SOURCE == partner_rank
            # 80, not ten because unit is bytes
            assert st.count == 80
            assert st.cancelled == 0


@pytest.mark.mpi
def test_mpi_isendrecv():
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    assert mpi_comm.Get_size() == 2
    commi_comm = MPICommunicator(mpi_comm)
    rank = commi_comm.Get_rank()
    partner_rank = int(not rank)
    req1 = req2 = None
    if rank == 0:
        req1 = commi_comm.isend(1, dest=partner_rank, tag=0)
        req2 = commi_comm.irecv(source=partner_rank, tag=1)
        req1.wait()
        val = req2.wait()
        assert val == 12345
    else:
        req1 = commi_comm.irecv(source=partner_rank, tag=0)
        req2 = commi_comm.isend(12345, dest=partner_rank, tag=1)

        val = req1.wait()
        req2.wait()
        assert val == 1

@pytest.mark.mpi
def test_mpi_isendrecv_np():
    pytest.importorskip("numpy")
    import numpy as np
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    commi_comm = MPICommunicator(mpi_comm)

    rank = commi_comm.Get_rank()
    partner_rank = int(not rank)
    data_1 = np.arange(0, 10)
    data_2 = np.arange(10, 0, -1)
    req1 = req2 = None
    if rank == 0:
        req1 = commi_comm.Isend(data_1, dest=partner_rank, tag=0)
        req2 = commi_comm.Irecv(data_2, source=partner_rank, tag=1)
        req1.wait()
        req2.wait()
        assert np.allclose(data_1, data_2)
    else:
        data_recv = np.arange(20, 30)
        req1 = commi_comm.Irecv(data_recv, source=partner_rank, tag=0)
        req2 = commi_comm.Isend(data_1, dest=partner_rank, tag=1)
        req1.wait()
        req2.wait()
        assert np.allclose(data_recv, data_1)

def test_status_conversion_copy():
    from commi.mpi_communicator import _wrap_status
    from mpi4py import MPI
    st_mpi = MPI.Status()
    st_commi = Status(st_mpi.Get_count(),
                      st_mpi.Is_cancelled(),
                      st_mpi.Get_source(),
                      st_mpi.Get_tag(),
                      st_mpi.Get_error()
                      )

    conv_status = _wrap_status(st_mpi)
    assert conv_status == st_commi
