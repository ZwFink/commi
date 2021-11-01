class Communicator:
    def __init__(self):
        pass

    def send(self, data, dest=0, tag=0):
        pass

    def recv(self, data=None, src=0, tag=0):
        pass

    def Send(self, data, dest=0, tag=0):
        pass

    def Recv(self, data=None, dest=0, tag=0):
        pass

    def Isend(self, data, dest=0, tag=0):
        pass

    def Irecv(self, data=None, dest=0, tag=0):
        pass

    def isend(self, data, dest=0, tag=0):
        pass

    def irecv(self, data=None, dest=0, tag=0):
        pass

    def Get_rank(self):
        pass

    def get_communicator(self):
        pass

class Request:
    def __init__(self):
        pass
    def wait(self):
        pass
