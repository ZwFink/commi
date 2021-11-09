from abc import ABC, abstractmethod

class Communicator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def send(self, data, dest=0, tag=0):
        pass

    @abstractmethod
    def recv(self, data=None, src=0, tag=0):
        pass

    @abstractmethod
    def Send(self, data, dest=0, tag=0):
        pass

    @abstractmethod
    def Recv(self, data=None, dest=0, tag=0):
        pass

    @abstractmethod
    def Isend(self, data, dest=0, tag=0):
        pass

    @abstractmethod
    def Irecv(self, data=None, dest=0, tag=0):
        pass

    @abstractmethod
    def isend(self, data, dest=0, tag=0):
        pass

    @abstractmethod
    def irecv(self, data=None, dest=0, tag=0):
        pass

    @abstractmethod
    def Get_rank(self):
        pass

    @abstractmethod
    def get_communicator(self):
        pass

class Request:
    def __init__(self):
        pass
    def wait(self):
        pass
