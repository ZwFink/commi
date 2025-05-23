import ctypes

class LCG:
    def __init__(self):
        self.lib = ctypes.CDLL('./librandomdraw.so')
        self.lib.LCG_Create.restype = ctypes.c_void_p
        self.lib.random_draw.restype = ctypes.c_ulonglong
        self.lcg_ptr = ctypes.c_void_p(self.lib.LCG_Create())
        self.init()

    def __del__(self):
        self.lib.LCG_Destroy(self.lcg_ptr)
    def init(self):
        self.lib.LCG_init(self.lcg_ptr)
    def jump(self, m, bound):
        c_m = ctypes.c_ulonglong(m)
        c_bound = ctypes.c_ulonglong(bound)
        self.lib.LCG_jump(c_m, c_bound, self.lcg_ptr)
    def random_draw(self, mu):
        c_mu = ctypes.c_double(mu)
        random_val = self.lib.random_draw(c_mu, self.lcg_ptr)
        return ctypes.c_ulonglong(random_val).value